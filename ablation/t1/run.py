"""Isolated, serial training and frozen evaluation for the simple T1 study."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "dvsrc").is_dir())
STUDY = ROOT / "ablation/t1"
sys.path.insert(0, str(ROOT))

from dvsrc.config import ExperimentConfig
from dvsrc.utils import atomic_json, sha256_file


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_metrics(metrics):
    for protocol in ("t1_1v1", "t1_5v1"):
        for name in ("accuracy", "eer", "roc_auc"):
            value = metrics[protocol]["overall"][name]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {protocol} {name}")


def matrix():
    return read(STUDY / "matrix.json")


def run_dir(setting):
    return ROOT / matrix()["runs_root"] / setting["id"] / "seed_42"


def config_for(setting):
    config = ExperimentConfig.from_yaml(STUDY / setting["config"])
    config.train.output_dir = str(run_dir(setting).resolve())
    return config


def config_data(config):
    return json.loads(json.dumps(config.to_dict()))


def validate_configs():
    m = matrix()
    if m["seed"] != 42 or m["baseline_id"] != "S2F0":
        raise ValueError("This study requires S2F0 and seed 42")
    expected_changes = {
        "S2C0": {"t1_sequence_residual": False, "t1_pair_stabilize": False},
        "S2F0": {}, "S2M1": {"fusion": "sequence_only"},
        "S2M2": {"fusion": "image_only"}, "S2F1": {"fusion": "late"},
        "S2P1": {"t1_pair_features": "diff_only"},
        "S2P2": {"t1_pair_features": "product_only"},
        "S2G1": {"t1_set_aggregation": "mean"},
    }
    if [s["id"] for s in m["settings"]] != list(expected_changes):
        raise ValueError("Expected the eight preregistered simple T1 settings")
    base = config_data(ExperimentConfig.from_yaml(ROOT / "configs/t1_simple_v2_5090.yaml"))
    for setting in m["settings"]:
        config = config_for(setting)
        actual = config_data(config)
        expected = json.loads(json.dumps(base))
        expected["model"].update(expected_changes[setting["id"]])
        expected["train"]["output_dir"] = actual["train"]["output_dir"]
        if setting["change"] != expected_changes[setting["id"]] or actual != expected:
            raise ValueError(f"Unexpected config change for {setting['id']}")
        if not (config.model.t1_variant == "simple_v2" and config.model.use_tsa
                and not config.model.use_qrsa and not config.model.use_t1_residual_verification
                and not config.model.use_t1_evidence_anchor
                and config.train.t1_genuine_identity_only
                and config.train.seed == 42 and config.train.stage_a_epochs == 10
                and config.train.stage_a_selection_policy == "dual_t1"
                and not config.train.stage_a_finalize_test
                and not config.train.stage_a_evaluate_test_each_epoch
                and config.data.benchmark_root == m["benchmark_root"]):
            raise ValueError(f"Simple T1 experiment contract violated: {setting['id']}")


def verify_benchmark():
    m = matrix()
    audit = read(ROOT / m["benchmark_root"] / "audit_report.json")
    episodes = ROOT / m["benchmark_root"] / "episodes/fold_0"
    required = {f"{split}_{protocol}.jsonl" for split in ("train", "val", "test")
                for protocol in ("t1_1v1", "t1_5v1")}
    if not all((episodes / name).is_file() for name in required):
        raise ValueError("Missing frozen episode files")
    if not audit.get("ok") or audit.get("split_digest") != m["split_digest"]:
        raise ValueError("Frozen benchmark audit/split mismatch")


def identity(setting):
    return {
        "study_id": matrix()["study_id"], "experiment_id": setting["id"], "seed": 42,
        "evaluation_version": "t1_raw_logit_roc_v2",
        "runner_sha256": sha256_file(Path(__file__)),
        "dataset_manifest_sha256": sha256_file(ROOT / config_for(setting).data.dataset_root / "manifest.json"),
        "episode_sha256": {
            p.name: sha256_file(p) for p in sorted(
                (ROOT / matrix()["benchmark_root"] / "episodes/fold_0").glob("*.jsonl")
            )
        },
        "config": config_data(config_for(setting)),
        "matrix_sha256": sha256_file(STUDY / "matrix.json"),
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): sha256_file(p)
            for p in sorted((ROOT / "dvsrc").glob("*.py"))
        },
        "benchmark_audit_sha256": sha256_file(
            ROOT / matrix()["benchmark_root"] / "audit_report.json"
        ),
    }


def verify_identity(setting):
    saved = read(run_dir(setting) / "run_manifest.json")
    if saved["identity"] != identity(setting):
        raise ValueError(f"Run config, source, or benchmark changed: {setting['id']}")


@contextmanager
def study_lock():
    # Kernel locks release automatically on process exit, including abnormal exit.
    with (STUDY / "execution.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def require_gpu():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 30000 * 1024**2:
        raise RuntimeError("The 5090 profile requires a visible CUDA GPU with at least 30000 MiB")


def finalize_validation(setting):
    import torch
    output = run_dir(setting)
    verify_identity(setting)
    status = read(output / "t1_training_status.json")
    if status.get("running") is not False or status.get("test_executed") is not False:
        raise ValueError("Training must finish as Validation-only before freeze")
    if (output / "test").exists():
        raise ValueError("Cannot finalize Validation after a Test directory was created")
    best = output / "best_t1.pt"
    state = torch.load(best, map_location="cpu", weights_only=False)
    if json.loads(json.dumps(state["config"])) != config_data(config_for(setting)):
        raise ValueError("Checkpoint config differs from the preregistered run")
    selected = state["validation"]
    validate_metrics(selected)
    atomic_json(output / "validation_selected.json", selected)
    calibration = output / "t1_calibration.json"
    values = read(calibration)
    for protocol in ("t1_1v1", "t1_5v1"):
        if not (math.isfinite(values[protocol + "_temperature"])
                and values[protocol + "_temperature"] > 0
                and 0 <= values[protocol + "_threshold"] <= 1):
            raise ValueError("Invalid Validation calibrator")
    record = {
        "id": setting["id"], "seed": 42, "stage_epoch": state.get("stage_epoch"),
        "checkpoint": best.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256_file(best),
        "calibration": calibration.relative_to(ROOT).as_posix(),
        "calibration_sha256": sha256_file(calibration),
        "config": config_data(config_for(setting)),
        "validation_only": True,
    }
    atomic_json(output / "validation_checkpoint.json", record)
    return record


def verify_record(setting, record):
    verify_identity(setting)
    if record.get("id") != setting["id"] or record.get("seed") != 42:
        raise ValueError("Frozen identity mismatch")
    if record.get("config") != config_data(config_for(setting)) or not record.get("validation_only"):
        raise ValueError("Frozen config mismatch")
    for key, filename in (("checkpoint", "best_t1.pt"), ("calibration", "t1_calibration.json")):
        expected = (run_dir(setting) / filename).resolve()
        if (ROOT / record[key]).resolve() != expected or sha256_file(expected) != record[key + "_sha256"]:
            raise ValueError(f"Frozen {key} changed for {setting['id']}")


def train(settings):
    require_gpu()
    for setting in settings:
        output = run_dir(setting)
        frozen = output / "validation_checkpoint.json"
        if frozen.exists():
            verify_record(setting, read(frozen))
            print(f"SKIP completed Validation: {setting['id']}", flush=True)
            continue
        if (output / "test").exists():
            raise ValueError("Training after Test access is forbidden")
        manifest = output / "run_manifest.json"
        if manifest.exists():
            verify_identity(setting)
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError(f"Unidentified existing run: {output}")
            atomic_json(manifest, {"identity": identity(setting), "created_at_utc": now()})
        status = output / "t1_training_status.json"
        if status.exists() and read(status).get("running") is False:
            finalize_validation(setting)
            continue
        command = [sys.executable, "-u", "-m", "dvsrc.cli", "train-t1", "--config",
                   str(STUDY / setting["config"]), "--output", str(output.resolve())]
        resume = output / "last_t1.pt"
        if resume.exists():
            command.extend(["--resume", str(resume)])
        print(f"TRAIN {setting['id']} resume={resume.exists()}", flush=True)
        with (output / "stdout.log").open("ab") as stdout, (output / "stderr.log").open("ab") as stderr:
            subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
        finalize_validation(setting)


def freeze():
    m = matrix()
    entries = []
    for setting in m["settings"]:
        record = read(run_dir(setting) / "validation_checkpoint.json")
        verify_record(setting, record)
        entries.append(record)
    path = ROOT / m["freeze_manifest"]
    if path.exists():
        saved = read(path)
        if saved.get("study_id") != m["study_id"] or saved["entries"] != entries:
            raise ValueError("Cannot replace the frozen Test matrix")
        return saved
    if any((run_dir(s) / "test").exists() for s in m["settings"]):
        raise ValueError("Test artifacts exist before the matrix freeze")
    saved = {"study_id": m["study_id"], "frozen_at_utc": now(), "entries": entries}
    atomic_json(path, saved)
    return saved


def test(settings):
    import torch
    from dvsrc.calibration import Calibrator
    from dvsrc.trainer import Trainer
    path = ROOT / matrix()["freeze_manifest"]
    if not path.is_file():
        raise ValueError(f"All {len(matrix()['settings'])} Validation checkpoints must be frozen before Test")
    frozen = freeze()  # Validate all identities and file hashes on every invocation.
    require_gpu()
    for setting in settings:
        record = next(e for e in frozen["entries"] if e["id"] == setting["id"])
        output = run_dir(setting) / "test"
        started, completed = output / "test_access_started.json", output / "test_release_manifest.json"
        metrics = output / "test_metrics.json"
        if completed.exists():
            release = read(completed)
            if (release.get("complete") is not True or release["frozen_record"] != record
                    or release["metrics_sha256"] != sha256_file(metrics)):
                raise ValueError("Completed Test artifacts changed")
            print(f"SKIP completed Test: {setting['id']}", flush=True)
            continue
        if started.exists():
            if read(started)["frozen_record"] != record:
                raise ValueError("Only the same frozen checkpoint/calibrator can resume Test")
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError("Unidentified Test artifacts")
            atomic_json(started, {"frozen_record": record, "started_at_utc": now()})
        print(f"TEST {setting['id']}", flush=True)
        if not metrics.exists():
            config = config_for(setting)
            config.train.output_dir = str(output.resolve())
            with (output / "stdout.log").open("a", encoding="utf-8") as stdout, \
                    (output / "stderr.log").open("a", encoding="utf-8") as stderr:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    try:
                        trainer = Trainer(config)
                        trainer.load_checkpoint(ROOT / record["checkpoint"])
                        calibrator = Calibrator(**read(ROOT / record["calibration"]))
                        result = trainer.evaluate_split(
                            "test", calibrator=calibrator, export=True, protocols=("t1_1v1", "t1_5v1"),
                        )
                        validate_metrics(result)
                        atomic_json(metrics, result)
                        del trainer
                    except Exception:
                        traceback.print_exc()
                        raise
            gc.collect()
            torch.cuda.empty_cache()
        validate_metrics(read(metrics))
        atomic_json(completed, {"frozen_record": record, "completed_at_utc": now(),
                                "metrics_sha256": sha256_file(metrics), "complete": True})


def status():
    rows = []
    for setting in matrix()["settings"]:
        output = run_dir(setting)
        state = output / "t1_training_status.json"
        rows.append({"id": setting["id"], "validation_complete": (output / "validation_checkpoint.json").exists(),
                     "test_complete": (output / "test/test_release_manifest.json").exists(),
                     "training": read(state) if state.exists() else None})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def export():
    m = matrix()
    for setting in m["settings"]:
        if not (run_dir(setting) / "test/test_release_manifest.json").is_file():
            raise ValueError(f"Export requires all {len(m['settings'])} Test releases")
    frozen = freeze()
    rows = []
    destination = ROOT / m["results_root"]
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result export: {destination}")
    allowed = {".json", ".jsonl", ".log", ".yaml", ".csv"}
    sources = []
    for setting in m["settings"]:
        output = run_dir(setting)
        release = read(output / "test/test_release_manifest.json")
        record = next(e for e in frozen["entries"] if e["id"] == setting["id"])
        if (release.get("complete") is not True or release["frozen_record"] != record
                or release["metrics_sha256"] != sha256_file(output / "test/test_metrics.json")):
            raise ValueError("Test release integrity mismatch")
        for source in output.rglob("*"):
            if source.is_file() and not source.is_symlink() and source.suffix in allowed:
                relative = Path("runs") / setting["id"] / "seed_42" / source.relative_to(output)
                sources.append((source, relative))
        for split, filename in (("validation_selected", "validation_selected.json"),
                                ("validation_calibrated", "validation_metrics.json"),
                                ("test", "test/test_metrics.json")):
            metrics = read(output / filename)
            validate_metrics(metrics)
            for protocol in ("t1_1v1", "t1_5v1"):
                rows.append({"study_id": m["study_id"], "id": setting["id"], "seed": 42,
                             "split": split, "protocol": protocol, **metrics[protocol]["overall"]})
    target = Path(tempfile.mkdtemp(prefix=".export-", dir=STUDY))
    for source, relative in sources:
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    atomic_json(target / "matrix.json", m)
    atomic_json(target / "test_checkpoint_manifest.json", frozen)
    atomic_json(target / "summary.json", {"study_id": m["study_id"], "status": "complete",
                                          "validation_completed": len(m["settings"]), "test_completed": len(m["settings"]),
                                          "weights_included": False, "evaluation_version": m.get("evaluation_version", "t1_raw_logit_roc_v2"),
                                          "test_reused_after_v1": m.get("test_reused_after_v1", True), "rows": rows})
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (target / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    hashes = {p.relative_to(target).as_posix(): sha256_file(p) for p in target.rglob("*") if p.is_file()}
    atomic_json(target / "file_hashes.json", hashes)
    target.rename(destination)
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "status", "train", "freeze", "test", "export"))
    parser.add_argument("--id", help="Optional single preregistered setting, train/test only")
    args = parser.parse_args()
    os.chdir(ROOT)
    validate_configs()
    if args.id and (args.command not in {"train", "test"} or args.id not in {s["id"] for s in matrix()["settings"]}):
        parser.error("--id requires a registered setting and train/test command")
    if args.command == "check":
        print(f"{len(matrix()['settings'])} T1 configs valid; seed=42; TSA always enabled; no training/Test executed")
        return
    if args.command == "status":
        status()
        return
    verify_benchmark()
    settings = [s for s in matrix()["settings"] if args.id is None or s["id"] == args.id]
    with study_lock():
        if args.command == "train":
            train(settings)
            if args.id is None:
                freeze()
        elif args.command == "freeze":
            freeze()
        elif args.command == "test":
            test(settings)
        else:
            export()


if __name__ == "__main__":
    main()
