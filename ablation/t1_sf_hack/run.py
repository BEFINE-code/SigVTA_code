"""Isolated T1 SF-hack ablation. Same freeze/test policy as ablation/t1."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "dvsrc").is_dir())
STUDY = ROOT / "ablation/t1_sf_hack"
sys.path.insert(0, str(ROOT))


def _load_t1_run():
    path = ROOT / "ablation/t1/run.py"
    spec = importlib.util.spec_from_file_location("t1_simple_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t1run = _load_t1_run()
t1run.ROOT = ROOT
t1run.STUDY = STUDY

from dvsrc.config import ExperimentConfig
from dvsrc.utils import atomic_json, sha256_file

H_T1_LOCAL_OVERRIDES = {
    "data": {"num_workers": 2, "memory_cache_items": 64},
    "model": {"image_chunk_size": 4},
    "train": {
        "batch_t1_1v1": 8, "batch_t1_5v1": 2,
        "stage_a_batch_t1_1v1": 8, "stage_a_batch_t1_5v1": 2,
        "stage_a_grad_accumulation": 8, "eval_batch_multiplier": 1,
    },
}
SERVER_IDS = ("H-T2", "H-T3", "H-N1", "H-N2")


def settings_for(*ids):
    lookup = {s["id"]: s for s in t1run.matrix()["settings"]}
    return [lookup[item] for item in ids]


def validate_configs():
    m = t1run.matrix()
    if m["seed"] != 42 or m["baseline_id"] != "H-T1":
        raise ValueError("This study requires H-T1 and seed 42")
    expected_changes = {setting["id"]: setting["change"] for setting in m["settings"]}
    if [setting["id"] for setting in m["settings"]] != list(expected_changes):
        raise ValueError("Expected the preregistered SF-hack settings")
    base = t1run.config_data(ExperimentConfig.from_yaml(STUDY / "configs/base.yaml"))
    for setting in m["settings"]:
        config = t1run.config_for(setting)
        actual = t1run.config_data(config)
        expected = json.loads(json.dumps(base))
        expected["data"].update(expected_changes[setting["id"]])
        expected["train"]["output_dir"] = actual["train"]["output_dir"]
        if setting["id"] == "H-T1":
            expected["data"].update(H_T1_LOCAL_OVERRIDES["data"])
            expected["model"].update(H_T1_LOCAL_OVERRIDES["model"])
            expected["train"].update(H_T1_LOCAL_OVERRIDES["train"])
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
            raise ValueError(f"SF-hack T1 experiment contract violated: {setting['id']}")


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    memory_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    if memory_mib < 12000:
        raise RuntimeError(f"Local 5080 profile requires at least 12000 MiB, found {memory_mib:.0f}")


def freeze(ids=None):
    path = ROOT / t1run.matrix()["freeze_manifest"]
    if ids is None:
        if path.exists():
            ids = tuple(t1run.read(path).get("ids") or SERVER_IDS)
        else:
            ids = SERVER_IDS
    chosen = settings_for(*ids)
    entries = []
    for setting in chosen:
        record = t1run.read(t1run.run_dir(setting) / "validation_checkpoint.json")
        t1run.verify_record(setting, record)
        entries.append(record)
    path = ROOT / t1run.matrix()["freeze_manifest"]
    if path.exists():
        saved = t1run.read(path)
        if saved.get("study_id") != t1run.matrix()["study_id"] or saved["entries"] != entries:
            raise ValueError("Cannot replace the frozen Test matrix")
        return saved
    if any((t1run.run_dir(s) / "test").exists() for s in chosen):
        raise ValueError("Test artifacts exist before the matrix freeze")
    saved = {"study_id": t1run.matrix()["study_id"], "frozen_at_utc": t1run.now(),
             "ids": [s["id"] for s in chosen], "entries": entries}
    atomic_json(path, saved)
    return saved


def export(ids=None):
    chosen = settings_for(*(ids or SERVER_IDS))
    m = t1run.matrix()
    for setting in chosen:
        if not (t1run.run_dir(setting) / "test/test_release_manifest.json").is_file():
            raise ValueError(f"Export requires Test releases for {', '.join(s['id'] for s in chosen)}")
    frozen = freeze([s["id"] for s in chosen])
    rows = []
    destination = ROOT / m["results_root"]
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result export: {destination}")
    allowed = {".json", ".jsonl", ".log", ".yaml", ".csv"}
    sources = []
    for setting in chosen:
        output = t1run.run_dir(setting)
        release = t1run.read(output / "test/test_release_manifest.json")
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
            metrics = t1run.read(output / filename)
            t1run.validate_metrics(metrics)
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
                                          "validation_completed": len(chosen), "test_completed": len(chosen),
                                          "skipped_ids": ["H-T1"], "weights_included": False,
                                          "evaluation_version": m.get("evaluation_version", "t1_raw_logit_roc_v2"),
                                          "test_reused_after_v1": False, "rows": rows})
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (target / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    hashes = {p.relative_to(target).as_posix(): sha256_file(p) for p in target.rglob("*") if p.is_file()}
    atomic_json(target / "file_hashes.json", hashes)
    target.rename(destination)
    print(destination)


t1run.validate_configs = validate_configs
t1run.require_gpu = require_gpu
t1run.freeze = freeze
matrix = t1run.matrix
config_for = t1run.config_for
run_dir = t1run.run_dir
read = t1run.read
identity = t1run.identity
status = t1run.status


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "status", "train", "freeze", "test", "export"))
    parser.add_argument("--id", help="Optional single preregistered setting, train/test only")
    parser.add_argument("--from-id", help="Train/test this setting and every later one")
    args = parser.parse_args()
    validate_configs()
    ids = [s["id"] for s in matrix()["settings"]]
    if args.id and args.from_id:
        parser.error("use either --id or --from-id")
    if args.id and (args.command not in {"train", "test"} or args.id not in ids):
        parser.error("--id requires a registered setting and train/test command")
    if args.from_id and (args.command not in {"train", "test"} or args.from_id not in ids):
        parser.error("--from-id requires a registered setting and train/test command")
    if args.command == "check":
        print(f"{len(ids)} SF-hack T1 configs valid; seed=42; TSA always enabled; no training/Test executed")
        return
    if args.command == "status":
        status()
        return
    t1run.verify_benchmark()
    if args.from_id:
        chosen = matrix()["settings"][ids.index(args.from_id):]
    elif args.id:
        chosen = [s for s in matrix()["settings"] if s["id"] == args.id]
    else:
        chosen = settings_for(*SERVER_IDS)
    with t1run.study_lock():
        if args.command == "train":
            t1run.train(chosen)
            if args.id is None:
                freeze([s["id"] for s in chosen])
        elif args.command == "freeze":
            freeze([s["id"] for s in chosen])
        elif args.command == "test":
            t1run.test(chosen)
        else:
            export([s["id"] for s in chosen])


if __name__ == "__main__":
    main()
