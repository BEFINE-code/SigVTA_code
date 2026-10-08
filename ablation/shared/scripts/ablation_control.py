from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "dvsrc").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dvsrc.config import ExperimentConfig
from dvsrc.model import DVSRNet
from dvsrc.trainer import Trainer
from dvsrc.utils import atomic_json, seed_everything, sha256_file

MATRIX_PATH = ROOT / os.environ.get("ABLATION_MATRIX", "ablation/t2/matrix.json")


def load_matrix() -> dict[str, Any]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def setting_by_id(experiment_id: str) -> dict[str, Any]:
    for setting in load_matrix()["settings"]:
        if setting["id"] == experiment_id:
            return setting
    raise ValueError(f"Unknown ablation id: {experiment_id}")


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_for(setting: dict[str, Any], seed: int, output: Path) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(ROOT / setting["config"])
    config.train.seed = seed
    config.train.output_dir = str(output)
    return config


def validate_contract(config: ExperimentConfig, setting: dict[str, Any]) -> None:
    errors = []
    expected = {
        "model.variant": (config.model.variant, "t2_abc_v1"),
        "train.stage_b_epochs": (config.train.stage_b_epochs, 16),
        "train.selection_policy": (config.train.selection_policy, "unified_abc_v14"),
        "train.unified_official_loss_weight": (
            config.train.unified_official_loss_weight, 0.0,
        ),
        "train.checkpoint_factor_gates_enabled": (
            config.train.checkpoint_factor_gates_enabled, False,
        ),
        "train.evaluate_test_each_epoch": (config.train.evaluate_test_each_epoch, False),
        "train.stage_b_strict_t1_isolation": (
            config.train.stage_b_strict_t1_isolation, True,
        ),
        "train.t2_copy_encoder_from_t1": (
            config.train.t2_copy_encoder_from_t1,
            setting["t2_copy_encoder_from_t1"],
        ),
    }
    for name, (actual, required) in expected.items():
        if actual != required:
            errors.append(f"{name}: expected {required!r}, got {actual!r}")
    if errors:
        raise ValueError("Ablation contract violation: " + "; ".join(errors))


def verify_inputs(matrix: dict[str, Any]) -> tuple[Path, str]:
    parent = ROOT / matrix["parent_checkpoint"]
    if not parent.is_file():
        raise FileNotFoundError(f"Frozen T1 parent not found: {parent}")
    parent_digest = sha256_file(parent)
    if parent_digest != matrix["parent_sha256"]:
        raise ValueError(
            f"Frozen T1 parent SHA256 mismatch: {parent_digest}"
        )
    audit_path = ROOT / "datasets/protocols/t2/t2_audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("ok") or audit.get("split_digest") != matrix["split_digest"]:
        raise ValueError("Frozen T2 benchmark audit or split digest mismatch")
    return parent, parent_digest


def parameter_snapshot(config: ExperimentConfig, stage_epoch: int) -> dict[str, Any]:
    seed_everything(config.train.seed)
    model = DVSRNet(replace(config.model, pretrained=False))
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = model
    trainer._prepare_stage_b_modules(stage_epoch)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    frozen = [name for name, value in model.named_parameters() if not value.requires_grad]
    return {
        "stage_epoch": stage_epoch,
        "training_phase": trainer._stage_b_phase(stage_epoch),
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "trainable_tensor_count": len(trainable),
        "frozen_tensor_count": len(frozen),
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
    }


def prepare(args: argparse.Namespace) -> None:
    matrix = load_matrix()
    setting = setting_by_id(args.id)
    if args.seed not in matrix["seeds"]:
        raise ValueError(f"Seed {args.seed} is not preregistered")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    forbidden = [output / "test_access_started.json", output / "test_release_manifest.json"]
    if any(path.exists() for path in forbidden):
        raise RuntimeError(f"Test has already been accessed for {args.id} seed {args.seed}")
    config = config_for(setting, args.seed, output)
    validate_contract(config, setting)
    parent, parent_digest = verify_inputs(matrix)
    warmup = parameter_snapshot(config, 0)
    joint = parameter_snapshot(config, config.train.t2_head_warmup_epochs)
    atomic_json(output / "trainable_parameters.json", {
        "schema": "paper_database_v3_trainable_parameters_v1",
        "experiment_id": args.id,
        "seed": args.seed,
        "head_warmup": warmup,
        "joint_training": joint,
    })
    if not config.train.t2_balanced_b_enabled:
        atomic_json(output / "t2_b_status.json", {
            "stage": "t2_b",
            "enabled": False,
            "completed_epochs": 0,
            "reason": "disabled by the resolved experiment configuration",
        })
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment_id") != args.id or manifest.get("seed") != args.seed:
            raise ValueError(f"Existing run manifest identity mismatch: {manifest_path}")
        print(json.dumps(manifest, ensure_ascii=False))
        return
    manifest = {
        "schema": "paper_database_v3_ablation_run_v2",
        "experiment_id": args.id,
        "slug": setting["slug"],
        "seed": args.seed,
        "config": setting["config"],
        "resolved_output": str(output),
        "parent_checkpoint": str(parent),
        "parent_sha256": parent_digest,
        "split_digest": matrix["split_digest"],
        "selection_policy": config.train.selection_policy,
        "official_loss_weight": config.train.unified_official_loss_weight,
        "factor_performance_gates_enabled": config.train.checkpoint_factor_gates_enabled,
        "test_each_epoch": config.train.evaluate_test_each_epoch,
        "test_policy": (
            f"one access after the {len(matrix['settings'])}-run seed-42 "
            "validation checkpoint manifest is frozen"
        ),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "hostname": socket.gethostname(),
        "created_at_utc": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False))


def finalize_validation(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    best = output / "best.pt"
    status_path = output / "stage_b_status.json"
    if not best.is_file() or not status_path.is_file():
        raise FileNotFoundError(f"Incomplete Validation run: {output}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("running") is not False or not status.get("best_checkpoint_exists"):
        raise RuntimeError(f"Stage B has not completed cleanly: {output}")
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    validation = checkpoint.get("validation")
    if not isinstance(validation, dict) or "t2" not in validation:
        raise ValueError(f"Validation-selected checkpoint lacks T2 metrics: {best}")
    source = json.loads((output / "stage_b_source.json").read_text(encoding="utf-8"))
    source_digest = source.get("source_frozen_t1_digest") or source.get("frozen_t1_digest")
    loaded_digest = source.get("loaded_frozen_t1_digest", source_digest)
    if not source_digest or source_digest != loaded_digest:
        raise ValueError(f"T1 digest evidence is missing or inconsistent: {output}")
    atomic_json(output / "validation_metrics.json", validation)
    atomic_json(output / "t1_digest.json", {
        "source_frozen_t1_digest": source_digest,
        "loaded_frozen_t1_digest": loaded_digest,
        "unchanged": True,
    })
    checkpoint_record = {
        "schema": "paper_database_v3_validation_checkpoint_v1",
        "checkpoint": str(best),
        "sha256": sha256_file(best),
        "selection_policy": checkpoint["config"]["train"]["selection_policy"],
        "stage_epoch": checkpoint.get("stage_epoch"),
        "validation_only": True,
        "finalized_at_utc": utc_now(),
    }
    atomic_json(output / "validation_checkpoint.json", checkpoint_record)
    print(json.dumps(checkpoint_record, ensure_ascii=False))


def freeze_tests(args: argparse.Namespace) -> None:
    matrix = load_matrix()
    entries = []
    for setting in matrix["settings"]:
        for seed in matrix["seeds"]:
            output = ROOT / "ablation/t2/runs" / setting["slug"] / f"seed_{seed}"
            record_path = output / "validation_checkpoint.json"
            if not record_path.is_file():
                raise FileNotFoundError(f"Missing Validation checkpoint record: {record_path}")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            checkpoint = Path(record["checkpoint"])
            if sha256_file(checkpoint) != record["sha256"]:
                raise ValueError(f"Checkpoint changed after Validation freeze: {checkpoint}")
            entries.append({
                "id": setting["id"], "slug": setting["slug"], "seed": seed,
                "config": setting["config"], "output": str(output.resolve()),
                "checkpoint": str(checkpoint.resolve()), "sha256": record["sha256"],
            })
    target = ROOT / args.output
    if target.exists():
        raise FileExistsError(f"Refusing to replace frozen Test manifest: {target}")
    atomic_json(target, {
        "schema": "paper_database_v3_test_checkpoint_manifest_v1",
        "frozen_at_utc": utc_now(),
        "entry_count": len(entries),
        "entries": entries,
    })
    print(target)


def finalize_test(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    release_path = output / "t2_release_metrics.json"
    if not release_path.is_file():
        raise FileNotFoundError(f"T2 Test release metrics are missing: {release_path}")
    payload = json.loads(release_path.read_text(encoding="utf-8"))
    test_report = payload.get("test", payload)
    e0 = test_report["metrics"]["E0_predicted_ABC"]
    atomic_json(output / "test_metrics.json", {"t2": e0})
    started = json.loads((output / "test_access_started.json").read_text(encoding="utf-8"))
    atomic_json(output / "test_release_manifest.json", {
        **started,
        "schema": "paper_database_v3_test_release_v1",
        "completed_at_utc": utc_now(),
        "complete": True,
        "release_metrics_sha256": sha256_file(release_path),
    })


def check_parallel(args: argparse.Namespace) -> None:
    if args.max_parallel < 2:
        print(json.dumps({"safe": True, "max_parallel": args.max_parallel}))
        return
    settings = [setting_by_id(experiment_id) for experiment_id in args.ids]
    profiles = []
    for setting in settings:
        path = ROOT / "ablation/t2/profiles" / setting["slug"] / "seed_42/stage_b_profile.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing joint-phase profile for {setting['id']}: {path}. Run profile first."
            )
        profile = json.loads(path.read_text(encoding="utf-8"))
        if profile.get("training_phase") != "joint":
            raise ValueError(f"Profile is not from the joint phase: {path}")
        profiles.append((setting["id"], float(profile["peak_reserved_gib"])))
    peaks = sorted(profiles, key=lambda item: item[1], reverse=True)
    required = peaks[0][1] * args.max_parallel + args.reserve_gib
    total = args.gpu_memory_mib / 1024
    report = {
        "safe": required <= total,
        "max_parallel": args.max_parallel,
        "gpu_total_gib": total,
        "reserve_gib": args.reserve_gib,
        "worst_case_required_gib": required,
        "profiles": dict(profiles),
    }
    print(json.dumps(report, ensure_ascii=False))
    if not report["safe"]:
        raise RuntimeError(
            f"Parallel profile budget requires {required:.2f} GiB, GPU has {total:.2f} GiB"
        )


def begin_test(args: argparse.Namespace) -> None:
    matrix = load_matrix()
    if args.seed not in matrix["seeds"]:
        raise ValueError(f"Seed {args.seed} is not preregistered")
    setting = setting_by_id(args.id)
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        entry for entry in manifest["entries"]
        if entry["id"] == args.id and entry["seed"] == args.seed
    ]
    if len(matches) != 1:
        raise ValueError(f"Frozen Test manifest has no unique entry for {args.id}/{args.seed}")
    entry = matches[0]
    expected_output = (
        ROOT / "ablation/t2/runs" / setting["slug"] / f"seed_{args.seed}"
    ).resolve()
    if entry.get("config") != setting["config"] or Path(entry["output"]).resolve() != expected_output:
        raise ValueError(f"Frozen Test manifest identity mismatch for {args.id}/{args.seed}")
    output = Path(entry["output"])
    started = output / "test_access_started.json"
    completed = output / "test_release_manifest.json"
    if completed.exists():
        print("SKIP")
        return
    if started.exists():
        raise FileExistsError(
            f"Test was already started for {args.id} seed {args.seed}; refusing a second access"
        )
    checkpoint = Path(entry["checkpoint"])
    if sha256_file(checkpoint) != entry["sha256"]:
        raise ValueError(f"Frozen Test checkpoint digest mismatch: {checkpoint}")
    atomic_json(started, {
        "schema": "paper_database_v3_test_access_v1",
        "experiment_id": args.id,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": entry["sha256"],
        "started_at_utc": utc_now(),
        "complete": False,
    })
    print("RUN")
    print(entry["config"])
    print(entry["output"])
    print(entry["checkpoint"])


def resolve(args: argparse.Namespace) -> None:
    setting = setting_by_id(args.id)
    print((ROOT / setting["config"]).resolve())
    print(setting["slug"])
    print(setting["schedule"])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    resolve_parser = sub.add_parser("resolve")
    resolve_parser.add_argument("--id", required=True)
    resolve_parser.set_defaults(func=resolve)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--id", required=True)
    prepare_parser.add_argument("--seed", required=True, type=int)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.set_defaults(func=prepare)
    finalize_parser = sub.add_parser("finalize-validation")
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.set_defaults(func=finalize_validation)
    freeze_parser = sub.add_parser("freeze-tests")
    freeze_parser.add_argument(
        "--output", default="ablation/t2/test_checkpoint_manifest_v2.json",
    )
    freeze_parser.set_defaults(func=freeze_tests)
    parallel_parser = sub.add_parser("check-parallel")
    parallel_parser.add_argument("--ids", nargs="+", required=True)
    parallel_parser.add_argument("--max-parallel", type=int, required=True)
    parallel_parser.add_argument("--gpu-memory-mib", type=int, required=True)
    parallel_parser.add_argument("--reserve-gib", type=float, default=4.0)
    parallel_parser.set_defaults(func=check_parallel)
    begin_test_parser = sub.add_parser("begin-test")
    begin_test_parser.add_argument("--id", required=True)
    begin_test_parser.add_argument("--seed", required=True, type=int)
    begin_test_parser.add_argument(
        "--manifest", default="ablation/t2/test_checkpoint_manifest_v2.json",
    )
    begin_test_parser.set_defaults(func=begin_test)
    test_parser = sub.add_parser("finalize-test")
    test_parser.add_argument("--output", required=True)
    test_parser.set_defaults(func=finalize_test)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
