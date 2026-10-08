from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .baselines import DTWBaseline, GlobalFeatureBaseline
from .calibration import Calibrator
from .config import ExperimentConfig
from .data import (
    EpisodeBuilder, EpisodeDataset, SignatureStore,
    T1SingleSplitBuilder, T2SingleSplitBuilder,
    audit_benchmark, audit_t1_single_benchmark,
    audit_t2_single_benchmark,
)
from .metrics import t1_metrics, t2_metrics, writer_bootstrap
from .task_package import export_task_package
from .trainer import Trainer, smoke_forward
from .utils import atomic_json, write_jsonl


def _prepare(args: argparse.Namespace) -> None:
    store = SignatureStore(args.dataset, verify_hashes=args.verify_hashes)
    summary = EpisodeBuilder(store, args.output, seed=args.seed).build()
    report = audit_benchmark(args.dataset, args.output, full_hash=args.verify_hashes)
    print(json.dumps({"summary": summary, "audit": report}, ensure_ascii=False, indent=2))


def _audit(args: argparse.Namespace) -> None:
    print(json.dumps(audit_benchmark(args.dataset, args.benchmark, args.full_hash), ensure_ascii=False, indent=2))


def _prepare_t1_single(args: argparse.Namespace) -> None:
    store = SignatureStore(args.dataset, verify_hashes=args.verify_hashes)
    summary = T1SingleSplitBuilder(store, args.output, seed=args.seed).build()
    report = audit_t1_single_benchmark(
        args.dataset, args.output, full_hash=args.verify_hashes,
    )
    print(json.dumps({"summary": summary, "audit": report}, ensure_ascii=False, indent=2))


def _audit_t1_single(args: argparse.Namespace) -> None:
    report = audit_t1_single_benchmark(args.dataset, args.benchmark, args.full_hash)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _prepare_t2_single(args: argparse.Namespace) -> None:
    store = SignatureStore(args.dataset, verify_hashes=args.verify_hashes)
    summary = T2SingleSplitBuilder(
        store,
        args.benchmark,
        seed=args.seed,
        candidate_count=args.candidate_count,
        source_benchmark_root=args.source_benchmark,
        swap_val_test=args.swap_val_test,
    ).build()
    report = audit_t2_single_benchmark(
        args.dataset,
        args.benchmark,
        full_hash=args.verify_hashes,
        candidate_count=args.candidate_count,
    )
    print(json.dumps({"summary": summary, "audit": report}, ensure_ascii=False, indent=2))


def _audit_t2_single(args: argparse.Namespace) -> None:
    report = audit_t2_single_benchmark(
        args.dataset,
        args.benchmark,
        full_hash=args.full_hash,
        candidate_count=args.candidate_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _train(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    if args.fold is not None:
        config.data.fold = args.fold
    if args.seed is not None:
        config.train.seed = args.seed
    if args.output:
        config.train.output_dir = args.output
    elif args.fold is not None or args.seed is not None:
        config.train.output_dir = f"models/development/runs/full_fold_{config.data.fold}_seed_{config.train.seed}"
    if args.held_out_forger:
        config.train.output_dir = str(Path(config.train.output_dir) / f"lofo_{args.held_out_forger}")
    result = Trainer(config).train(
        held_out_forger=args.held_out_forger,
        resume_path=args.resume,
        initialize_path=args.initialize,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _train_t1(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    if args.seed is not None:
        config.train.seed = args.seed
    if args.workers is not None:
        config.data.num_workers = args.workers
    if args.output:
        config.train.output_dir = args.output
    output = Path(config.train.output_dir)
    if args.resume is None and any(output.glob("*.pt")):
        raise FileExistsError(
            f"Refusing to overwrite an existing T1 run with checkpoints: {output}"
        )
    result = Trainer(config).train_t1(resume_path=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _smoke(args: argparse.Namespace) -> None:
    result = smoke_forward(ExperimentConfig.from_yaml(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _configure_stage_b(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(args.config)
    if args.fold is not None:
        config.data.fold = args.fold
    if args.seed is not None:
        config.train.seed = args.seed
    config.data.num_workers = args.workers
    config.train.output_dir = args.output
    if args.grad_accumulation is not None:
        config.train.grad_accumulation = args.grad_accumulation
    if args.pcgrad is not None:
        config.train.pcgrad = args.pcgrad
    return config


def _profile_stage_b(args: argparse.Namespace) -> None:
    config = _configure_stage_b(args)
    result = Trainer(config).profile_stage_b(
        args.checkpoint, steps=args.steps, stage_epoch=args.stage_epoch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _train_stage_b(args: argparse.Namespace) -> None:
    config = _configure_stage_b(args)
    result = Trainer(config).train_stage_b(
        stage_a_checkpoint=args.stage_a_checkpoint,
        resume_path=args.resume,
        epochs=args.epochs,
        held_out_forger=args.held_out_forger,
        finalize=args.finalize,
        reset_patience=args.reset_patience,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    if args.fold is not None:
        config.data.fold = args.fold
    config.data.num_workers = args.workers
    trainer = Trainer(config)
    trainer.load_checkpoint(args.checkpoint)
    calibration_path = Path(args.calibration) if args.calibration else Path(args.checkpoint).parent / "calibration.json"
    calibrator = Calibrator(**json.loads(calibration_path.read_text())) if calibration_path.exists() else None
    output = Path(args.output or Path(args.checkpoint).parent / "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    trainer.output = output
    metrics = trainer.evaluate_split(args.split, calibrator=calibrator, export=True)
    for name in ("l4", "l20_restricted"):
        path = trainer.t2_fold_dir / f"{args.split}_t2_{name}.jsonl"
        if not path.exists():
            continue
        rows = trainer.predict(trainer.manifest_loader(path), calibrator)
        write_jsonl(output / f"{args.split}_t2_{name}.jsonl", rows)
        result = t2_metrics(rows)
        result["writer_bootstrap_joint_accuracy"] = writer_bootstrap(
            rows, lambda subset: t2_metrics(subset)["joint_accuracy"], repetitions=2000,
        )
        metrics[f"t2_{name}"] = result
    atomic_json(output / f"{args.split}_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _evaluate_t2_release(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    config.data.num_workers = args.workers
    if args.output:
        config.train.output_dir = args.output
    trainer = Trainer(config)
    trainer.load_checkpoint(args.checkpoint)
    splits = ("val", "test") if args.split == "all" else (args.split,)
    a_threshold = None
    if config.train.t2_balanced_a_enabled:
        a_threshold, _ = trainer.evaluate_balanced_a("val")
    reports = {
        split: trainer.evaluate_t2_release(
            split, export=True, a_threshold=a_threshold,
        ) for split in splits
    }
    atomic_json(Path(config.train.output_dir) / "t2_release_metrics.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


def _evaluate_t1(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    if args.fold is not None:
        config.data.fold = args.fold
    config.data.num_workers = args.workers
    config.train.output_dir = args.output
    trainer = Trainer(config)
    trainer.load_checkpoint(args.checkpoint)
    calibrator = trainer.fit_t1_calibration()
    protocols = ("t1_1v1", "t1_5v1")
    validation = trainer.evaluate_split("val", calibrator=calibrator, export=True, protocols=protocols)
    test = trainer.evaluate_split("test", calibrator=calibrator, export=True, protocols=protocols)
    metrics = {"validation": validation, "test": test}
    atomic_json(trainer.output / "t1_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _export_task_package(args: argparse.Namespace) -> None:
    report = export_task_package(
        args.checkpoint,
        args.task,
        args.output,
        calibration_path=args.calibration,
        initialized_from_t1=args.initialized_from_t1,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _dtw(args: argparse.Namespace) -> None:
    import numpy as np

    from .calibration import eer_threshold

    store = SignatureStore(args.dataset)
    baseline = DTWBaseline(store)
    fold = Path(args.benchmark) / f"episodes/fold_{args.fold}"
    outputs = Path(args.output)
    outputs.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"fold": args.fold}
    limit = slice(None, args.limit or None)
    for protocol in ("t1_1v1", "t1_5v1"):
        validation = EpisodeDataset(fold / f"val_{protocol}.jsonl").episodes[limit]
        test = EpisodeDataset(fold / f"test_{protocol}.jsonl").episodes[limit]
        val_raw = np.asarray([baseline.predict_t1(row) for row in validation])
        location, scale = float(np.median(val_raw)), max(float(np.std(val_raw)), 1e-5)
        val_score = 1 / (1 + np.exp(-(val_raw - location) / scale))
        threshold = eer_threshold(np.asarray([row["label"] for row in validation]), val_score)
        test_raw = np.asarray([baseline.predict_t1(row) for row in test])
        test_score = 1 / (1 + np.exp(-(test_raw - location) / scale))
        rows = [{**row, "score": float(score)} for row, score in zip(test, test_score)]
        write_jsonl(outputs / f"test_{protocol}.jsonl", rows)
        metrics[protocol] = t1_metrics(rows, threshold)
        metrics[protocol]["calibration"] = {
            "location": location, "scale": scale, "threshold": float(threshold), "split": "validation",
        }

    val_t2 = EpisodeDataset(fold / "val_t2.jsonl").episodes[limit]
    val_scores = [baseline.scores_t2(row) for row in val_t2]
    best = np.asarray([scores.max() for scores in val_scores])
    thresholds = np.quantile(best, np.linspace(0.02, 0.98, 97))
    best_threshold, best_accuracy = float(thresholds[0]), -1.0
    for threshold in thresholds:
        predictions = [int(scores.argmax()) if scores.max() >= threshold else -1 for scores in val_scores]
        accuracy = float(np.mean([prediction == row["target_index"]
                                  for prediction, row in zip(predictions, val_t2)]))
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = float(threshold), accuracy
    scale = max(float(np.std(best)), 1e-5)
    test_t2 = EpisodeDataset(fold / "test_t2.jsonl").episodes[limit]
    rows = []
    for row in test_t2:
        scores = baseline.scores_t2(row)
        rank = np.exp((scores - scores.max()) / scale)
        rank /= rank.sum()
        exist = 1 / (1 + np.exp(-(scores.max() - best_threshold) / scale))
        rows.append({**row, "joint_probabilities": np.r_[exist * rank, 1 - exist].tolist()})
    write_jsonl(outputs / "test_t2.jsonl", rows)
    metrics["t2"] = t2_metrics(rows)
    metrics["t2"]["calibration"] = {
        "unknown_threshold": best_threshold, "scale": scale,
        "validation_joint_accuracy": best_accuracy, "split": "validation",
    }
    metrics["pair_distances"] = len(baseline.pair_cache)
    atomic_json(outputs / "metrics.json", metrics)
    summary = {
        "fold": args.fold,
        "t1_1v1_auc": metrics["t1_1v1"]["overall"]["roc_auc"],
        "t1_5v1_auc": metrics["t1_5v1"]["overall"]["roc_auc"],
        "t2_joint_accuracy": metrics["t2"]["joint_accuracy"],
        "t2_rank1": metrics["t2"]["source_present"]["rank_1"],
        "pair_distances": metrics["pair_distances"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _feature_baseline(args: argparse.Namespace) -> None:
    import numpy as np

    from .calibration import eer_threshold

    store = SignatureStore(args.dataset)
    baseline = GlobalFeatureBaseline(store)
    rotations = json.loads((Path(args.benchmark) / "folds/fold_rotation.json").read_text())
    output = Path(args.output)
    fold_metrics = {}
    for fold in range(5):
        baseline.fit_normalization(set(rotations[str(fold)]["train"]))
        folder = Path(args.benchmark) / f"episodes/fold_{fold}"
        fold_metrics[str(fold)] = {}
        for protocol in ("t1_1v1", "t1_5v1"):
            val = EpisodeDataset(folder / f"val_{protocol}.jsonl").episodes
            test = EpisodeDataset(folder / f"test_{protocol}.jsonl").episodes
            val_raw = np.asarray([baseline.score_t1(row) for row in val])
            location, scale = np.median(val_raw), max(np.std(val_raw), 1e-5)
            val_score = 1 / (1 + np.exp(-(val_raw - location) / scale))
            threshold = eer_threshold(np.asarray([row["label"] for row in val]), val_score)
            test_raw = np.asarray([baseline.score_t1(row) for row in test])
            test_score = 1 / (1 + np.exp(-(test_raw - location) / scale))
            rows = [{**row, "score": float(score)} for row, score in zip(test, test_score)]
            write_jsonl(output / f"fold_{fold}/test_{protocol}.jsonl", rows)
            fold_metrics[str(fold)][protocol] = t1_metrics(rows, threshold)
        val_t2 = EpisodeDataset(folder / "val_t2.jsonl").episodes
        val_scores = [baseline.scores_t2(row) for row in val_t2]
        best = np.asarray([scores.max() for scores in val_scores])
        thresholds = np.quantile(best, np.linspace(0.02, 0.98, 97))
        best_threshold, best_accuracy = float(thresholds[0]), -1.0
        for threshold in thresholds:
            correct = []
            for row, scores in zip(val_t2, val_scores):
                prediction = int(scores.argmax()) if scores.max() >= threshold else -1
                correct.append(prediction == row["target_index"])
            if np.mean(correct) > best_accuracy:
                best_threshold, best_accuracy = float(threshold), float(np.mean(correct))
        scale = max(float(np.std(best)), 1e-5)
        test_t2 = EpisodeDataset(folder / "test_t2.jsonl").episodes
        rows = []
        for row in test_t2:
            scores = baseline.scores_t2(row)
            rank = np.exp((scores - scores.max()) / scale)
            rank /= rank.sum()
            exist = 1 / (1 + np.exp(-(scores.max() - best_threshold) / scale))
            rows.append({**row, "joint_probabilities": np.r_[exist * rank, 1 - exist].tolist()})
        write_jsonl(output / f"fold_{fold}/test_t2.jsonl", rows)
        fold_metrics[str(fold)]["t2"] = t2_metrics(rows)
    aggregate = {
        "folds": fold_metrics,
        "mean": {
            "t1_1v1_auc": float(np.mean([value["t1_1v1"]["overall"]["roc_auc"] for value in fold_metrics.values()])),
            "t1_5v1_auc": float(np.mean([value["t1_5v1"]["overall"]["roc_auc"] for value in fold_metrics.values()])),
            "t2_joint_accuracy": float(np.mean([value["t2"]["joint_accuracy"] for value in fold_metrics.values()])),
            "t2_rank1": float(np.mean([value["t2"]["source_present"]["rank_1"] for value in fold_metrics.values()])),
        },
    }
    atomic_json(output / "metrics.json", aggregate)
    print(json.dumps(aggregate["mean"], ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dvsrc", description="DVSR-Net T1/T2 benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="materialize writer-disjoint benchmark episodes")
    prepare.add_argument("--dataset", default="datasets/signatures")
    prepare.add_argument("--output", default="datasets/protocols/development")
    prepare.add_argument("--seed", type=int, default=20260723)
    prepare.add_argument("--verify-hashes", action="store_true")
    prepare.set_defaults(func=_prepare)
    audit = sub.add_parser("audit", help="validate an existing benchmark")
    audit.add_argument("--dataset", default="datasets/signatures")
    audit.add_argument("--benchmark", default=".Trash/workspace_reorg_20260907_131226/benchmark/datasets/legacy_main")
    audit.add_argument("--full-hash", action="store_true")
    audit.set_defaults(func=_audit)
    prepare_t1 = sub.add_parser(
        "prepare-t1-single", help="materialize the frozen 55/12/12 T1-only benchmark",
    )
    prepare_t1.add_argument("--dataset", default="datasets/signatures")
    prepare_t1.add_argument("--output", default="datasets/cache/benchmark_t1_single_rebuild")
    prepare_t1.add_argument("--seed", type=int, default=20260731)
    prepare_t1.add_argument("--verify-hashes", action="store_true")
    prepare_t1.set_defaults(func=_prepare_t1_single)
    audit_t1 = sub.add_parser("audit-t1-single", help="validate the frozen T1-only benchmark")
    audit_t1.add_argument("--dataset", default="datasets/signatures")
    audit_t1.add_argument("--benchmark", default="datasets/protocols/t1")
    audit_t1.add_argument("--full-hash", action="store_true")
    audit_t1.set_defaults(func=_audit_t1_single)
    prepare_t2 = sub.add_parser(
        "prepare-t2-single", help="materialize T2 episodes on the frozen T1 writer split",
    )
    prepare_t2.add_argument("--dataset", default="datasets/signatures")
    prepare_t2.add_argument("--benchmark", default="datasets/protocols/t2")
    prepare_t2.add_argument("--seed", type=int, default=20260731)
    prepare_t2.add_argument("--candidate-count", type=int, default=5)
    prepare_t2.add_argument(
        "--source-benchmark",
        default="datasets/protocols/t1",
        help="copy the frozen split, normalization, and T1 episode context into a new benchmark root",
    )
    prepare_t2.add_argument(
        "--swap-val-test",
        action="store_true",
        help="derive the new benchmark with source validation/test writer roles swapped",
    )
    prepare_t2.add_argument("--verify-hashes", action="store_true")
    prepare_t2.set_defaults(func=_prepare_t2_single)
    audit_t2 = sub.add_parser("audit-t2-single", help="validate the frozen single-split T2 benchmark")
    audit_t2.add_argument("--dataset", default="datasets/signatures")
    audit_t2.add_argument("--benchmark", default="datasets/protocols/t2")
    audit_t2.add_argument(
        "--candidate-count",
        type=int,
        help="expected candidate count; inferred from t2_benchmark_manifest.json when omitted",
    )
    audit_t2.add_argument("--full-hash", action="store_true")
    audit_t2.set_defaults(func=_audit_t2_single)
    train = sub.add_parser("train", help="run Stage A/B training and Stage C calibration")
    train.add_argument("--config", default=".Trash/workspace_reorg_20260907_131226/benchmark/configs/full.yaml")
    train.add_argument("--fold", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--output")
    train.add_argument("--held-out-forger", choices=("F01", "F02", "F03", "F04"))
    train.add_argument("--resume", help="resume a Stage A last.pt checkpoint")
    train.add_argument("--initialize", help="transfer compatible representation/T1 weights into a new run")
    train.set_defaults(func=_train)
    train_t1 = sub.add_parser("train-t1", help="train, calibrate, and test only the T1 heads")
    train_t1.add_argument("--config", default="models/v3.0/configs/local_5080_t1_single_v5r1.yaml")
    train_t1.add_argument("--seed", type=int)
    train_t1.add_argument("--workers", type=int)
    train_t1.add_argument("--output")
    train_t1.add_argument("--resume", help="resume a Stage-A last_t1.pt checkpoint")
    train_t1.set_defaults(func=_train_t1)
    smoke = sub.add_parser("smoke", help="one forward/backward batch for every protocol")
    smoke.add_argument("--config", default=".Trash/workspace_reorg_20260907_131226/benchmark/configs/smoke.yaml")
    smoke.set_defaults(func=_smoke)
    profile_b = sub.add_parser("profile-stage-b", help="time a bounded number of real Stage B training steps")
    profile_b.add_argument("--config", default=".Trash/workspace_reorg_20260907_131226/benchmark/configs/full.yaml")
    profile_b.add_argument("--checkpoint", required=True)
    profile_b.add_argument("--fold", type=int)
    profile_b.add_argument("--seed", type=int)
    profile_b.add_argument("--workers", type=int, default=0)
    profile_b.add_argument("--grad-accumulation", type=int)
    profile_b.add_argument("--pcgrad", action=argparse.BooleanOptionalAction, default=None)
    profile_b.add_argument("--steps", type=int, default=100)
    profile_b.add_argument(
        "--stage-epoch", type=int, default=0,
        help="Stage-B epoch whose trainability phase should be profiled",
    )
    profile_b.add_argument("--output", required=True)
    profile_b.set_defaults(func=_profile_stage_b)
    train_b = sub.add_parser(
        "train-stage-b",
        help="start T2 from a T1 checkpoint or resume its T2-A/Stage-B checkpoint",
    )
    train_b.add_argument("--config", default=".Trash/workspace_reorg_20260907_131226/benchmark/configs/full.yaml")
    source = train_b.add_mutually_exclusive_group(required=True)
    source.add_argument("--stage-a-checkpoint")
    source.add_argument("--resume")
    train_b.add_argument("--fold", type=int)
    train_b.add_argument("--seed", type=int)
    train_b.add_argument("--workers", type=int, default=0)
    train_b.add_argument("--grad-accumulation", type=int)
    train_b.add_argument("--pcgrad", action=argparse.BooleanOptionalAction, default=None)
    train_b.add_argument("--epochs", type=int)
    train_b.add_argument("--held-out-forger", choices=("F01", "F02", "F03", "F04"))
    train_b.add_argument("--finalize", action="store_true")
    train_b.add_argument(
        "--reset-patience", action="store_true",
        help="reset the Stage B early-stopping counter and selection baseline after a training-policy change",
    )
    train_b.add_argument("--output", required=True)
    train_b.set_defaults(func=_train_stage_b)
    evaluate = sub.add_parser("evaluate", help="evaluate a checkpoint on main and candidate-size manifests")
    evaluate.add_argument("--config", default=".Trash/workspace_reorg_20260907_131226/benchmark/configs/full.yaml")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--calibration")
    evaluate.add_argument("--fold", type=int)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=_evaluate)
    evaluate_release = sub.add_parser(
        "evaluate-t2-release", help="evaluate E0, oracle-A, and oracle-A+B on one T2 checkpoint",
    )
    evaluate_release.add_argument("--config", default=".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_t2_abc_v1.yaml")
    evaluate_release.add_argument("--checkpoint", required=True)
    evaluate_release.add_argument("--split", choices=("val", "test", "all"), default="all")
    evaluate_release.add_argument("--workers", type=int, default=4)
    evaluate_release.add_argument("--output")
    evaluate_release.set_defaults(func=_evaluate_t2_release)
    evaluate_t1 = sub.add_parser("evaluate-t1", help="calibrate and evaluate only the trained T1 heads")
    evaluate_t1.add_argument("--config", default=".Trash/workspace_reorg_20260907_131226/benchmark/configs/full.yaml")
    evaluate_t1.add_argument("--checkpoint", required=True)
    evaluate_t1.add_argument("--fold", type=int)
    evaluate_t1.add_argument("--workers", type=int, default=0)
    evaluate_t1.add_argument("--output", required=True)
    evaluate_t1.set_defaults(func=_evaluate_t1)
    export_package = sub.add_parser(
        "export-task-package",
        help="export one strict T1 or T2 inference package from a training checkpoint",
    )
    export_package.add_argument("--checkpoint", required=True)
    export_package.add_argument("--task", choices=("t1", "t2"), required=True)
    export_package.add_argument("--output", required=True)
    export_package.add_argument("--calibration")
    export_package.add_argument(
        "--initialized-from-t1",
        help="T1 checkpoint copied into the T2 encoder; recorded as provenance only",
    )
    export_package.set_defaults(func=_export_task_package)
    feature = sub.add_parser("feature-baseline", help="run validation-calibrated global feature baseline on five folds")
    feature.add_argument("--dataset", default="datasets/signatures")
    feature.add_argument("--benchmark", default=".Trash/workspace_reorg_20260907_131226/benchmark/datasets/legacy_main")
    feature.add_argument("--output", default="baselines/shared/runs/global_feature_baseline")
    feature.set_defaults(func=_feature_baseline)
    dtw = sub.add_parser("dtw", help="run a validation-calibrated DTW baseline for one fold")
    dtw.add_argument("--dataset", default="datasets/signatures")
    dtw.add_argument("--benchmark", default=".Trash/workspace_reorg_20260907_131226/benchmark/datasets/legacy_main")
    dtw.add_argument("--fold", type=int, default=0)
    dtw.add_argument("--output", default="baselines/shared/runs/dtw_fold_0")
    dtw.add_argument("--limit", type=int, default=0)
    dtw.set_defaults(func=_dtw)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
