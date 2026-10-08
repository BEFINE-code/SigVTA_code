from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .data import BenchmarkRepository, SignatureStore
from .evaluation import (
    evaluate_always_unknown, evaluate_t1, evaluate_t2, evaluate_two_stage_t2, write_json,
)
from .features import FeatureStore


RUN_ROOT = "repro_baselines_final_l5_v3"


def environment() -> dict[str, Any]:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "repro_baselines": __version__,
    }
    try:
        import sklearn
        result["scikit_learn"] = sklearn.__version__
    except (ImportError, RecursionError) as error:
        result["scikit_learn_error"] = f"{type(error).__name__}: {error}"
    try:
        import xgboost
        result["xgboost"] = xgboost.__version__
    except (ImportError, RecursionError) as error:
        result["xgboost_error"] = f"{type(error).__name__}: {error}"
    try:
        import dtaidistance
        result["dtaidistance"] = dtaidistance.__version__
    except ImportError as error:
        result["dtaidistance_error"] = f"{type(error).__name__}: {error}"
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except ImportError as error:
        result["torch_error"] = f"{type(error).__name__}: {error}"
    return result


def prepare(args: argparse.Namespace) -> tuple[BenchmarkRepository, FeatureStore, Path, dict[str, Any]]:
    repo = BenchmarkRepository(args.root)
    audit = repo.verify()
    output = Path(args.output).resolve() if args.output else (
        repo.root / "runs" / RUN_ROOT / args.model / f"seed_{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    store = SignatureStore(repo.dataset)
    features = FeatureStore(store, repo.root / "runs" / RUN_ROOT / "cache" / "global_features_v1.npz")
    features.build()
    features.fit_normalization(repo.writers("train"))
    return repo, features, output, audit


def verify(args: argparse.Namespace) -> None:
    repo = BenchmarkRepository(args.root)
    print(repo.verify())


def run(args: argparse.Namespace) -> None:
    from .classical import (
        GlobalDistanceScorer, StableRandomScorer, fit_pair_model,
        t1_training_pairs, t2_training_pairs,
    )

    repo, features, output, audit = prepare(args)
    started = time.perf_counter()
    metrics: dict[str, Any] = {
        "model": args.model,
        "seed": args.seed,
        "audit": audit,
        "environment": environment(),
    }
    if args.model == "sanity":
        scorer = StableRandomScorer(args.seed)
        metrics["t1_random"] = evaluate_t1(repo, scorer, output / "random")
        metrics["t2_random"] = evaluate_t2(repo, scorer, output / "random")
        metrics["t2_constant_rejection_controls"] = evaluate_always_unknown(repo, output / "constant_rejection")
    elif args.model == "global":
        scorer = GlobalDistanceScorer(features)
        metrics["t1"] = evaluate_t1(repo, scorer, output)
        metrics["t2"] = evaluate_t2(repo, scorer, output)
    elif args.model == "dtw":
        from .dtw import DTWScorer

        scorer = DTWScorer(
            features.store,
            repo.root / "runs" / RUN_ROOT / "cache" / "dtw_v1.npz",
        )
        metrics["t1"] = evaluate_t1(repo, scorer, output)
        metrics["t2"] = evaluate_t2(repo, scorer, output)
    elif args.model in {"svm", "rf", "xgb"}:
        import joblib

        train_t1 = repo.episodes("t1_1v1", "train")
        pairs_t1, labels_t1 = t1_training_pairs(train_t1)
        scorer_t1, fit_t1 = fit_pair_model(args.model, features, pairs_t1, labels_t1, args.seed)
        joblib.dump(scorer_t1.model, output / "t1_model.joblib")
        metrics["t1_fit"] = fit_t1
        metrics["t1"] = evaluate_t1(repo, scorer_t1, output)

        train_t2 = repo.episodes("t2", "train")
        pairs_t2, labels_t2 = t2_training_pairs(train_t2, args.seed)
        scorer_t2, fit_t2 = fit_pair_model(args.model, features, pairs_t2, labels_t2, args.seed)
        joblib.dump(scorer_t2.model, output / "t2_model.joblib")
        metrics["t2_fit"] = fit_t2
        metrics["t2"] = evaluate_t2(repo, scorer_t2, output)
    elif args.model in {"two_stage_logistic", "two_stage_xgb"}:
        import joblib

        from .set_models import fit_two_stage_model, score_set_features

        train_t1 = repo.episodes("t1_1v1", "train")
        pairs_t1, labels_t1 = t1_training_pairs(train_t1)
        scorer_t1, fit_t1 = fit_pair_model("xgb", features, pairs_t1, labels_t1, args.seed)
        joblib.dump(scorer_t1.model, output / "t1_model.joblib")
        metrics["t1_fit"] = fit_t1
        metrics["t1"] = evaluate_t1(repo, scorer_t1, output)

        existence_name = args.model.removeprefix("two_stage_")
        ranker, existence_model, fit_t2 = fit_two_stage_model(
            features, repo.episodes("t2", "train"), repo.writers("train"),
            existence_name, args.seed,
        )
        joblib.dump(ranker.model, output / "t2_ranker.joblib")
        joblib.dump(existence_model, output / "t2_existence.joblib")
        metrics["t2_fit"] = fit_t2
        metrics["t2"] = evaluate_two_stage_t2(
            repo, ranker, existence_model, score_set_features, output,
        )
    elif args.model == "deepsets":
        from .deepsets import fit_and_evaluate_deepsets

        metrics["scope"] = "T2-only set model"
        metrics["t2"], metrics["t2_fit"] = fit_and_evaluate_deepsets(
            repo, features, output, args.seed,
        )
    elif args.model in {"cnn", "bilstm", "transformer"}:
        import torch

        from .deep import DeepConfig, SequenceStore, fit_deep_pair_model

        sequence_store = SequenceStore(
            features.store,
            repo.writers("train"),
            repo.root / "runs" / RUN_ROOT / "cache" / "sequences_128_v1.npz",
        )
        config = DeepConfig()
        train_t1_pairs, train_t1_labels = t1_training_pairs(repo.episodes("t1_1v1", "train"))
        val_t1_pairs, val_t1_labels = t1_training_pairs(repo.episodes("t1_1v1", "val"))
        scorer_t1, fit_t1 = fit_deep_pair_model(
            args.model, sequence_store,
            train_t1_pairs, train_t1_labels, val_t1_pairs, val_t1_labels,
            args.seed, output / "t1_model.pt", config,
        )
        metrics["t1_fit"] = fit_t1
        metrics["t1"] = evaluate_t1(repo, scorer_t1, output)
        del scorer_t1
        torch.cuda.empty_cache()

        train_t2_pairs, train_t2_labels = t2_training_pairs(repo.episodes("t2", "train"), args.seed)
        val_t2_pairs, val_t2_labels = t2_training_pairs(
            repo.episodes("t2", "val"), args.seed, negative_ratio=1_000_000,
        )
        scorer_t2, fit_t2 = fit_deep_pair_model(
            args.model, sequence_store,
            train_t2_pairs, train_t2_labels, val_t2_pairs, val_t2_labels,
            args.seed + 1000, output / "t2_model.pt", config,
        )
        metrics["t2_fit"] = fit_t2
        metrics["t2"] = evaluate_t2(repo, scorer_t2, output)
    elif args.model == "resnet18":
        import torch

        from .image import ImageConfig, ImageStore, fit_resnet18_pair_model

        image_store = ImageStore(
            features.store,
            repo.root / "benchmark" / "cache" / "rendered_png",
            repo.root / "runs" / RUN_ROOT / "cache" / "dynamic_rgb_160_v1.npz",
        )
        config = ImageConfig()
        weights_path = repo.root / "models" / "resnet18-f37072fd.pth"
        train_t1_pairs, train_t1_labels = t1_training_pairs(repo.episodes("t1_1v1", "train"))
        val_t1_pairs, val_t1_labels = t1_training_pairs(repo.episodes("t1_1v1", "val"))
        scorer_t1, fit_t1 = fit_resnet18_pair_model(
            image_store, train_t1_pairs, train_t1_labels, val_t1_pairs, val_t1_labels,
            args.seed, weights_path, output / "t1_model.pt", config,
        )
        metrics["t1_fit"] = fit_t1
        metrics["t1"] = evaluate_t1(repo, scorer_t1, output)
        del scorer_t1
        torch.cuda.empty_cache()

        train_t2_pairs, train_t2_labels = t2_training_pairs(repo.episodes("t2", "train"), args.seed)
        val_t2_pairs, val_t2_labels = t2_training_pairs(
            repo.episodes("t2", "val"), args.seed, negative_ratio=1_000_000,
        )
        scorer_t2, fit_t2 = fit_resnet18_pair_model(
            image_store, train_t2_pairs, train_t2_labels, val_t2_pairs, val_t2_labels,
            args.seed + 1000, weights_path, output / "t2_model.pt", config,
        )
        metrics["t2_fit"] = fit_t2
        metrics["t2"] = evaluate_t2(repo, scorer_t2, output)
    elif args.model in {"tarnn", "tarnn_contrastive"}:
        import torch

        from .aligned import (
            AlignedPairStore, TARNNConfig, fit_metric_tarnn_pair_model, fit_tarnn_pair_model,
        )
        from .deep import SequenceStore

        sequence_store = SequenceStore(
            features.store,
            repo.writers("train"),
            repo.root / "runs" / RUN_ROOT / "cache" / "sequences_128_v1.npz",
        )
        config = TARNNConfig()
        aligned_store = AlignedPairStore(sequence_store, config.window_fraction)
        fit_model = (
            fit_metric_tarnn_pair_model
            if args.model == "tarnn_contrastive"
            else fit_tarnn_pair_model
        )
        train_t1_pairs, train_t1_labels = t1_training_pairs(repo.episodes("t1_1v1", "train"))
        val_t1_pairs, val_t1_labels = t1_training_pairs(repo.episodes("t1_1v1", "val"))
        scorer_t1, fit_t1 = fit_model(
            aligned_store, train_t1_pairs, train_t1_labels, val_t1_pairs, val_t1_labels,
            args.seed, output / "t1_model.pt", config,
        )
        metrics["t1_fit"] = fit_t1
        metrics["t1"] = evaluate_t1(repo, scorer_t1, output)
        del scorer_t1
        torch.cuda.empty_cache()

        train_t2_pairs, train_t2_labels = t2_training_pairs(repo.episodes("t2", "train"), args.seed)
        val_t2_pairs, val_t2_labels = t2_training_pairs(
            repo.episodes("t2", "val"), args.seed, negative_ratio=1_000_000,
        )
        scorer_t2, fit_t2 = fit_model(
            aligned_store, train_t2_pairs, train_t2_labels, val_t2_pairs, val_t2_labels,
            args.seed + 1000, output / "t2_model.pt", config,
        )
        metrics["t2_fit"] = fit_t2
        metrics["t2"] = evaluate_t2(repo, scorer_t2, output)
    elif args.model == "lnps_rnn":
        import torch

        from .deep import DeepConfig, LNPSSequenceStore, fit_deep_pair_model

        sequence_store = LNPSSequenceStore(
            features.store,
            repo.writers("train"),
            repo.root / "runs" / RUN_ROOT / "cache" / "lnps_order2_128_v1.npz",
        )
        config = DeepConfig()
        train_t1_pairs, train_t1_labels = t1_training_pairs(repo.episodes("t1_1v1", "train"))
        val_t1_pairs, val_t1_labels = t1_training_pairs(repo.episodes("t1_1v1", "val"))
        scorer_t1, fit_t1 = fit_deep_pair_model(
            "bilstm", sequence_store,
            train_t1_pairs, train_t1_labels, val_t1_pairs, val_t1_labels,
            args.seed, output / "t1_model.pt", config,
        )
        fit_t1["architecture"] = "lnps_rnn_adapted"
        fit_t1["adaptation"] = "length-normalized 2D path, order-2 prefix signature, Siamese BiLSTM"
        metrics["t1_fit"] = fit_t1
        metrics["t1"] = evaluate_t1(repo, scorer_t1, output)
        del scorer_t1
        torch.cuda.empty_cache()

        train_t2_pairs, train_t2_labels = t2_training_pairs(repo.episodes("t2", "train"), args.seed)
        val_t2_pairs, val_t2_labels = t2_training_pairs(
            repo.episodes("t2", "val"), args.seed, negative_ratio=1_000_000,
        )
        scorer_t2, fit_t2 = fit_deep_pair_model(
            "bilstm", sequence_store,
            train_t2_pairs, train_t2_labels, val_t2_pairs, val_t2_labels,
            args.seed + 1000, output / "t2_model.pt", config,
        )
        fit_t2["architecture"] = "lnps_rnn_adapted"
        fit_t2["adaptation"] = "length-normalized 2D path, order-2 prefix signature, Siamese BiLSTM"
        metrics["t2_fit"] = fit_t2
        metrics["t2"] = evaluate_t2(repo, scorer_t2, output)
    elif args.model == "synsig2vec":
        from .classical import t1_training_pairs
        from .synsig2vec import fit_official_synsig2vec

        if not args.official_root or not args.prepared:
            raise ValueError("synsig2vec requires --official-root and --prepared")
        validation_pairs, validation_labels = t1_training_pairs(
            repo.episodes("t1_1v1", "val"),
        )
        scorer, fit = fit_official_synsig2vec(
            features.store, validation_pairs, validation_labels,
            args.official_root, args.prepared, output / "model.pt", args.seed,
        )
        metrics["fit"] = fit
        metrics["t1"] = evaluate_t1(repo, scorer, output)
        metrics["t2"] = evaluate_t2(repo, scorer, output)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    metrics["total_seconds"] = time.perf_counter() - started
    write_json(output / "metrics.json", metrics)
    print(output / "metrics.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", default=".")
    verify_parser.set_defaults(func=verify)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", default=".")
    run_parser.add_argument(
        "--model",
        choices=(
            "sanity", "global", "dtw", "svm", "rf", "xgb",
            "two_stage_logistic", "two_stage_xgb",
            "deepsets",
            "cnn", "bilstm", "transformer", "resnet18", "tarnn", "tarnn_contrastive",
            "lnps_rnn", "synsig2vec",
        ),
        required=True,
    )
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--output")
    run_parser.add_argument("--official-root")
    run_parser.add_argument("--prepared")
    run_parser.set_defaults(func=run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
