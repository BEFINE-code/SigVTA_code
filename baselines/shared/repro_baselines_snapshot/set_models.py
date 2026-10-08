from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from .classical import fit_pair_model, t2_training_pairs
from .evaluation import score_t2
from .features import FeatureStore


def score_set_features(candidate_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(candidate_scores, dtype=np.float64)
    ordered = np.sort(scores, axis=1)[:, ::-1]
    gaps = ordered[:, :-1] - ordered[:, 1:]
    shifted = ordered - ordered.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities.clip(min=1e-12))).sum(axis=1, keepdims=True)
    summary = np.column_stack([
        scores.mean(axis=1),
        scores.std(axis=1),
        scores.min(axis=1),
        scores.max(axis=1),
        *[np.quantile(scores, quantile, axis=1) for quantile in (0.25, 0.5, 0.75)],
        ordered[:, 0] - scores.mean(axis=1),
        (scores >= 0.5).mean(axis=1),
    ])
    return np.concatenate([ordered, gaps, summary, entropy], axis=1).astype(np.float32)


def _existence_classifier(name: str, seed: int) -> Any:
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown existence classifier: {name}")


def fit_two_stage_model(
    features: FeatureStore,
    train_rows: list[dict[str, Any]],
    train_writers: set[str],
    existence_model: str,
    seed: int,
    folds: int = 5,
) -> tuple[Any, Any, dict[str, Any]]:
    started = time.perf_counter()
    writers = sorted(train_writers)
    writer_folds = [set(writers[index::folds]) for index in range(folds)]
    candidate_counts = {len(row["candidate_ids"]) for row in train_rows}
    if len(candidate_counts) != 1:
        raise RuntimeError(f"Inconsistent T2 candidate counts: {sorted(candidate_counts)}")
    oof_scores = np.full((len(train_rows), candidate_counts.pop()), np.nan, dtype=np.float64)
    fold_metadata: list[dict[str, Any]] = []
    for fold_index, held_out in enumerate(writer_folds):
        fit_rows = [row for row in train_rows if row["target_writer_id"] not in held_out]
        held_indices = [
            index for index, row in enumerate(train_rows)
            if row["target_writer_id"] in held_out
        ]
        held_rows = [train_rows[index] for index in held_indices]
        fold_features = copy.copy(features)
        fold_features.fit_normalization(train_writers - held_out)
        pairs, labels = t2_training_pairs(fit_rows, seed + fold_index)
        scorer, fit_meta = fit_pair_model("xgb", fold_features, pairs, labels, seed + fold_index)
        oof_scores[held_indices] = score_t2(held_rows, scorer)
        fold_metadata.append({
            "fold": fold_index,
            "held_out_writers": sorted(held_out),
            "held_out_episodes": len(held_rows),
            **fit_meta,
        })
    if not np.isfinite(oof_scores).all():
        raise RuntimeError("OOF candidate scores are incomplete")
    existence_labels = np.asarray([row["target_index"] >= 0 for row in train_rows], dtype=np.int64)
    existence_features = score_set_features(oof_scores)
    classifier = _existence_classifier(existence_model, seed)
    classifier.fit(existence_features, existence_labels)

    final_pairs, final_labels = t2_training_pairs(train_rows, seed)
    ranker, final_fit = fit_pair_model("xgb", features, final_pairs, final_labels, seed)
    metadata = {
        "ranker": "xgb_pair_model",
        "existence_model": existence_model,
        "stacking": "5-fold writer-group OOF",
        "set_feature_dim": int(existence_features.shape[1]),
        "existence_positive": int(existence_labels.sum()),
        "existence_negative": int((existence_labels == 0).sum()),
        "folds": fold_metadata,
        "final_ranker_fit": final_fit,
        "fit_seconds": time.perf_counter() - started,
    }
    return ranker, classifier, metadata
