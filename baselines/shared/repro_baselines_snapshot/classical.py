from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import FeatureStore


class GlobalDistanceScorer:
    def __init__(self, features: FeatureStore) -> None:
        self.features = features

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        first = self.features.vectors(pair[0] for pair in pairs)
        second = self.features.vectors(pair[1] for pair in pairs)
        return -np.sqrt(np.mean(np.square(first - second), axis=1))


class StableRandomScorer:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        values = []
        for first, second in pairs:
            names = sorted((first, second))
            digest = hashlib.sha256(f"{self.seed}:{names[0]}:{names[1]}".encode("utf-8")).digest()
            values.append(int.from_bytes(digest[:8], "big") / float(2**64 - 1))
        return np.asarray(values, dtype=np.float64)


@dataclass
class PairModelScorer:
    features: FeatureStore
    model: Any

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        matrix = self.features.pair_matrix(pairs)
        if hasattr(self.model, "decision_function"):
            return np.asarray(self.model.decision_function(matrix), dtype=np.float64)
        return np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=np.float64)


def t1_training_pairs(rows: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], np.ndarray]:
    labels: dict[tuple[str, str], int] = {}
    for row in rows:
        pair = tuple(sorted((row["reference_ids"][0], row["query_id"])))
        value = int(row["label"])
        if pair in labels and labels[pair] != value:
            raise RuntimeError(f"Conflicting T1 pair label: {pair}")
        labels[pair] = value
    pairs = sorted(labels)
    return pairs, np.asarray([labels[pair] for pair in pairs], dtype=np.int64)


def t2_training_pairs(
    rows: list[dict[str, Any]], seed: int, negative_ratio: int = 8,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    labels: dict[tuple[str, str], int] = {}
    for row in rows:
        for index, candidate in enumerate(row["candidate_ids"]):
            pair = tuple(sorted((candidate, row["query_id"])))
            value = int(index == row["target_index"] and row["target_index"] >= 0)
            if pair in labels and labels[pair] != value:
                raise RuntimeError(f"Conflicting T2 pair label: {pair}")
            labels[pair] = value
    positives = sorted(pair for pair, label in labels.items() if label == 1)
    negatives = sorted(pair for pair, label in labels.items() if label == 0)
    rng = np.random.default_rng(seed)
    keep = min(len(negatives), negative_ratio * len(positives))
    selected = [negatives[index] for index in sorted(rng.choice(len(negatives), size=keep, replace=False))]
    pairs = positives + selected
    y = np.r_[np.ones(len(positives), dtype=np.int64), np.zeros(len(selected), dtype=np.int64)]
    order = rng.permutation(len(pairs))
    return [pairs[index] for index in order], y[order]


def make_classifier(name: str, seed: int, labels: np.ndarray) -> Any:
    if name == "svm":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        return make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale", class_weight="balanced", cache_size=4096, random_state=seed),
        )
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        positives = max(int(labels.sum()), 1)
        negatives = max(int((labels == 0).sum()), 1)
        return XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="logloss", n_jobs=-1, random_state=seed,
            scale_pos_weight=negatives / positives,
        )
    raise ValueError(f"Unknown classifier: {name}")


def fit_pair_model(
    name: str, features: FeatureStore, pairs: list[tuple[str, str]], labels: np.ndarray, seed: int,
) -> tuple[PairModelScorer, dict[str, Any]]:
    matrix = features.pair_matrix(pairs)
    model = make_classifier(name, seed, labels)
    started = time.perf_counter()
    model.fit(matrix, labels)
    elapsed = time.perf_counter() - started
    metadata = {
        "fit_seconds": elapsed,
        "training_pairs": len(pairs),
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int((labels == 0).sum()),
        "pair_feature_dim": int(matrix.shape[1]),
    }
    return PairModelScorer(features, model), metadata
