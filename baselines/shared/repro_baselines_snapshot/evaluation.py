from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dvsrc.metrics import eer, roc_auc, t1_metrics, writer_bootstrap

from .data import BenchmarkRepository, T2_CANDIDATE_COUNT


T2_TYPE_NAMES = ("source_present", "source_absent", "rf_no_source")
T2_SF_ABSENT_CLASS = T2_CANDIDATE_COUNT
T2_RF_CLASS = T2_CANDIDATE_COUNT + 1


class PairScorer(Protocol):
    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def score_t1(rows: list[dict[str, Any]], scorer: PairScorer) -> np.ndarray:
    pairs: list[tuple[str, str]] = []
    counts = []
    for row in rows:
        counts.append(len(row["reference_ids"]))
        pairs.extend((reference, row["query_id"]) for reference in row["reference_ids"])
    pair_scores = scorer.score_pairs(pairs)
    output, offset = [], 0
    for count in counts:
        output.append(float(pair_scores[offset:offset + count].mean()))
        offset += count
    return np.asarray(output, dtype=np.float64)


def _scored_t1(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    return [{**row, "score": float(score)} for row, score in zip(rows, scores)]


def evaluate_t1(repo: BenchmarkRepository, scorer: PairScorer, output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"validation": {}, "test": {}, "calibration": {}}
    for protocol in ("t1_1v1", "t1_5v1"):
        validation = repo.episodes(protocol, "val")
        val_scores = score_t1(validation, scorer)
        threshold = eer(np.asarray([row["label"] for row in validation]), val_scores)[1]
        val_rows = _scored_t1(validation, val_scores)
        test = repo.episodes(protocol, "test", final=True)
        test_rows = _scored_t1(test, score_t1(test, scorer))
        val_metrics = t1_metrics(val_rows, threshold)
        test_metrics = t1_metrics(test_rows, threshold)
        test_metrics["writer_bootstrap_auc"] = writer_bootstrap(
            test_rows,
            lambda sample: roc_auc(
                np.asarray([row["label"] for row in sample]),
                np.asarray([row["score"] for row in sample]),
            ),
        )
        result["validation"][protocol] = val_metrics
        result["test"][protocol] = test_metrics
        result["calibration"][protocol] = {"threshold": float(threshold), "split": "validation"}
        write_jsonl(output / "predictions" / f"val_{protocol}.jsonl", val_rows)
        write_jsonl(output / "predictions" / f"test_{protocol}.jsonl", test_rows)
    return result


def _candidate_count(rows: list[dict[str, Any]]) -> int:
    counts = {len(row["candidate_ids"]) for row in rows}
    if counts != {T2_CANDIDATE_COUNT}:
        raise RuntimeError(f"Expected {T2_CANDIDATE_COUNT} T2 candidates, observed {sorted(counts)}")
    return T2_CANDIDATE_COUNT


def score_t2(rows: list[dict[str, Any]], scorer: PairScorer) -> np.ndarray:
    count = _candidate_count(rows)
    pairs = [(candidate, row["query_id"]) for row in rows for candidate in row["candidate_ids"]]
    scores = np.asarray(scorer.score_pairs(pairs), dtype=np.float64)
    if scores.size != len(rows) * count:
        raise RuntimeError(f"Pair scorer returned {scores.size} values for {len(rows) * count} pairs")
    return scores.reshape(len(rows), count)


def _truth_classes(rows: list[dict[str, Any]]) -> np.ndarray:
    classes = []
    for row in rows:
        episode_type = row["episode_type"]
        if episode_type == "source_present":
            target = int(row["target_index"])
            if not 0 <= target < T2_CANDIDATE_COUNT:
                raise RuntimeError(f"Invalid source-present target: {target}")
            classes.append(target)
        elif episode_type == "source_absent":
            classes.append(T2_SF_ABSENT_CLASS)
        elif episode_type == "rf_no_source":
            classes.append(T2_RF_CLASS)
        else:
            raise RuntimeError(f"Unknown T2 episode type: {episode_type}")
    return np.asarray(classes, dtype=np.int64)


def _boundary_threshold(values: np.ndarray, boundary: int) -> float:
    if boundary == 0:
        return float(np.nextafter(values[0], -np.inf))
    if boundary == len(values):
        return float(np.nextafter(values[-1], np.inf))
    return float((values[boundary - 1] + values[boundary]) / 2.0)


def fit_type_calibration(
    rows: list[dict[str, Any]], candidate_scores: np.ndarray, type_scores: np.ndarray,
) -> dict[str, Any]:
    """Fit two validation-only thresholds for the final three T2 episode states."""
    _candidate_count(rows)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    type_scores = np.asarray(type_scores, dtype=np.float64)
    if candidate_scores.shape != (len(rows), T2_CANDIDATE_COUNT) or type_scores.shape != (len(rows),):
        raise ValueError("T2 calibration arrays have incompatible shapes")
    if not np.isfinite(candidate_scores).all() or not np.isfinite(type_scores).all():
        raise ValueError("T2 calibration scores must be finite")

    true_types = np.asarray([T2_TYPE_NAMES.index(row["episode_type"]) for row in rows])
    rank_hits = candidate_scores.argmax(axis=1) == np.asarray([
        row["target_index"] if row["episode_type"] == "source_present" else -1 for row in rows
    ])
    unique, inverse = np.unique(type_scores, return_inverse=True)
    group_count = len(unique)
    best: tuple[tuple[float, float, float, int, int], tuple[int, int, int], int, int] | None = None

    for order in itertools.permutations(range(3)):
        grouped = []
        for predicted_type in order:
            correct = true_types == predicted_type
            if predicted_type == 0:
                correct &= rank_hits
            grouped.append(np.bincount(inverse, weights=correct, minlength=group_count))
        prefixes = [np.r_[0.0, np.cumsum(values)] for values in grouped]
        first_advantage = prefixes[0] - prefixes[1]
        best_first_value = -np.inf
        best_first_index = 0
        for second_boundary in range(group_count + 1):
            value = float(first_advantage[second_boundary])
            if value > best_first_value:
                best_first_value = value
                best_first_index = second_boundary
            correct_count = (
                prefixes[0][best_first_index]
                + prefixes[1][second_boundary] - prefixes[1][best_first_index]
                + prefixes[2][-1] - prefixes[2][second_boundary]
            )
            predicted = np.full(len(rows), order[2], dtype=np.int64)
            predicted[type_scores < _boundary_threshold(unique, second_boundary)] = order[1]
            predicted[type_scores < _boundary_threshold(unique, best_first_index)] = order[0]
            type_macro = float(np.mean([
                (predicted[true_types == kind] == kind).mean() for kind in range(3)
            ]))
            rank = (
                float(correct_count / len(rows)), type_macro,
                -abs(second_boundary - group_count / 2), -best_first_index, -second_boundary,
            )
            if best is None or rank > best[0]:
                best = (rank, order, best_first_index, second_boundary)
    assert best is not None
    rank, order, first_boundary, second_boundary = best
    return {
        "method": "validation_two_threshold_seven_class_adaptation",
        "type_score_regions_low_mid_high": [T2_TYPE_NAMES[index] for index in order],
        "lower_threshold": _boundary_threshold(unique, first_boundary),
        "upper_threshold": _boundary_threshold(unique, second_boundary),
        "validation_e0_seven_class_accuracy": rank[0],
        "validation_episode_type_macro_accuracy": rank[1],
        "rank_temperature": max(float(np.std(candidate_scores)), 1e-5),
        "candidate_count": T2_CANDIDATE_COUNT,
        "split": "validation",
    }


def _predicted_type(type_score: float, calibration: dict[str, Any]) -> int:
    if type_score < float(calibration["lower_threshold"]):
        region = 0
    elif type_score < float(calibration["upper_threshold"]):
        region = 1
    else:
        region = 2
    return T2_TYPE_NAMES.index(calibration["type_score_regions_low_mid_high"][region])


def make_seven_class_rows(
    rows: list[dict[str, Any]], candidate_scores: np.ndarray, type_scores: np.ndarray,
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    temperature = float(calibration["rank_temperature"])
    for row, scores, type_score in zip(rows, candidate_scores, type_scores):
        logits = (scores - scores.max()) / temperature
        rank_probabilities = np.exp(logits)
        rank_probabilities /= rank_probabilities.sum()
        predicted_type = _predicted_type(float(type_score), calibration)
        type_probabilities = np.full(3, 5e-4, dtype=np.float64)
        type_probabilities[predicted_type] = 0.999
        seven = np.r_[type_probabilities[0] * rank_probabilities, type_probabilities[1:]]
        output.append({
            **row,
            "candidate_scores": np.asarray(scores, dtype=float).tolist(),
            "rank_probabilities": rank_probabilities.tolist(),
            "type_score": float(type_score),
            "type_probabilities": type_probabilities.tolist(),
            "seven_class_probabilities": seven.tolist(),
            "joint_probabilities": np.r_[seven[:T2_CANDIDATE_COUNT], seven[T2_SF_ABSENT_CLASS:].sum()].tolist(),
            "predicted_episode_type": T2_TYPE_NAMES[predicted_type],
        })
    return output


def _safe_binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    recalls = [float((predictions[labels == value] == value).mean()) for value in (0, 1)]
    return {
        "accuracy": float((labels == predictions).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "negative_recall": recalls[0],
        "positive_recall": recalls[1],
    }


def _seven_class_accuracy(rows: list[dict[str, Any]]) -> float:
    truth = _truth_classes(rows)
    predictions = np.asarray([
        int(np.asarray(row["seven_class_probabilities"]).argmax()) for row in rows
    ])
    return float((truth == predictions).mean())


def _rank_accuracy(rows: list[dict[str, Any]], rank: int) -> float:
    present = [row for row in rows if row["episode_type"] == "source_present"]
    hits = [
        row["target_index"] in np.argsort(np.asarray(row["rank_probabilities"]))[-rank:]
        for row in present
    ]
    return float(np.mean(hits))


def seven_class_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = _truth_classes(rows)
    probabilities = np.asarray([row["seven_class_probabilities"] for row in rows])
    predictions = probabilities.argmax(axis=1)
    types = np.asarray([row["episode_type"] for row in rows])
    e0 = float((truth == predictions).mean())

    e1_predictions = predictions.copy()
    non_rf = truth != T2_RF_CLASS
    e1_predictions[non_rf] = np.asarray([
        int(np.asarray(row["seven_class_probabilities"])[:T2_RF_CLASS].argmax())
        for row in np.asarray(rows, dtype=object)[non_rf]
    ])
    e1_predictions[~non_rf] = T2_RF_CLASS
    e1 = float((e1_predictions == truth).mean())
    present = truth < T2_CANDIDATE_COUNT
    rank_predictions = np.asarray([
        int(np.asarray(row["rank_probabilities"]).argmax()) for row in rows
    ])
    e2 = float((~present).sum() + (rank_predictions[present] == truth[present]).sum()) / len(rows)

    rf_labels = (truth == T2_RF_CLASS).astype(np.int64)
    rf_predictions = (predictions == T2_RF_CLASS).astype(np.int64)
    sf_mask = ~rf_labels.astype(bool)
    pool_labels = (truth[sf_mask] < T2_CANDIDATE_COUNT).astype(np.int64)
    pool_predictions = (e1_predictions[sf_mask] < T2_CANDIDATE_COUNT).astype(np.int64)
    rank1 = _rank_accuracy(rows, 1)
    rank3 = _rank_accuracy(rows, min(3, T2_CANDIDATE_COUNT))
    episode_metrics = {
        name: {
            "accuracy": float((predictions[types == name] == truth[types == name]).mean()),
            "count": int((types == name).sum()),
        }
        for name in T2_TYPE_NAMES
    }
    return {
        "e0_seven_class_accuracy": e0,
        "e1_oracle_a_accuracy": e1,
        "e2_oracle_ab_accuracy": e2,
        "joint_accuracy": e0,
        "A_rf_vs_sf": _safe_binary_metrics(rf_labels, rf_predictions),
        "B_source_in_pool_given_sf": _safe_binary_metrics(pool_labels, pool_predictions),
        "C_source_ranking": {"rank_1": rank1, "rank_3": rank3, "count": int(present.sum())},
        "source_present": {"rank_1": rank1, "rank_3": rank3},
        "episode_type_metrics": episode_metrics,
        "count": len(rows),
    }


def _finish_t2(
    validation: list[dict[str, Any]], test: list[dict[str, Any]],
    val_scores: np.ndarray, test_scores: np.ndarray,
    val_type_scores: np.ndarray, test_type_scores: np.ndarray, output: Path,
) -> dict[str, Any]:
    calibration = fit_type_calibration(validation, val_scores, val_type_scores)
    val_rows = make_seven_class_rows(validation, val_scores, val_type_scores, calibration)
    test_rows = make_seven_class_rows(test, test_scores, test_type_scores, calibration)
    val_metrics = seven_class_metrics(val_rows)
    test_metrics = seven_class_metrics(test_rows)
    test_metrics["writer_bootstrap_e0_accuracy"] = writer_bootstrap(test_rows, _seven_class_accuracy)
    test_metrics["writer_bootstrap_rank_1"] = writer_bootstrap(test_rows, lambda sample: _rank_accuracy(sample, 1))
    write_jsonl(output / "predictions" / "val_t2.jsonl", val_rows)
    write_jsonl(output / "predictions" / "test_t2.jsonl", test_rows)
    return {"validation": val_metrics, "test": test_metrics, "calibration": calibration}


def fit_unknown_threshold(rows: list[dict[str, Any]], candidate_scores: np.ndarray) -> dict[str, Any]:
    return fit_type_calibration(rows, candidate_scores, candidate_scores.max(axis=1))


def make_t2_rows(
    rows: list[dict[str, Any]], candidate_scores: np.ndarray, calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    return make_seven_class_rows(rows, candidate_scores, candidate_scores.max(axis=1), calibration)


def fit_existence_threshold(
    rows: list[dict[str, Any]], candidate_scores: np.ndarray, exist_probabilities: np.ndarray,
) -> dict[str, Any]:
    return fit_type_calibration(rows, candidate_scores, exist_probabilities)


def make_two_stage_t2_rows(
    rows: list[dict[str, Any]], candidate_scores: np.ndarray,
    exist_probabilities: np.ndarray, calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    return make_seven_class_rows(rows, candidate_scores, exist_probabilities, calibration)


def evaluate_t2(repo: BenchmarkRepository, scorer: PairScorer, output: Path) -> dict[str, Any]:
    validation = repo.episodes("t2", "val")
    test = repo.episodes("t2", "test", final=True)
    val_scores = score_t2(validation, scorer)
    test_scores = score_t2(test, scorer)
    return _finish_t2(
        validation, test, val_scores, test_scores,
        val_scores.max(axis=1), test_scores.max(axis=1), output,
    )


def evaluate_two_stage_t2(
    repo: BenchmarkRepository, scorer: PairScorer, existence_model: Any,
    set_feature_builder: Any, output: Path,
) -> dict[str, Any]:
    validation = repo.episodes("t2", "val")
    test = repo.episodes("t2", "test", final=True)
    val_scores = score_t2(validation, scorer)
    test_scores = score_t2(test, scorer)
    val_exist = np.asarray(existence_model.predict_proba(set_feature_builder(val_scores))[:, 1])
    test_exist = np.asarray(existence_model.predict_proba(set_feature_builder(test_scores))[:, 1])
    return _finish_t2(validation, test, val_scores, test_scores, val_exist, test_exist, output)


def evaluate_always_unknown(repo: BenchmarkRepository, output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"adaptation": "final seven-class constant rejection controls"}
    for split in ("val", "test"):
        source = repo.episodes("t2", split, final=split == "test")
        split_result = {}
        for name, predicted_class in (("always_sf_source_absent", T2_SF_ABSENT_CLASS), ("always_rf_no_source", T2_RF_CLASS)):
            rows = []
            for row in source:
                probabilities = np.zeros(T2_CANDIDATE_COUNT + 2)
                probabilities[predicted_class] = 1.0
                rows.append({
                    **row,
                    "rank_probabilities": (np.ones(T2_CANDIDATE_COUNT) / T2_CANDIDATE_COUNT).tolist(),
                    "seven_class_probabilities": probabilities.tolist(),
                })
            split_result[name] = seven_class_metrics(rows)
            write_jsonl(output / name / "predictions" / f"{split}_t2.jsonl", rows)
        result["validation" if split == "val" else "test"] = split_result
    return result
