from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable

import numpy as np


def binary_roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if (labels.ndim != 1 or scores.shape != labels.shape or not len(labels)
            or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all()):
        raise ValueError("ROC requires nonempty, finite, aligned binary labels and scores")
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    # Threshold >= score includes every member of a tie, not just the first one.
    distinct = np.r_[scores[1:] != scores[:-1], True]
    true_positive = np.cumsum(labels)[distinct]
    false_positive = np.cumsum(1 - labels)[distinct]
    positives, negatives = max(labels.sum(), 1), max((1 - labels).sum(), 1)
    return np.r_[0.0, false_positive / negatives], np.r_[0.0, true_positive / positives], np.r_[np.inf, scores[distinct]]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = binary_roc(labels, scores)
    return float(np.trapz(np.r_[0, tpr, 1], np.r_[0, fpr, 1]))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if not np.asarray(labels).sum():
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    y = np.asarray(labels)[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = binary_roc(labels, scores)
    frr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - frr)))
    return float((fpr[index] + frr[index]) / 2), float(thresholds[index])


def binary_summary(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5,
                   ranking_scores: np.ndarray | None = None) -> dict[str, float]:
    labels, scores = np.asarray(labels), np.asarray(scores)
    predictions = scores >= threshold
    positives, negatives = labels == 1, labels == 0
    ranking = scores if ranking_scores is None else np.asarray(ranking_scores)
    value_eer, _ = eer(labels, ranking)
    far = float(predictions[negatives].mean()) if negatives.any() else float("nan")
    frr = float((~predictions[positives]).mean()) if positives.any() else float("nan")
    fpr, tpr, _ = binary_roc(labels, ranking)
    eligible = np.where(fpr <= 0.01)[0]
    tar_far_1 = float(tpr[eligible[-1]]) if len(eligible) else 0.0
    return {"accuracy": float((predictions == labels).mean()), "eer": value_eer,
            "roc_auc": roc_auc(labels, ranking), "far": far, "frr": frr, "tar_at_far_1pct": tar_far_1}


def balanced_a_metrics(rows: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, float]:
    """Evaluate RF/SF on a query-balanced view; RF is the positive class."""
    labels = np.asarray([row["rf_label"] for row in rows], dtype=int)
    scores = np.asarray([row["rf_probability"] for row in rows], dtype=float)
    if len(rows) == 0 or len(np.unique(labels)) < 2:
        raise ValueError("Balanced T2-A metrics require both SF and RF queries")
    predictions = scores >= threshold
    rf_recall = float(predictions[labels == 1].mean())
    sf_recall = float((~predictions[labels == 0]).mean())
    summary = binary_summary(labels, scores, threshold)
    summary.update({
        "balanced_accuracy": 0.5 * (rf_recall + sf_recall),
        "rf_recall": rf_recall,
        "sf_recall": sf_recall,
        "f1": _f1(labels, predictions),
        "threshold": float(threshold),
        "majority_baseline": float(max(labels.mean(), 1 - labels.mean())),
        "n_queries": int(len(rows)),
        "n_sf": int((labels == 0).sum()),
        "n_rf": int((labels == 1).sum()),
    })
    return summary


def best_balanced_a_threshold(rows: list[dict[str, Any]]) -> float:
    labels = np.asarray([row["rf_label"] for row in rows], dtype=int)
    scores = np.asarray([row["rf_probability"] for row in rows], dtype=float)
    if len(rows) == 0 or len(np.unique(labels)) < 2:
        raise ValueError("Balanced T2-A threshold fitting requires both SF and RF queries")
    unique = np.unique(scores)
    candidates = np.r_[
        np.nextafter(unique[0], -np.inf),
        (unique[:-1] + unique[1:]) / 2,
        np.nextafter(unique[-1], np.inf),
    ]
    ranked = []
    for threshold in candidates:
        metrics = balanced_a_metrics(rows, float(threshold))
        ranked.append((
            metrics["balanced_accuracy"],
            min(metrics["rf_recall"], metrics["sf_recall"]),
            -abs(float(threshold) - 0.5),
            float(threshold),
        ))
    return max(ranked)[-1]


def t1_metrics(rows: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    def summary(group):
        has_logits = ["raw_logit" in row for row in group]
        if any(has_logits) and not all(has_logits):
            raise ValueError("Cannot mix raw-logit and probability-only T1 predictions")
        ranking = np.asarray([row["raw_logit"] for row in group]) if all(has_logits) else None
        return binary_summary(np.asarray([row["label"] for row in group]),
                              np.asarray([row["score"] for row in group]), threshold, ranking)

    output: dict[str, Any] = {
        "overall": summary(rows), "n_episodes": len(rows),
        "evaluation_version": "t1_raw_logit_roc_v2",
        "ranking_score_source": "raw_logit" if all("raw_logit" in row for row in rows) else "probability",
    }
    for attack in ("zero_effort", "RF", "SF"):
        subset = [row for row in rows if row["label"] == 1 or row["attack_type"] == attack]
        if len({row["label"] for row in subset}) == 2:
            output[f"genuine_vs_{attack}"] = summary(subset)
    condition_groups = defaultdict(list)
    for row in rows:
        condition_groups[row.get("condition_relation", "unknown")].append(row)
    output["conditions"] = {
        name: summary(group)
        for name, group in condition_groups.items() if len({r["label"] for r in group}) == 2
    }
    genuine_by_state = defaultdict(list)
    for row in rows:
        if row["label"] == 1:
            genuine_by_state[row["query_state"]].append(row["score"] >= threshold)
    output["genuine_tpr_by_query_state"] = {
        state: float(np.mean(correct)) for state, correct in genuine_by_state.items()
    }
    if genuine_by_state:
        output["worst_genuine_condition_tpr"] = min(output["genuine_tpr_by_query_state"].values())
    return output


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = np.logical_and(labels == 1, predictions == 1).sum()
    fp = np.logical_and(labels == 0, predictions == 1).sum()
    fn = np.logical_and(labels == 1, predictions == 0).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def t2_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    joint_correct, hierarchical_correct, present_ranks = [], [], []
    subtype_correct: list[bool] = []
    subtype_by_type: dict[str, list[bool]] = defaultdict(list)
    subtype_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    exist_labels, exist_scores, exist_predictions = [], [], []
    unknown_label, unknown_score = [], []
    by_type: dict[str, list[bool]] = defaultdict(list)
    hierarchical_by_type: dict[str, list[bool]] = defaultdict(list)
    by_forger: dict[str, list[bool]] = defaultdict(list)
    by_source: dict[str, list[bool]] = defaultdict(list)
    by_batch: dict[str, list[bool]] = defaultdict(list)
    by_batch_relation: dict[str, list[bool]] = defaultdict(list)
    bayesian_stage_correct: list[list[bool]] = [[], [], [], []]
    bayesian_stage_nll: list[list[float]] = [[], [], [], []]
    bayesian_monotonic: list[bool] = []
    stateful_correct: dict[int, list[bool]] = defaultdict(list)
    stateful_nll: dict[int, list[float]] = defaultdict(list)
    stateful_query_nll: list[float] = []
    stateful_final_nll: list[float] = []
    stateful_arrival_gain: list[float] = []
    stateful_final_match: list[bool] = []
    for row in rows:
        probabilities = np.asarray(row["joint_probabilities"])
        prediction = int(probabilities.argmax())
        candidate_count = len(probabilities) - 1
        target = row["target_index"] if row["target_index"] >= 0 else candidate_count
        correct = prediction == target
        joint_correct.append(correct)
        by_type[row["episode_type"]].append(correct)
        exist_score = float(row.get("exist_probability", 1 - probabilities[-1]))
        exist_threshold = float(row.get("exist_threshold", 0.5))
        exist_prediction = exist_score >= exist_threshold
        hierarchical_prediction = int(probabilities[:-1].argmax()) if exist_prediction else candidate_count
        hierarchical_hit = hierarchical_prediction == target
        hierarchical_correct.append(hierarchical_hit)
        hierarchical_by_type[row["episode_type"]].append(hierarchical_hit)
        exist_labels.append(int(row["target_index"] >= 0))
        exist_scores.append(exist_score)
        exist_predictions.append(int(exist_prediction))
        if row.get("forger_id"):
            by_forger[row["forger_id"]].append(correct)
        if row.get("reference_nw_index") is not None:
            by_source[str(row["reference_nw_index"])].append(correct)
        if row.get("collection_batch"):
            by_batch[row["collection_batch"]].append(correct)
            candidate_batches = row.get("candidate_collection_batches", [])
            relation = "same_batch" if candidate_batches and all(
                batch == row["collection_batch"] for batch in candidate_batches
            ) else "cross_or_mixed_batch"
            by_batch_relation[relation].append(correct)
        unknown_label.append(int(row["target_index"] < 0))
        unknown_score.append(1 - exist_score)
        if "type_probabilities" in row:
            type_names = ("source_present", "source_absent", "rf_no_source")
            predicted_type = type_names[int(np.asarray(row["type_probabilities"]).argmax())]
            subtype_hit = predicted_type == row["episode_type"]
            subtype_correct.append(subtype_hit)
            subtype_by_type[row["episode_type"]].append(subtype_hit)
            subtype_confusion[row["episode_type"]][predicted_type] += 1
        if "bayesian_stage_probabilities" in row:
            stage_probability = np.asarray(row["bayesian_stage_probabilities"], dtype=float)
            target_type = {"source_present": 0, "source_absent": 1, "rf_no_source": 2}[row["episode_type"]]
            correct_probability = np.clip(stage_probability[:, target_type], 1e-8, 1.0)
            for stage_index in range(4):
                bayesian_stage_correct[stage_index].append(
                    int(stage_probability[stage_index].argmax()) == target_type
                )
                bayesian_stage_nll[stage_index].append(float(-np.log(correct_probability[stage_index])))
            bayesian_monotonic.append(bool(np.all(np.diff(correct_probability) >= -1e-8)))
        if "stateful_prefix_probabilities" in row:
            prefix_probability = np.asarray(row["stateful_prefix_probabilities"], dtype=float)
            prefix_sizes = [int(size) for size in row["stateful_prefix_sizes"]]
            episode_type = row["episode_type"]
            final_target = {"source_present": 0, "source_absent": 1, "rf_no_source": 2}[episode_type]
            dynamic_targets = []
            for prefix_index, size in enumerate(prefix_sizes):
                observed = row["target_index"] >= 0 and row["target_index"] < size
                dynamic_target = 2 if episode_type == "rf_no_source" else 0 if observed else 1
                dynamic_targets.append(dynamic_target)
                probability = np.clip(prefix_probability[prefix_index, dynamic_target], 1e-8, 1.0)
                stateful_correct[size].append(int(prefix_probability[prefix_index].argmax()) == dynamic_target)
                stateful_nll[size].append(float(-np.log(probability)))
            query_probability = np.asarray(row["bayesian_stage_probabilities"], dtype=float)[1]
            query_target = 2 if episode_type == "rf_no_source" else 1
            stateful_query_nll.append(float(-np.log(np.clip(query_probability[query_target], 1e-8, 1.0))))
            stateful_final_nll.append(float(-np.log(np.clip(prefix_probability[-1, final_target], 1e-8, 1.0))))
            stateful_final_match.append(bool(np.allclose(
                prefix_probability[-1], np.asarray(row["type_probabilities"]), atol=1e-6,
            )))
            if episode_type == "source_present":
                first_observed = next((i for i, target_type in enumerate(dynamic_targets) if target_type == 0), None)
                if first_observed is not None:
                    before = query_probability[0] if first_observed == 0 else prefix_probability[first_observed - 1, 0]
                    stateful_arrival_gain.append(float(prefix_probability[first_observed, 0] - before))
        if row["target_index"] >= 0:
            candidate_scores = probabilities[:-1]
            rank = int((-candidate_scores).argsort().tolist().index(row["target_index"]) + 1)
            present_ranks.append(rank)
    ranks = np.asarray(present_ranks)
    unknown_label_array, unknown_score_array = np.asarray(unknown_label), np.asarray(unknown_score)
    exist_label_array = np.asarray(exist_labels)
    exist_score_array = np.asarray(exist_scores)
    exist_prediction_array = np.asarray(exist_predictions)
    unknown_prediction = unknown_score_array >= 0.5
    type_accuracy = {key: float(np.mean(value)) for key, value in by_type.items()}
    hierarchical_type_accuracy = {key: float(np.mean(value)) for key, value in hierarchical_by_type.items()}
    present_mask, unknown_mask = exist_label_array == 1, exist_label_array == 0
    present_recall = float(exist_prediction_array[present_mask].mean()) if present_mask.any() else 0.0
    unknown_recall = float((1 - exist_prediction_array[unknown_mask]).mean()) if unknown_mask.any() else 0.0
    balanced_accuracy = 0.5 * (present_recall + unknown_recall)
    always_unknown = float(unknown_mask.mean())
    output = {
        "n_episodes": len(rows), "joint_accuracy": float(np.mean(joint_correct)),
        "hierarchical_accuracy": float(np.mean(hierarchical_correct)),
        "always_unknown_accuracy": always_unknown,
        "collapse_margin": float(np.mean(hierarchical_correct)) - always_unknown,
        "existence": {
            "accuracy": float((exist_prediction_array == exist_label_array).mean()),
            "balanced_accuracy": balanced_accuracy,
            "source_present_recall": present_recall, "unknown_recall": unknown_recall,
            "auroc": roc_auc(exist_label_array, exist_score_array),
        },
        "source_present": {
            "n": int(len(ranks)), "rank_1": float((ranks <= 1).mean()), "rank_3": float((ranks <= 3).mean()),
            "mrr": float((1 / ranks).mean()), "ndcg": float((1 / np.log2(ranks + 1)).mean()),
        } if len(ranks) else {},
        "unknown": {
            "auroc": roc_auc(unknown_label_array, unknown_score_array),
            "auprc": average_precision(unknown_label_array, unknown_score_array),
            "f1": _f1(unknown_label_array, unknown_prediction),
        },
        "episode_type_accuracy": type_accuracy,
        "episode_type_macro_accuracy": float(np.mean(list(type_accuracy.values()))),
        "hierarchical_episode_type_accuracy": hierarchical_type_accuracy,
        "hierarchical_episode_type_macro_accuracy": float(np.mean(list(hierarchical_type_accuracy.values()))),
        "forger_accuracy": {key: float(np.mean(value)) for key, value in by_forger.items()},
        "source_index_accuracy": {key: float(np.mean(value)) for key, value in by_source.items()},
        "collection_batch_accuracy": {key: float(np.mean(value)) for key, value in by_batch.items()},
        "batch_relation_accuracy": {key: float(np.mean(value)) for key, value in by_batch_relation.items()},
    }
    if subtype_correct:
        output["subtype"] = {
            "accuracy": float(np.mean(subtype_correct)),
            "macro_recall": float(np.mean([np.mean(value) for value in subtype_by_type.values()])),
            "recall": {key: float(np.mean(value)) for key, value in subtype_by_type.items()},
            "confusion": {actual: dict(predicted) for actual, predicted in subtype_confusion.items()},
        }
    if bayesian_stage_correct[0]:
        stage_names = ("prior", "query", "global_set", "local_match")
        accuracy = {
            name: float(np.mean(bayesian_stage_correct[index]))
            for index, name in enumerate(stage_names)
        }
        nll = {
            name: float(np.mean(bayesian_stage_nll[index]))
            for index, name in enumerate(stage_names)
        }
        output["bayesian_progression"] = {
            "accuracy": accuracy,
            "nll": nll,
            "information_gain_nll": nll["prior"] - nll["local_match"],
            "final_accuracy_gain": accuracy["local_match"] - accuracy["prior"],
            "sample_monotonic_fraction": float(np.mean(bayesian_monotonic)),
            "validation_nll_non_increasing": all(
                nll[right] <= nll[left]
                for left, right in zip(stage_names, stage_names[1:])
            ),
        }
    if stateful_correct:
        accuracy = {str(size): float(np.mean(values)) for size, values in sorted(stateful_correct.items())}
        nll = {str(size): float(np.mean(stateful_nll[size])) for size in sorted(stateful_nll)}
        output["stateful_progression"] = {
            "dynamic_accuracy": accuracy,
            "dynamic_nll": nll,
            "query_nll": float(np.mean(stateful_query_nll)),
            "final_nll": float(np.mean(stateful_final_nll)),
            "final_information_gain_nll": float(np.mean(stateful_query_nll) - np.mean(stateful_final_nll)),
            "mean_source_arrival_probability_gain": (
                float(np.mean(stateful_arrival_gain)) if stateful_arrival_gain else 0.0
            ),
            "final_matches_full_set": bool(all(stateful_final_match)),
        }
    return output


def conditional_t2_factor_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the A (RF) and B (in-set given SF) decisions explicitly."""
    rf_labels = np.asarray([row["episode_type"] == "rf_no_source" for row in rows], dtype=int)
    rf_scores = np.asarray([row["rf_probability"] for row in rows], dtype=float)
    rf_predictions = rf_scores >= 0.5
    sf_rows = [row for row in rows if row["episode_type"] != "rf_no_source"]
    in_set_labels = np.asarray([
        row["episode_type"] == "source_present" for row in sf_rows
    ], dtype=int)
    in_set_scores = np.asarray([row["in_set_probability"] for row in sf_rows], dtype=float)
    in_set_predictions = in_set_scores >= 0.5
    rf = binary_summary(rf_labels, rf_scores)
    rf_positive = rf_labels == 1
    rf_negative = rf_labels == 0
    rf.update({
        "f1": _f1(rf_labels, rf_predictions),
        "positive_recall": (
            float(rf_predictions[rf_positive].mean()) if rf_positive.any() else float("nan")
        ),
        "negative_recall": (
            float((~rf_predictions[rf_negative]).mean()) if rf_negative.any() else float("nan")
        ),
    })
    in_set = binary_summary(in_set_labels, in_set_scores)
    in_set_positive = in_set_labels == 1
    in_set_negative = in_set_labels == 0
    in_set.update({
        "f1": _f1(in_set_labels, in_set_predictions),
        "present_recall": (
            float(in_set_predictions[in_set_positive].mean())
            if in_set_positive.any() else float("nan")
        ),
        "absent_recall": (
            float((~in_set_predictions[in_set_negative]).mean())
            if in_set_negative.any() else float("nan")
        ),
    })
    return {
        "A_rf_vs_sf": rf,
        "B_source_in_pool_given_sf": in_set,
        "n_all": len(rows),
        "n_sf": len(sf_rows),
    }


def writer_bootstrap(rows: list[dict[str, Any]], metric: Callable[[list[dict[str, Any]]], float],
                     repetitions: int = 2000, seed: int = 2026) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["target_writer_id"]].append(row)
    writers = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(writers, size=len(writers), replace=True)
        resample = [row for writer in sampled for row in groups[str(writer)]]
        values.append(metric(resample))
    return {"estimate": float(metric(rows)), "ci_low": float(np.nanpercentile(values, 2.5)),
            "ci_high": float(np.nanpercentile(values, 97.5)), "repetitions": repetitions}
