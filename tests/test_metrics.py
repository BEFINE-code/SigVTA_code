import warnings

import numpy as np
import torch

from dvsrc.calibration import Calibrator, eer_threshold
from dvsrc.metrics import (
    balanced_a_metrics, best_balanced_a_threshold, binary_summary,
    conditional_t2_factor_metrics, t1_metrics, t2_metrics,
)
from dvsrc.trainer import Trainer


def test_binary_metrics_perfect_scores():
    summary = binary_summary(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert summary["accuracy"] == 1.0
    assert summary["roc_auc"] == 1.0
    assert summary["eer"] == 0.0


def test_eer_threshold_does_not_depend_on_numpy_reductions():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])

    assert eer_threshold(labels, scores) == 0.8


def test_eer_threshold_accepts_a_missing_class():
    assert eer_threshold(np.asarray([1, 1]), np.asarray([0.4, 0.8])) == 0.4


def test_balanced_a_threshold_and_metrics_do_not_reward_class_imbalance():
    rows = [
        {"rf_label": 0, "rf_probability": 0.1},
        {"rf_label": 0, "rf_probability": 0.3},
        {"rf_label": 1, "rf_probability": 0.7},
        {"rf_label": 1, "rf_probability": 0.9},
    ]

    threshold = best_balanced_a_threshold(rows)
    summary = balanced_a_metrics(rows, threshold)

    assert 0.3 < threshold < 0.7
    assert summary["balanced_accuracy"] == 1.0
    assert summary["rf_recall"] == 1.0
    assert summary["sf_recall"] == 1.0
    assert summary["majority_baseline"] == 0.5


def test_t1_metrics_omits_attack_strata_without_negative_examples():
    rows = [
        {"label": 1, "score": 0.9, "attack_type": "genuine", "query_state": "GENUINE"},
        {"label": 0, "score": 0.1, "attack_type": "SF", "query_state": "SF"},
    ]

    summary = t1_metrics(rows)

    assert "genuine_vs_SF" in summary
    assert "genuine_vs_RF" not in summary
    assert "genuine_vs_zero_effort" not in summary


def test_t2_metrics_perfect_joint_prediction():
    rows = [
        {"joint_probabilities": [0.8, 0.1, 0.1], "target_index": 0, "episode_type": "source_present",
         "forger_id": "F01", "reference_nw_index": 1},
        {"joint_probabilities": [0.1, 0.1, 0.8], "target_index": -1, "episode_type": "source_absent",
         "forger_id": "F01", "reference_nw_index": 1},
    ]
    summary = t2_metrics(rows)
    assert summary["joint_accuracy"] == 1.0
    assert summary["source_present"]["rank_1"] == 1.0
    assert summary["hierarchical_accuracy"] == 1.0
    assert summary["existence"]["balanced_accuracy"] == 1.0


def test_t2_metrics_exposes_always_unknown_collapse():
    rows = [
        {"joint_probabilities": [0.1, 0.1, 0.8], "exist_probability": 0.2,
         "target_index": 0, "episode_type": "source_present"},
        {"joint_probabilities": [0.1, 0.1, 0.8], "exist_probability": 0.2,
         "target_index": -1, "episode_type": "source_absent"},
    ]

    summary = t2_metrics(rows)

    assert summary["hierarchical_accuracy"] == summary["always_unknown_accuracy"] == 0.5
    assert summary["collapse_margin"] == 0.0
    assert summary["existence"]["balanced_accuracy"] == 0.5


def test_t2_metrics_reports_v5_subtype_confusion():
    rows = [
        {"joint_probabilities": [0.8, 0.1, 0.1], "exist_probability": 0.9,
         "target_index": 0, "episode_type": "source_present", "type_probabilities": [0.8, 0.1, 0.1]},
        {"joint_probabilities": [0.1, 0.1, 0.8], "exist_probability": 0.1,
         "target_index": -1, "episode_type": "source_absent", "type_probabilities": [0.1, 0.8, 0.1]},
        {"joint_probabilities": [0.1, 0.1, 0.8], "exist_probability": 0.1,
         "target_index": -1, "episode_type": "rf_no_source", "type_probabilities": [0.1, 0.2, 0.7]},
    ]

    summary = t2_metrics(rows)

    assert summary["subtype"]["accuracy"] == 1.0
    assert summary["subtype"]["macro_recall"] == 1.0


def test_t2_metrics_reports_dynamic_stateful_prefix_progression():
    rows = [
        {
            "joint_probabilities": [0.05, 0.05, 0.8, 0.05, 0.05],
            "exist_probability": 0.8, "target_index": 2, "episode_type": "source_present",
            "type_probabilities": [0.8, 0.1, 0.1],
            "bayesian_stage_probabilities": [[0.5, 0.2, 0.3], [0.2, 0.7, 0.1],
                                                [0.4, 0.5, 0.1], [0.8, 0.1, 0.1]],
            "stateful_prefix_sizes": [1, 2, 4],
            "stateful_prefix_probabilities": [[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]],
        },
        {
            "joint_probabilities": [0.02, 0.02, 0.02, 0.02, 0.92],
            "exist_probability": 0.02, "target_index": -1, "episode_type": "rf_no_source",
            "type_probabilities": [0.02, 0.03, 0.95],
            "bayesian_stage_probabilities": [[0.5, 0.2, 0.3], [0.05, 0.05, 0.9],
                                                [0.03, 0.04, 0.93], [0.02, 0.03, 0.95]],
            "stateful_prefix_sizes": [1, 2, 4],
            "stateful_prefix_probabilities": [[0.05, 0.05, 0.9], [0.03, 0.04, 0.93], [0.02, 0.03, 0.95]],
        },
    ]

    summary = t2_metrics(rows)["stateful_progression"]

    assert summary["dynamic_accuracy"] == {"1": 1.0, "2": 1.0, "4": 1.0}
    assert summary["final_information_gain_nll"] > 0
    assert summary["mean_source_arrival_probability_gain"] > 0
    assert summary["final_matches_full_set"] is True


def test_v5_calibration_uses_subtype_present_probability():
    calibrator = Calibrator(t2_rank_temperature=1.0, t2_type_temperature=1.0)
    rank_logits = torch.zeros(2, 2)
    exist_logits = torch.zeros(2)
    type_logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])

    joint = calibrator.calibrate_t2(rank_logits, exist_logits, type_logits)

    assert joint[0, :-1].sum() > 0.9
    assert joint[1, -1] > 0.9


def test_conditional_t2_factor_metrics_separate_a_and_b():
    rows = [
        {"episode_type": "source_present", "rf_probability": 0.1, "in_set_probability": 0.9},
        {"episode_type": "source_absent", "rf_probability": 0.1, "in_set_probability": 0.1},
        {"episode_type": "rf_no_source", "rf_probability": 0.9, "in_set_probability": 0.8},
    ]

    summary = conditional_t2_factor_metrics(rows)

    assert summary["A_rf_vs_sf"]["accuracy"] == 1.0
    assert summary["A_rf_vs_sf"]["f1"] == 1.0
    assert summary["B_source_in_pool_given_sf"]["accuracy"] == 1.0
    assert summary["B_source_in_pool_given_sf"]["f1"] == 1.0


def test_conditional_t2_factor_metrics_handles_sf_only_balanced_b_view_without_warning():
    rows = [
        {"episode_type": "source_present", "rf_probability": 0.1, "in_set_probability": 0.9},
        {"episode_type": "source_absent", "rf_probability": 0.1, "in_set_probability": 0.1},
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = conditional_t2_factor_metrics(rows)

    assert np.isnan(summary["A_rf_vs_sf"]["positive_recall"])
    assert summary["A_rf_vs_sf"]["negative_recall"] == 1.0
    assert summary["B_source_in_pool_given_sf"]["accuracy"] == 1.0


def test_progressive_release_uses_one_models_rank_probabilities():
    rows = [
        {
            "episode_id": "present", "episode_type": "source_present", "target_index": 1,
            "rank_probabilities": [0.2, 0.8], "rf_probability": 0.4,
            "in_set_probability": 0.25,
        },
        {
            "episode_id": "absent", "episode_type": "source_absent", "target_index": -1,
            "rank_probabilities": [0.7, 0.3], "rf_probability": 0.4,
            "in_set_probability": 0.75,
        },
        {
            "episode_id": "rf", "episode_type": "rf_no_source", "target_index": -1,
            "rank_probabilities": [0.6, 0.4], "rf_probability": 0.2,
            "in_set_probability": 0.9,
        },
    ]

    e1 = Trainer._released_t2_rows(rows, True, False, "E1")
    e2 = Trainer._released_t2_rows(rows, True, True, "E2")

    assert e1[2]["joint_probabilities"] == [0.0, 0.0, 1.0]
    assert e2[0]["joint_probabilities"] == [0.2, 0.8, 0.0]
    assert e2[1]["joint_probabilities"] == [0.0, 0.0, 1.0]
    assert e2[2]["joint_probabilities"] == [0.0, 0.0, 1.0]
