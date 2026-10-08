from contextlib import nullcontext

import pytest
import torch

from dvsrc.cli import build_parser
from dvsrc.config import ExperimentConfig
from dvsrc.data import t2_balanced_a_query_view, t2_balanced_b_view
from dvsrc.losses import T2Loss
from dvsrc.model import DVSRNet
from dvsrc.trainer import (
    BalancedEpisodeBatchSampler, Trainer, counterfactual_t2_pairs,
    deterministic_stratified_subset,
)


class DummyEpisodeDataset:
    def __init__(self, episodes):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)


def _t2_episode(query_id, writer, episode_type, index):
    return {
        "episode_id": f"{query_id}-{episode_type}-{index}",
        "fold_id": 0,
        "split": "val",
        "protocol": "t2_source_ranking",
        "query_id": query_id,
        "target_writer_id": writer,
        "episode_type": episode_type,
    }


def test_t2_a_view_is_unique_and_eval_balanced_per_writer():
    episodes = []
    for writer in ("W1", "W2"):
        for index in range(2):
            episodes.extend(
                _t2_episode(f"{writer}-sf-{index}", writer, episode_type, repeat)
                for episode_type in ("source_present", "source_absent")
                for repeat in range(2)
            )
        for index in range(3):
            episodes.extend(
                _t2_episode(f"{writer}-rf-{index}", writer, "rf_no_source", repeat)
                for repeat in range(2)
            )

    rows = t2_balanced_a_query_view(episodes, seed=42, balance=True)

    assert len(rows) == 8
    assert len({row["query_id"] for row in rows}) == len(rows)
    for writer in ("W1", "W2"):
        writer_rows = [row for row in rows if row["target_writer_id"] == writer]
        assert sum(row["rf_label"] == 0 for row in writer_rows) == 2
        assert sum(row["rf_label"] == 1 for row in writer_rows) == 2


def test_t2_a_train_sampler_emits_equal_sf_and_rf_queries():
    episodes = [
        {"protocol": "t2_a_query", "episode_type": "sf_query"}
        for _ in range(3)
    ] + [
        {"protocol": "t2_a_query", "episode_type": "rf_query"}
        for _ in range(7)
    ]
    sampler = BalancedEpisodeBatchSampler(DummyEpisodeDataset(episodes), batch_size=4, seed=42)

    for batch in sampler:
        types = [episodes[index]["episode_type"] for index in batch]
        assert types.count("sf_query") == 2
        assert types.count("rf_query") == 2


def test_stage_b_metric_reduction_accepts_tensor_losses_and_float_counts():
    assert Trainer._mean_training_values([
        torch.tensor(1.0), torch.tensor(3.0),
    ]) == pytest.approx(2.0)
    assert Trainer._mean_training_values([4.0, 4.0, 4.0]) == pytest.approx(4.0)


def test_t2_b_view_balances_present_and_absent_for_each_sf_query():
    episodes = []
    for query in ("sf-1", "sf-2"):
        episodes.extend(
            _t2_episode(query, "W1", "source_present", index) for index in range(3)
        )
        episodes.extend(
            _t2_episode(query, "W1", "source_absent", index) for index in range(2)
        )

    rows = t2_balanced_b_view(episodes, seed=42)

    assert len(rows) == 8
    for query in ("sf-1", "sf-2"):
        query_rows = [row for row in rows if row["query_id"] == query]
        assert sum(row["episode_type"] == "source_present" for row in query_rows) == 2
        assert sum(row["episode_type"] == "source_absent" for row in query_rows) == 2


def test_t2_b_train_sampler_balances_each_effective_update_window():
    episodes = [
        {"protocol": "t2_source_ranking", "episode_type": episode_type}
        for episode_type in ("source_present", "source_absent")
        for _ in range(7)
    ]
    sampler = BalancedEpisodeBatchSampler(
        DummyEpisodeDataset(episodes), batch_size=1, seed=42, t2_unknown_fraction=0.5,
    )

    sampled_types = [episodes[index]["episode_type"] for batch in sampler for index in batch]
    for start in range(0, len(sampled_types) - 7, 8):
        window = sampled_types[start:start + 8]
        assert window.count("source_present") == 4
        assert window.count("source_absent") == 4


def test_t2_b_pretraining_only_enables_in_set_head():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_t2_abc_balanced_ab_v3.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)

    trainer._prepare_t2_b_modules()

    enabled = {
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    }
    assert enabled
    assert all(name.startswith("t2.in_set_head.") for name in enabled)


def test_unified_v11_warmup_and_joint_trainability_contract():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_abc_v11.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)

    trainer._prepare_stage_b_modules(stage_epoch=0)
    warmup = {
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    }
    assert warmup
    assert all(name.startswith("t2.") for name in warmup)

    trainer._prepare_stage_b_modules(stage_epoch=config.train.t2_head_warmup_epochs)
    joint = {
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("t1.") for name in joint)
    assert any(name.startswith("encoder.") for name in joint)
    assert any(name.startswith("t2.") for name in joint)
    assert not any(name.startswith("t2_encoder.") for name in joint)
    assert trainer.model.t2_encoder is None


def test_unified_v12_trains_only_t2_adapter_and_probability_head():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_abc_v12.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)

    trainer._prepare_stage_b_modules(stage_epoch=0)
    enabled = {
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    }

    assert enabled
    assert all(name.startswith(("t2.", "t2_adapter.")) for name in enabled)
    assert any(name.startswith("t2_adapter.") for name in enabled)
    assert any(name.startswith("t2.") for name in enabled)
    assert trainer.model.t2_encoder is None


def test_unified_v14_switches_from_internal_rank_recovery_to_b_only_calibration():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_abc_v14.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)
    trainer.t2_loss = T2Loss(mode=config.train.t2_loss)

    trainer._prepare_stage_b_modules(stage_epoch=0)
    rank_enabled = {
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    }
    assert rank_enabled
    assert all(name.startswith(("t2_adapter.", "t2.relation.")) for name in rank_enabled)
    assert any(name.startswith("t2_adapter.sequence.") for name in rank_enabled)
    assert any(name.startswith("t2_adapter.fusion_tokens.") for name in rank_enabled)
    assert not any(name.startswith("t2.in_set_head.") for name in rank_enabled)

    trainer._prepare_stage_b_modules(stage_epoch=config.train.t2_rank_recovery_epochs)
    calibration_enabled = {
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    }
    assert calibration_enabled
    assert all(name.startswith("t2.in_set_head.") for name in calibration_enabled)
    assert trainer.model.t2_encoder is None


def test_unified_v14_dispatches_to_internal_adapter_distillation():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_abc_v14.yaml")
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    expected = {"stage": "internal"}
    trainer._run_t2_internal_adapter_distillation = lambda loader, scheduler: expected

    assert trainer._run_t2_adapter_distillation(object(), object()) is expected


def test_unified_scratch_v22_warms_head_then_trains_one_adapter_and_t2_head():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_scratch_abc_v22.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)
    trainer.t2_loss = T2Loss(mode=config.train.t2_loss)

    trainer._prepare_stage_b_modules(stage_epoch=0)
    warmup = {
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    }
    assert warmup
    assert all(name.startswith("t2.") for name in warmup)
    assert not any(name.startswith("t2_adapter.") for name in warmup)

    trainer._prepare_stage_b_modules(stage_epoch=config.train.t2_head_warmup_epochs)
    joint = {
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("t2_adapter.") for name in joint)
    assert any(name.startswith("t2.null_source_head.") for name in joint)
    assert not any(name.startswith(("encoder.", "t1.")) for name in joint)


def test_unified_scratch_initialization_transfers_only_own_t1_state(tmp_path):
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_scratch_abc_v22.yaml")
    config.model.pretrained = False
    config.model.fusion = "sequence_only"
    config.model.hidden_dim = 32
    config.model.conformer_layers = 1
    config.model.conformer_heads = 4
    config.model.conformer_ffn = 64
    config.model.sequence_tokens = 8
    config.model.local_tokens = 4
    config.model.t2_adapter_bottleneck = 8

    torch.manual_seed(40)
    source = DVSRNet(config.model)
    with torch.no_grad():
        next(source.encoder.parameters()).fill_(0.25)
        next(source.t1.parameters()).fill_(0.50)
        next(source.t2.parameters()).fill_(0.75)
    checkpoint_path = tmp_path / "best_t1.pt"
    torch.save({
        "epoch": 2,
        "model": source.state_dict(),
        "config": config.to_dict(),
        "validation": {"t1_1v1": {}, "t1_5v1": {}},
    }, checkpoint_path)

    torch.manual_seed(41)
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)
    trainer.output = tmp_path / "stage_b"
    trainer.output.mkdir()
    fresh_t2 = {
        key: value.clone() for key, value in trainer.model.t2.state_dict().items()
    }

    result = trainer.initialize_unified_scratch_stage_b(checkpoint_path)

    torch.testing.assert_close(
        next(trainer.model.encoder.parameters()), next(source.encoder.parameters()),
    )
    torch.testing.assert_close(
        next(trainer.model.t1.parameters()), next(source.t1.parameters()),
    )
    for key, value in trainer.model.t2.state_dict().items():
        torch.testing.assert_close(value, fresh_t2[key])
    assert result["report"]["external_teacher_used"] is False
    assert result["report"]["fresh_t2_head"] is True


def test_reconstruction_floor_allows_only_the_configured_sample_tolerance():
    value = 541 / 1152

    assert not Trainer._metric_meets_sample_tolerant_floor(value, 0.47, 1152, 0)
    assert Trainer._metric_meets_sample_tolerant_floor(value, 0.47, 1152, 1)
    assert not Trainer._metric_meets_sample_tolerant_floor(540 / 1152, 0.47, 1152, 1)


def test_unified_rea_v21_only_enables_energy_adapter_during_b_calibration():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_rea_v21.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)
    trainer.t2_loss = T2Loss(mode=config.train.t2_loss)

    trainer._prepare_stage_b_modules(stage_epoch=0)
    rank_enabled = {
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    }
    assert not any(name.startswith("t2_energy_adapter.") for name in rank_enabled)

    trainer._prepare_stage_b_modules(stage_epoch=config.train.t2_rank_recovery_epochs)
    calibration_enabled = {
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("t2_energy_adapter.") for name in calibration_enabled)
    assert all(
        name.startswith(("t2.in_set_head.", "t2_energy_adapter."))
        for name in calibration_enabled
    )


def test_t2_batches_balance_present_and_unknown():
    episodes = []
    for episode_type, count in (("source_present", 2), ("source_absent", 4), ("rf_no_source", 6)):
        episodes.extend(
            {"protocol": "t2_source_ranking", "episode_type": episode_type}
            for _ in range(count)
        )
    dataset = DummyEpisodeDataset(episodes)
    sampler = BalancedEpisodeBatchSampler(dataset, batch_size=2, seed=42, t2_unknown_fraction=0.5)

    sampled_types = []
    for batch in sampler:
        batch_types = [episodes[index]["episode_type"] for index in batch]
        assert batch_types.count("source_present") == 1
        assert sum(name != "source_present" for name in batch_types) == 1
        sampled_types.extend(batch_types)

    assert sampled_types.count("source_present") == 6
    assert sampled_types.count("source_absent") == 3
    assert sampled_types.count("rf_no_source") == 3


def test_t2_source_absent_can_be_paired_with_same_query_present_episode():
    episodes = [
        {"protocol": "t2_source_ranking", "episode_type": "source_present", "query_id": "sf-1"},
        {"protocol": "t2_source_ranking", "episode_type": "source_present", "query_id": "sf-2"},
        {"protocol": "t2_source_ranking", "episode_type": "source_absent", "query_id": "sf-1"},
        {"protocol": "t2_source_ranking", "episode_type": "rf_no_source", "query_id": "rf-1"},
    ]
    sampler = BalancedEpisodeBatchSampler(
        DummyEpisodeDataset(episodes), batch_size=2, seed=42,
        t2_unknown_fraction=0.5, pair_source_absent=True,
    )

    batches = [[episodes[index] for index in batch] for batch in sampler]
    paired_batch = next(batch for batch in batches if any(row["episode_type"] == "source_absent" for row in batch))

    assert {row["episode_type"] for row in paired_batch} == {"source_present", "source_absent"}
    assert {row["query_id"] for row in paired_batch} == {"sf-1"}


def test_counterfactual_pair_changes_only_the_source_slot():
    episodes = [
        {
            "episode_id": "present", "protocol": "t2_source_ranking",
            "episode_type": "source_present", "query_id": "sf-1",
            "candidate_ids": ["d1", "source", "d2", "d3"], "target_index": 1,
        },
        {
            "episode_id": "absent", "protocol": "t2_source_ranking",
            "episode_type": "source_absent", "query_id": "sf-1",
            "candidate_ids": ["x1", "x2", "x3", "x4"], "target_index": -1,
        },
    ]

    paired = counterfactual_t2_pairs(episodes, seed=42)
    present, absent = paired

    assert present["counterfactual_pair_id"] == absent["counterfactual_pair_id"]
    assert present["counterfactual_role"] == "present"
    assert absent["counterfactual_role"] == "absent"
    differences = [index for index, values in enumerate(zip(
        present["candidate_ids"], absent["candidate_ids"],
    )) if values[0] != values[1]]
    assert differences == [present["target_index"]]
    assert absent["candidate_ids"][present["target_index"]] != "source"


def test_unified_v13_validation_score_explicitly_rewards_rank_3():
    validation = {
        "t2": {
            "joint_accuracy": 0.55,
            "source_present": {"rank_1": 0.48, "rank_3": 0.82, "mrr": 0.64},
        },
        "t2_b_balanced": {"accuracy": 0.57},
    }
    lower_rank_3 = {
        **validation,
        "t2": {
            **validation["t2"],
            "source_present": {**validation["t2"]["source_present"], "rank_3": 0.70},
        },
    }

    score = Trainer._selection_score(validation, "unified_abc_v13")
    lower_score = Trainer._selection_score(lower_rank_3, "unified_abc_v13")

    assert score == pytest.approx(0.5905)
    assert score > lower_score


def test_unified_v14_validation_score_prioritizes_rank_recovery():
    validation = {
        "t2": {
            "joint_accuracy": 0.55,
            "source_present": {"rank_1": 0.50, "rank_3": 0.84, "mrr": 0.65},
        },
        "t2_b_balanced": {"accuracy": 0.58},
    }
    lower_rank_1 = {
        **validation,
        "t2": {
            **validation["t2"],
            "source_present": {**validation["t2"]["source_present"], "rank_1": 0.40},
        },
    }

    assert Trainer._selection_score(
        validation, "unified_abc_v14",
    ) > Trainer._selection_score(lower_rank_1, "unified_abc_v14")


def test_t1_sampling_keeps_existing_stratum_rotation():
    episodes = [
        {"protocol": "t1_1v1", "label": label, "attack_type": attack_type}
        for label, attack_type in ((1, "genuine"), (0, "RF"), (0, "SF"), (0, "zero_effort"))
        for _ in range(2)
    ]
    dataset = DummyEpisodeDataset(episodes)
    sampler = BalancedEpisodeBatchSampler(dataset, batch_size=2, seed=42)

    sampled = [episodes[index] for batch in sampler for index in batch]
    strata = ["genuine" if row["label"] else row["attack_type"] for row in sampled]

    assert {name: strata.count(name) for name in set(strata)} == {
        "genuine": 2, "RF": 2, "SF": 2, "zero_effort": 2,
    }


def test_stage_b_cli_accepts_one_time_patience_reset():
    args = build_parser().parse_args([
        "train-stage-b", "--resume", "last.pt", "--output", "run", "--reset-patience",
    ])

    assert args.reset_patience is True


def test_stage_a_cli_accepts_resume_checkpoint():
    args = build_parser().parse_args([
        "train-t1", "--resume", "last_t1.pt", "--output", "run",
    ])

    assert args.resume == "last_t1.pt"


def test_stage_a_test_target_requires_every_configured_v1_metric():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_sen_v20_5090.yaml")
    test = {
        "t1_1v1": {"overall": {"accuracy": 0.95, "eer": 0.05}},
        "t1_5v1": {"overall": {"accuracy": 0.98, "eer": 0.02}},
    }

    reached, checks = Trainer._stage_a_test_target_checks(test, config.train)

    assert reached is True
    assert all(checks.values())

    test["t1_5v1"]["overall"]["eer"] = 0.03
    reached, checks = Trainer._stage_a_test_target_checks(test, config.train)
    assert reached is False
    assert checks["t1_5v1_eer"] is False


def test_stage_a_resume_detects_validation_decline_from_existing_history():
    assert Trainer._stage_a_history_declined([
        {"selection_score": 0.9589},
        {"selection_score": 0.9583},
    ]) is True
    assert Trainer._stage_a_history_declined([
        {"selection_score": 0.9583},
        {"selection_score": 0.9589},
    ]) is False
    assert Trainer._stage_a_history_declined([{"selection_score": 0.9583}]) is False


def test_predict_allows_t2_factor_output_without_pim_statistics():
    class FactorOnlyModel:
        def eval(self):
            return self

        def __call__(self, batch):
            return {
                "joint_probability": torch.tensor([[0.6, 0.3, 0.1]]),
                "rank_logits": torch.tensor([[1.0, 0.5]]),
                "rank_probability": torch.tensor([[0.6, 0.4]]),
                "exist_logit": torch.tensor([1.0]),
                "exist_probability": torch.tensor([0.75]),
            }

    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = FactorOnlyModel()
    trainer._autocast = nullcontext
    batch = {
        "protocol": "t2_b_query",
        "set_mask": torch.tensor([[True, True]]),
        "metadata": [{"episode_id": "t2-b-without-pim"}],
    }

    rows = trainer.predict([batch])

    assert rows[0]["joint_probabilities"] == pytest.approx([0.6, 0.3, 0.1])
    assert "pim_statistics" not in rows[0]


def test_finite_optimizer_step_rejects_bad_gradient_before_parameter_update():
    trainer = Trainer.__new__(Trainer)
    trainer.config = type("Config", (), {
        "train": type("Train", (), {"grad_clip": 1.0})(),
    })()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    trainer.optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([float("nan")])

    with pytest.raises(FloatingPointError, match="gradients"):
        trainer._finite_optimizer_step([parameter], "test update")

    torch.testing.assert_close(parameter.detach(), torch.tensor([1.0]))


def test_stage_b_selection_uses_t2_macro_accuracy():
    validation = {
        "t1_5v1": {"overall": {"eer": 0.1}},
        "t2": {"joint_accuracy": 0.95, "episode_type_macro_accuracy": 0.6},
    }

    assert Trainer._selection_score(validation) == 0.75


def test_dual_t1_stage_a_selection_balances_both_protocols():
    validation = {
        "t1_1v1": {"overall": {"eer": 0.2}},
        "t1_5v1": {"overall": {"eer": 0.1}},
    }

    assert Trainer._stage_a_selection_score(validation, "dual_t1") == pytest.approx(0.85)


def test_source_retrieval_selection_does_not_reward_easy_subtype_macro_accuracy():
    validation = {
        "t2": {
            "hierarchical_accuracy": 0.55,
            "source_present": {"rank_1": 0.4, "mrr": 0.6},
            "existence": {"balanced_accuracy": 0.7},
            "collapse_margin": 0.05,
        },
    }

    assert Trainer._selection_score(validation, "source_retrieval_v5") == pytest.approx(0.5275)


def test_stage_b_late_unfreeze_keeps_early_encoder_and_t1_frozen():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v5r1_fulltrain.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)

    trainer._prepare_stage_b_modules()

    assert not any(parameter.requires_grad for parameter in trainer.model.encoder.sequence.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.encoder.sequence.blocks[-1].parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.encoder.image.features[0].parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.encoder.image.features[-1].parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.encoder.fusion.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.t1.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.t2.parameters())


def test_v6_stage_b_freezes_all_encoder_and_t1_parameters():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v6.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)

    trainer._prepare_stage_b_modules()

    assert not any(parameter.requires_grad for parameter in trainer.model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.t1.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.t2.parameters())


def test_v7_stage_b_switches_from_isolated_rank_to_isolated_open_parameters():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v7.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)

    trainer._prepare_stage_b_modules(stage_epoch=0)
    assert all(parameter.requires_grad for parameter in trainer.model.t2.rank_branch.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in trainer.model.t2.named_parameters()
        if not name.startswith("rank_branch.")
    )

    trainer._prepare_stage_b_modules(stage_epoch=config.train.t2_rank_phase_epochs)
    assert not any(parameter.requires_grad for parameter in trainer.model.t2.rank_branch.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in trainer.model.t2.named_parameters()
        if not name.startswith("rank_branch.")
    )


def test_v8_rank_phase_includes_incremental_rank_then_freezes_it_for_open_phase():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v8.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)

    trainer._prepare_stage_b_modules(stage_epoch=0)
    assert all(parameter.requires_grad for parameter in trainer.model.t2.rank_branch.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.t2.incremental_rank.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.t2.state_packet.parameters())

    trainer._prepare_stage_b_modules(stage_epoch=config.train.t2_rank_phase_epochs)
    assert not any(parameter.requires_grad for parameter in trainer.model.t2.rank_branch.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.t2.incremental_rank.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.t2.state_packet.parameters())


def test_v9_training_phases_protect_t1_and_end_with_head_only_calibration():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v9.yaml")
    config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)

    trainer._prepare_stage_b_modules(stage_epoch=0)
    assert not any(parameter.requires_grad for parameter in trainer.model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.t1.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.t2_encoder.parameters())
    assert any(parameter.requires_grad for parameter in trainer.model.t2.candidate_score.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.t2.rf_gate.parameters())

    trainer._prepare_stage_b_modules(stage_epoch=4)
    assert all(parameter.requires_grad for parameter in trainer.model.t2_encoder.parameters())
    assert any(parameter.requires_grad for parameter in trainer.model.t2.rf_gate.parameters())

    trainer._prepare_stage_b_modules(stage_epoch=16)
    assert not any(parameter.requires_grad for parameter in trainer.model.t2_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.t2.candidate_score.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.t2.rf_gate.parameters())


def test_v8_initialization_transfers_compatible_v7_t2_modules(tmp_path):
    source_config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v7.yaml")
    source_config.model.pretrained = False
    source_model = DVSRNet(source_config.model)
    source_parameter = next(source_model.t2.rank_branch.parameters())
    source_parameter.data.fill_(0.125)
    checkpoint_path = tmp_path / "v7.pt"
    torch.save({
        "model": source_model.state_dict(), "config": source_config.to_dict(), "epoch": 3,
    }, checkpoint_path)

    target_config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v8.yaml")
    target_config.model.pretrained = False
    trainer = Trainer.__new__(Trainer)
    trainer.config = target_config
    trainer.device = torch.device("cpu")
    trainer.model = DVSRNet(target_config.model)
    trainer.output = tmp_path

    result = trainer.initialize_isolated_stage_b(checkpoint_path)

    transferred_parameter = next(trainer.model.t2.rank_branch.parameters())
    torch.testing.assert_close(transferred_parameter, torch.full_like(transferred_parameter, 0.125))
    assert result["report"]["transferred_t2_tensors"] > 0
    assert result["report"]["fresh_t2"] is False
    assert result["report"]["fresh_stateful_v8_modules"] is True


def test_v3_selection_uses_both_t1_heads_and_balanced_t2_metrics():
    validation = {
        "t1_1v1": {"overall": {"eer": 0.2}},
        "t1_5v1": {"overall": {"eer": 0.1}},
        "t2": {
            "existence": {"balanced_accuracy": 0.8},
            "source_present": {"rank_1": 0.5, "mrr": 0.6},
        },
    }

    assert Trainer._selection_score(validation, "balanced_v3") == pytest.approx(0.7625)


def test_sampler_cycle_changes_oversampling_order():
    episodes = [
        {"protocol": "t1_1v1", "label": 1, "attack_type": "genuine"}
        for _ in range(10)
    ] + [
        {"protocol": "t1_1v1", "label": 0, "attack_type": "SF"}
        for _ in range(10)
    ]
    sampler = BalancedEpisodeBatchSampler(DummyEpisodeDataset(episodes), batch_size=4, seed=42)
    first = list(sampler)
    sampler.set_cycle(1)
    second = list(sampler)

    assert first != second


def test_reduced_training_subset_is_deterministic_and_stratified():
    episodes = []
    for episode_type, count in (("source_present", 20), ("source_absent", 8), ("rf_no_source", 12)):
        episodes.extend({
            "episode_id": f"{episode_type}-{index}",
            "protocol": "t2_source_ranking",
            "episode_type": episode_type,
        } for index in range(count))

    first = deterministic_stratified_subset(episodes, fraction=0.5, seed=4204)
    second = deterministic_stratified_subset(episodes, fraction=0.5, seed=4204)
    changed = deterministic_stratified_subset(episodes, fraction=0.5, seed=4205)

    assert [row["episode_id"] for row in first] == [row["episode_id"] for row in second]
    assert [row["episode_id"] for row in first] != [row["episode_id"] for row in changed]
    assert {name: sum(row["episode_type"] == name for row in first) for name in {
        "source_present", "source_absent", "rf_no_source",
    }} == {"source_present": 10, "source_absent": 4, "rf_no_source": 6}


def test_v4_selection_rewards_hierarchical_open_set_accuracy():
    validation = {
        "t1_1v1": {"overall": {"eer": 0.1}},
        "t1_5v1": {"overall": {"eer": 0.05}},
        "t2": {
            "existence": {"balanced_accuracy": 0.8},
            "source_present": {"rank_1": 0.5, "mrr": 0.6},
            "hierarchical_accuracy": 0.65,
        },
    }

    assert Trainer._selection_score(validation, "open_set_v4") == pytest.approx(0.741375)


def test_v5_selection_prioritizes_macro_hierarchy_and_penalizes_collapse():
    validation = {
        "t2": {
            "existence": {"balanced_accuracy": 0.7},
            "source_present": {"rank_1": 0.4, "mrr": 0.5},
            "hierarchical_accuracy": 0.55,
            "hierarchical_episode_type_macro_accuracy": 0.6,
            "collapse_margin": 0.05,
        },
    }
    collapsed = {"t2": {**validation["t2"], "collapse_margin": -0.1}}

    assert Trainer._selection_score(validation, "open_set_v5") == pytest.approx(0.54)
    assert Trainer._selection_score(collapsed, "open_set_v5") == pytest.approx(0.44)


def test_v2_rejects_a_legacy_checkpoint_contract():
    trainer = Trainer.__new__(Trainer)
    trainer.config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/full_v2.yaml")
    legacy_checkpoint = {"config": {"data": {}, "model": {}}}

    with pytest.raises(ValueError, match="V1 checkpoints cannot initialize V2"):
        trainer._validate_checkpoint_config(legacy_checkpoint)
