import pytest

from dvsrc.config import ExperimentConfig


def test_config_extends_deep_merges_nested_sections(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "data:\n  num_workers: 3\n"
        "model:\n  hidden_dim: 128\n  fusion: ms_caf\n"
        "train:\n  batch_t2: 7\n  t2_unknown_fraction: 0.4\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\n"
        "model:\n  fusion: late\n"
        "train:\n  output_dir: runs/child\n",
        encoding="utf-8",
    )

    config = ExperimentConfig.from_yaml(child)

    assert config.data.num_workers == 3
    assert config.model.hidden_dim == 128
    assert config.model.fusion == "late"
    assert config.train.batch_t2 == 7
    assert config.train.t2_unknown_fraction == 0.4
    assert config.train.output_dir == "runs/child"


def test_ablation_inherits_formal_training_policy():
    full = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/full.yaml")
    ablation = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/sequence_only.yaml")

    assert ablation.train.batch_t2 == full.train.batch_t2
    assert ablation.train.grad_accumulation == full.train.grad_accumulation
    assert ablation.train.t2_unknown_fraction == full.train.t2_unknown_fraction
    assert ablation.model.hidden_dim == full.model.hidden_dim
    assert ablation.model.fusion == "sequence_only"


def test_v2_config_has_an_independent_input_contract():
    legacy = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/full.yaml")
    v2 = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/full_v2.yaml")

    assert legacy.data.input_pipeline == "legacy_v1"
    assert legacy.model.feature_dim == 10
    assert legacy.model.sequence_stem == "legacy_stride8"
    assert v2.data.input_pipeline == "raw_rgb_v2"
    assert (v2.data.image_height, v2.data.image_width) == (448, 448)
    assert (v2.data.image_margin, v2.data.image_line_width, v2.data.image_supersample) == (16, 4, 2)
    assert v2.model.feature_dim == 5
    assert v2.model.sequence_stem == "raw_multiscale"
    assert v2.train.t2_loss == "factorized_v2"
    assert v2.train.reshuffle_on_cycle is True


def test_v3_enables_protocol_heads_hierarchical_t2_and_balanced_selection():
    v3 = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/full_v3.yaml")

    assert v3.model.variant == "v3"
    assert v3.train.t2_loss == "hierarchical_v3"
    assert v3.train.selection_policy == "balanced_v3"


def test_v4_uses_v1_representation_half_training_data_and_frozen_t2_stage():
    v4 = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v4.yaml")

    assert v4.data.input_pipeline == "legacy_v1"
    assert v4.data.train_episode_fraction == 0.5
    assert (v4.model.variant, v4.model.feature_dim, v4.model.sequence_stem) == (
        "v4", 10, "legacy_stride8",
    )
    assert v4.train.t2_loss == "open_set_v4"
    assert v4.train.selection_policy == "open_set_v4"
    assert v4.train.stage_b_train_t1 is False
    assert v4.train.stage_b_freeze_encoder is True


def test_v5_keeps_v4_compute_budget_and_enables_subtype_open_set_training():
    v4 = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v4.yaml")
    v5 = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v5.yaml")

    assert v5.model.variant == "v5"
    assert v5.train.t2_loss == "open_set_v5"
    assert v5.train.selection_policy == "open_set_v5"
    assert v5.data.train_episode_fraction == v4.data.train_episode_fraction == 0.5
    assert (v5.train.stage_a_epochs, v5.train.stage_b_epochs, v5.train.patience) == (10, 18, 6)
    assert v5.train.stage_b_train_t1 is False
    assert v5.train.stage_b_freeze_encoder is True


def test_v5r1_keeps_v5_budget_and_uses_evidence_anchored_variant():
    v5 = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v5.yaml")
    v5r1 = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v5r1.yaml")

    assert v5r1.model.variant == "v5r1"
    assert v5r1.train.t2_loss == "open_set_v5"
    assert v5r1.data.train_episode_fraction == v5.data.train_episode_fraction == 0.5
    assert (v5r1.train.stage_a_epochs, v5r1.train.stage_b_epochs) == (10, 18)
    assert v5r1.train.output_dir == ".Trash/workspace_reorg_20260907_131226/benchmark/runs/v5r1_local_5080_fold_0_seed_42"


def test_v5r1_fulltrain_uses_all_fold_zero_episodes_and_late_encoder_adaptation():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v5r1_fulltrain.yaml")

    assert config.data.fold == 0
    assert config.data.train_episode_fraction == 1.0
    assert (config.train.stage_a_epochs, config.train.stage_b_epochs, config.train.patience) == (30, 40, 10)
    assert config.train.stage_a_selection_policy == "dual_t1"
    assert config.train.stage_b_freeze_encoder is True
    assert config.train.stage_b_unfreeze_late_encoder is True
    assert config.train.t2_pair_source_absent is True
    assert config.train.selection_policy == "source_retrieval_v5"


def test_t1_single_config_preserves_manifest_ratio_and_uses_a_new_output():
    config = ExperimentConfig.from_yaml("models/configs/local_5080_t1_single_v5r1.yaml")

    assert config.data.benchmark_root == "datasets/protocols/t1"
    assert config.data.image_cache_root == (
        "datasets/cache/t1/rendered_png"
    )
    assert config.data.train_episode_fraction == 1.0
    assert config.model.variant == "v5r1"
    assert config.train.t1_balance_strata is False
    assert config.train.output_dir == "models/runs/t1_single_v1_v5r1_fresh_seed_42"


def test_t2_abc_uses_single_split_one_encoder_and_plain_conditional_training():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_t2_abc_v1.yaml")

    assert config.data.benchmark_root == (
        ".Trash/workspace_reorg_20260907_131226/benchmark/archive/final_cleanup_20260806/t1_protocol_history/"
        "benchmark_t1_single_v1_original_roles"
    )
    assert config.model.variant == "t2_abc_v1"
    assert config.model.t1_variant == "v5r1"
    assert config.train.t2_loss == "conditional_abc_v1"
    assert config.train.selection_policy == "conditional_abc_v1"
    assert config.train.stage_a_epochs == 0
    assert config.train.stage_b_epochs == 20
    assert config.train.t2_head_warmup_epochs == 2
    assert config.train.stage_b_strict_t1_isolation is True
    assert config.train.stage_b_train_t1 is False
    assert config.train.t2_pair_source_absent is False
    assert config.train.t2_counterfactual_pairs is False
    assert (config.train.batch_t2, config.train.grad_accumulation) == (2, 4)


def test_t2_balanced_a_v2_keeps_joint_ratio_and_adds_staged_a_training():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_t2_abc_balanced_a_v2.yaml")

    assert config.model.variant == "t2_abc_v1"
    assert config.train.t2_unknown_fraction == 0.5
    assert config.train.t2_balanced_a_enabled is True
    assert config.train.t2_a_pretrain_epochs == 8
    assert config.train.batch_t2_a == 8
    assert config.train.t2_a_loss_weight == 1.0
    assert config.train.t2_a_max_balanced_accuracy_drop == 0.02
    assert config.train.t2_a_min_class_recall == 0.20
    assert config.train.output_dir == ".Trash/workspace_reorg_20260907_131226/benchmark/runs/t2_abc_balanced_a_v2_seed_42"


def test_t2_balanced_ab_v3_adds_balanced_b_without_changing_joint_ratio():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_t2_abc_balanced_ab_v3.yaml")

    assert config.model.variant == "t2_abc_v1"
    assert config.train.t2_unknown_fraction == 0.5
    assert config.train.t2_balanced_a_enabled is True
    assert config.train.t2_balanced_b_enabled is True
    assert config.train.t2_b_pretrain_epochs == 4
    assert config.train.t2_b_patience == 2
    assert (config.train.batch_t2_b, config.train.t2_b_grad_accumulation) == (1, 8)
    assert config.train.t2_b_loss_weight == 1.0
    assert config.train.t2_b_max_balanced_accuracy_drop == 0.02
    assert config.train.t2_b_min_class_recall == 0.20
    assert config.train.output_dir == ".Trash/workspace_reorg_20260907_131226/benchmark/runs/t2_abc_balanced_ab_v3_seed_42"


def test_t2_balanced_ab_v3_l5_uses_isolated_benchmark_and_output():
    config = ExperimentConfig.from_yaml("models/configs/local_5080_t2_abc_balanced_ab_v3_l5.yaml")

    assert config.data.benchmark_root == "datasets/protocols/t2"
    assert config.model.variant == "t2_abc_v1"
    assert config.train.t2_balanced_a_enabled is True
    assert config.train.t2_balanced_b_enabled is True
    assert config.train.output_dir == "models/runs/t2_abc_balanced_ab_v3_l5_seed_42"


def test_unified_v11_keeps_tasks_and_uses_separate_episode_roots():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_abc_v11.yaml")

    assert config.data.benchmark_root == "datasets/protocols/t1"
    assert config.data.t2_benchmark_root == "datasets/protocols/t2"
    assert config.model.variant == "unified_abc_v11"
    assert config.model.t1_variant == "v5r1"
    assert (config.model.conformer_layers, config.model.hidden_dim) == (6, 256)
    assert config.model.image_backbone == "convnext_tiny"
    assert config.model.fusion == "ms_caf"
    assert config.train.t2_loss == "conditional_abc_v1"
    assert config.train.selection_policy == "unified_abc_v11"
    assert config.train.stage_b_train_t1 is True
    assert config.train.stage_b_strict_t1_isolation is False
    assert config.train.pcgrad is True


def test_unified_v12_uses_one_frozen_encoder_and_t2_adapter_training():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_abc_v12.yaml")

    assert config.data.benchmark_root == "datasets/protocols/t1"
    assert config.data.t2_benchmark_root == "datasets/protocols/t2"
    assert config.model.variant == "unified_abc_v12"
    assert config.model.t1_variant == "v5r1"
    assert (config.model.conformer_layers, config.model.hidden_dim) == (6, 256)
    assert config.model.t2_adapter_bottleneck == 32
    assert config.train.selection_policy == "unified_abc_v12"
    assert config.train.t2_loss == "conditional_abc_v1"
    assert config.train.t2_adapter_distillation_steps > 0
    assert config.train.stage_b_train_t1 is False
    assert config.train.stage_b_freeze_encoder is True
    assert config.train.stage_b_unfreeze_late_encoder is False
    assert config.train.stage_b_strict_t1_isolation is True
    assert config.train.pcgrad is False


def test_unified_v13_focuses_relation_distillation_without_changing_task_contracts():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_abc_v13.yaml")

    assert config.model.variant == "unified_abc_v13"
    assert config.model.t1_variant == "v5r1"
    assert config.model.t2_adapter_bottleneck == 64
    assert config.train.t2_loss == "conditional_abc_v1"
    assert config.train.selection_policy == "unified_abc_v13"
    assert config.train.stage_b_train_t1 is False
    assert config.train.stage_b_freeze_encoder is True
    assert config.train.stage_b_strict_t1_isolation is True
    assert config.train.t2_relation_distillation_epochs == 2
    assert config.train.unified_rank_loss_weight > config.train.unified_factor_b_loss_weight
    assert config.train.unified_factor_b_loss_weight > config.train.unified_factor_a_loss_weight
    assert config.train.t2_counterfactual_pairs is True
    assert config.train.t2_pair_source_absent is True
    assert config.train.evaluate_test_each_epoch is True


def test_unified_v14_uses_internal_adapter_rank_recovery_and_rank_gated_release():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_abc_v14.yaml")

    assert config.model.variant == "unified_abc_v14"
    assert config.model.t1_variant == "v5r1"
    assert (config.model.conformer_layers, config.model.hidden_dim) == (6, 256)
    assert config.train.t2_loss == "conditional_abc_v1"
    assert config.train.selection_policy == "unified_abc_v14"
    assert config.train.stage_b_strict_t1_isolation is True
    assert config.train.t2_adapter_internal_distillation_weight > 0
    assert config.train.t2_rank_recovery_epochs > 0
    assert config.train.t2_rank_recovery_epochs < config.train.stage_b_epochs
    assert config.train.t2_hard_negative_loss_weight > 0
    assert config.train.checkpoint_min_rank_1 >= 0.485
    assert config.train.checkpoint_min_rank_3 >= 0.84
    assert config.train.evaluate_test_each_epoch is True


def test_unified_sen_v20_uses_one_trainable_small_encoder_and_shared_evidence():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_sen_v20.yaml")

    assert config.model.variant == "unified_sen_v20"
    assert config.model.t1_variant == "v5r1"
    assert config.model.image_backbone == "convnext_small"
    assert config.model.t2_adapter_bottleneck == 128
    assert config.model.use_pim is False
    assert config.model.use_ocsr is False
    assert config.train.stage_a_epochs == 10
    assert config.train.stage_b_epochs == 16
    assert config.train.stage_b_train_t1 is True
    assert config.train.stage_b_freeze_encoder is False
    assert config.train.pcgrad is True
    assert config.train.selection_policy == "unified_sen_v20"
    assert config.train.unified_rank_margin_loss_weight > 0
    assert config.train.t2_initialization_checkpoint is None
    assert config.train.evaluate_test_each_epoch is False


def test_unified_rea_v21_restores_t2_relation_evidence_with_energy_calibration():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_rea_v21.yaml")

    assert config.model.variant == "unified_abc_v14"
    assert config.model.image_backbone == "convnext_tiny"
    assert config.model.use_pim is True
    assert config.model.use_ocsr is True
    assert config.model.t2_relation_energy_enabled is True
    assert config.train.stage_b_strict_t1_isolation is True
    assert config.train.pcgrad is False
    assert config.train.t2_adapter_distillation_steps == 4000
    assert config.train.t2_adapter_reconstruction_tolerance_samples == 1
    assert config.train.t2_rank_recovery_epochs == 5
    assert config.train.t2_rank_recovery_epochs < config.train.stage_b_epochs
    assert config.train.t2_b_min_class_recall >= 0.35
    assert config.train.checkpoint_min_rank_1 >= 0.48
    assert config.train.checkpoint_min_rank_3 >= 0.83
    assert config.train.evaluate_test_each_epoch is True


def test_unified_scratch_abc_v22_is_independent_and_validation_selected():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_scratch_abc_v22.yaml")

    assert config.model.variant == "unified_abc_v14"
    assert config.model.t2_null_source_enabled is True
    assert config.model.t2_relation_energy_enabled is False
    assert config.train.stage_a_epochs == 10
    assert config.train.t2_scratch_from_stage_a is True
    assert config.train.t2_initialization_checkpoint is None
    assert config.train.t2_adapter_distillation_steps == 0
    assert config.train.t2_head_warmup_epochs == 2
    assert config.train.t2_a_pretrain_epochs == 8
    assert config.train.t2_b_pretrain_epochs == 4
    assert config.train.stage_b_strict_t1_isolation is True
    assert config.train.evaluate_test_each_epoch is False


def test_unified_sen_v20_5090_preserves_effective_batches_with_more_parallelism():
    base = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_sen_v20.yaml")
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_sen_v20_5090.yaml")

    assert config.model.variant == base.model.variant == "unified_sen_v20"
    assert config.model.image_chunk_size == 4
    assert config.train.stage_b_workers_per_loader == 4
    assert config.train.stage_a_batch_t1_1v1 == 2 * base.train.stage_a_batch_t1_1v1
    assert config.train.stage_a_batch_t1_5v1 == 2 * base.train.stage_a_batch_t1_5v1
    assert config.train.batch_t1_1v1 == 2 * base.train.batch_t1_1v1
    assert config.train.batch_t1_5v1 == 2 * base.train.batch_t1_5v1
    assert config.train.batch_t2 == 2 * base.train.batch_t2
    assert config.train.stage_a_grad_accumulation * 2 == base.train.stage_a_grad_accumulation
    assert config.train.grad_accumulation * 2 == base.train.grad_accumulation
    assert config.train.batch_t2_b == 2 * base.train.batch_t2_b
    assert config.train.t2_b_grad_accumulation * 2 == base.train.t2_b_grad_accumulation
    assert config.train.stage_a_evaluate_test_each_epoch is True
    assert config.train.stage_a_stop_on_validation_decline is True
    assert config.train.stage_a_test_target_1v1_accuracy == pytest.approx(0.9484375)
    assert config.train.stage_a_test_target_1v1_eer == pytest.approx(0.05625)
    assert config.train.stage_a_test_target_5v1_accuracy == pytest.approx(0.9734375)
    assert config.train.stage_a_test_target_5v1_eer == pytest.approx(0.02604166666666665)


def test_v6_strictly_isolates_t1_and_uses_bayesian_counterfactual_training():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v6.yaml")

    assert config.model.variant == "v6"
    assert config.train.t2_loss == "bayesian_v6"
    assert config.train.stage_b_strict_t1_isolation is True
    assert config.train.stage_b_freeze_encoder is True
    assert config.train.stage_b_train_t1 is False
    assert config.train.stage_b_unfreeze_late_encoder is False
    assert config.train.t2_counterfactual_pairs is True
    assert config.data.train_episode_fraction == 1.0


def test_v7_separates_rank_and_open_phases_with_progressive_bayesian_gate():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v7.yaml")

    assert config.model.variant == "v7"
    assert config.train.t2_loss == "progressive_bayesian_v7"
    assert config.train.selection_policy == "progressive_bayesian_v7"
    assert config.train.stage_b_strict_t1_isolation is True
    assert (config.train.t2_rank_phase_epochs, config.train.t2_open_phase_epochs) == (10, 14)
    assert config.train.stage_b_epochs == 24
    assert config.train.checkpoint_require_bayesian_improvement is True


def test_v8_enables_appendable_evidence_prefix_training():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v8.yaml")

    assert config.model.variant == "v8"
    assert list(config.model.stateful_prefix_sizes) == [1, 2, 4, 8]
    assert config.train.t2_loss == "stateful_bayesian_v8"
    assert config.train.selection_policy == "stateful_bayesian_v8"
    assert (config.train.t2_rank_phase_epochs, config.train.t2_open_phase_epochs) == (4, 16)
    assert config.train.stage_b_epochs == 20


def test_v9_uses_v5r1_t1_and_three_isolated_t2_phases():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v9.yaml")

    assert (config.model.variant, config.model.t1_variant) == ("v9", "v5r1")
    assert config.train.t2_loss == "unified_evidence_v9"
    assert config.train.selection_policy == "unified_evidence_v9"
    assert config.train.stage_b_strict_t1_isolation is True
    assert config.train.stage_b_train_t1 is False
    assert config.train.stage_b_freeze_encoder is True
    assert (config.train.t2_representation_phase_epochs,
            config.train.t2_unified_phase_epochs,
            config.train.stage_b_epochs) == (4, 12, 20)


def test_v10_uses_two_independent_t2_training_paths():
    config = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/local_5080_v10.yaml")

    assert (config.model.variant, config.model.t1_variant) == ("v10", "v5r1")
    assert config.train.t2_loss == "dual_evidence_v10"
    assert config.train.selection_policy == "dual_evidence_v10"
    assert config.train.stage_b_strict_t1_isolation is True
    assert config.train.stage_b_train_t1 is False
    assert (config.train.t2_rank_phase_epochs,
            config.train.t2_open_phase_epochs,
            config.train.stage_b_epochs) == (5, 13, 20)


def test_v2_ablation_configs_change_one_input_axis_at_a_time():
    rgb = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/ablation_v2_rgb_only.yaml")
    raw = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/ablation_v2_raw_sequence_only.yaml")
    loss = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/ablation_v2_t2_loss_only.yaml")

    assert (rgb.data.input_pipeline, rgb.model.feature_dim, rgb.train.t2_loss) == (
        "rgb_legacy_ablation", 10, "legacy_joint",
    )
    assert (raw.data.input_pipeline, raw.model.feature_dim, raw.train.t2_loss) == (
        "gray_raw_ablation", 5, "legacy_joint",
    )
    assert (loss.data.input_pipeline, loss.model.feature_dim, loss.train.t2_loss) == (
        "legacy_v1", 10, "factorized_v2",
    )
