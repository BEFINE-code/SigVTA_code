from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"Cyclic config inheritance: {chain}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise ValueError(f"Config extends must be a path string: {resolved}")
    base = _load_yaml(resolved.parent / parent, (*stack, resolved))
    return _deep_merge(base, raw)


@dataclass
class DataConfig:
    dataset_root: str = "datasets/signatures"
    benchmark_root: str = "datasets/protocols/t1"
    t2_benchmark_root: str | None = None
    fold: int = 0
    input_pipeline: str = "legacy_v1"
    image_height: int = 256
    image_width: int = 512
    image_margin: int = 12
    image_line_width: int = 2
    image_supersample: int = 2
    image_speed_cap_mm_s: float = 800.0
    raw_time_scale_seconds: float = 60.0
    raw_position_scale_mm: float = 100.0
    max_points: int = 6144
    verify_hashes: bool = False
    cache_images: bool = True
    image_cache_root: str | None = None
    memory_cache: bool = True
    memory_cache_items: int = 256
    num_workers: int = 8
    train_episode_fraction: float = 1.0
    train_subset_seed: int = 42
    zero_channels: tuple[str, ...] = ()
    time_align: str | None = None
    time_align_points: int = 256
    time_align_constant_pressure: bool = False
    time_align_duration_ms: float | None = None

    def __post_init__(self) -> None:
        self.zero_channels = tuple(self.zero_channels or ())
        if isinstance(self.zero_channels, str) or any(not isinstance(name, str) for name in self.zero_channels):
            raise ValueError("zero_channels must be a sequence of feature names")
        if self.time_align == "":
            self.time_align = None
        if self.time_align not in {None, "arc_length"}:
            raise ValueError(f"Unsupported time_align: {self.time_align}")
        if self.time_align_points < 2:
            raise ValueError("time_align_points must be at least 2")


@dataclass
class ModelConfig:
    variant: str = "v2"
    t1_variant: str | None = None  # simple_v1: global-only pairs and sorted-logit 5v1 MLP
    t1_pair_features: str = "diff_product"
    t1_set_aggregation: str = "sorted_mlp"
    t1_sequence_residual: bool = False
    t1_pair_stabilize: bool = False
    feature_dim: int = 10
    sequence_stem: str = "legacy_stride8"
    sequence_model_tokens: int = 128
    hidden_dim: int = 256
    conformer_layers: int = 6
    conformer_heads: int = 8
    conformer_ffn: int = 1024
    conformer_kernel: int = 31
    sequence_tokens: int = 32
    local_tokens: int = 16
    dropout: float = 0.1
    image_backbone: str = "convnext_tiny"
    pretrained: bool = True
    fusion: str = "ms_caf"
    use_tsa: bool = True
    use_qrsa: bool = True
    use_t1_residual_verification: bool = True
    use_t1_evidence_anchor: bool = True
    use_pim: bool = True
    use_ocsr: bool = True
    sinkhorn_iters: int = 12
    sinkhorn_epsilon: float = 0.08
    image_chunk_size: int = 12
    stateful_prefix_sizes: tuple[int, ...] = (1, 2, 4, 8)
    t2_adapter_bottleneck: int = 32
    t2_relation_energy_enabled: bool = False
    t2_null_source_enabled: bool = False


@dataclass
class TrainConfig:
    seed: int = 42
    stage_a_epochs: int = 25
    stage_a_selection_policy: str = "t1_5v1"
    stage_a_evaluate_test_each_epoch: bool = False
    stage_a_finalize_test: bool = True
    stage_a_stop_on_validation_decline: bool = False
    stage_a_test_target_1v1_accuracy: float | None = None
    stage_a_test_target_1v1_eer: float | None = None
    stage_a_test_target_5v1_accuracy: float | None = None
    stage_a_test_target_5v1_eer: float | None = None
    stage_b_epochs: int = 40
    stage_b_workers_per_loader: int | None = None
    batch_t1_1v1: int = 16
    batch_t1_5v1: int = 4
    batch_t2: int = 2
    t2_unknown_fraction: float = 0.5
    t1_balance_strata: bool = True
    grad_accumulation: int = 4
    stage_a_batch_t1_1v1: int = 32
    stage_a_batch_t1_5v1: int = 8
    stage_a_grad_accumulation: int = 2
    eval_batch_multiplier: int = 4
    backbone_lr: float = 2e-5
    encoder_lr: float = 1e-4
    head_lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    patience: int = 10
    amp: bool = True
    t1_genuine_identity_only: bool = False
    pcgrad: bool = True
    t2_loss: str = "legacy_joint"
    selection_policy: str = "legacy_v2"
    reshuffle_on_cycle: bool = False
    stage_b_train_t1: bool = True
    stage_b_freeze_encoder: bool = False
    stage_b_unfreeze_late_encoder: bool = False
    t2_encoder_trainability: str = "full"
    t2_copy_encoder_from_t1: bool = True
    t2_freeze_inactive_modalities: bool = False
    t2_freeze_inactive_relation_modules: bool = False
    t2_pair_source_absent: bool = False
    stage_b_strict_t1_isolation: bool = False
    t2_counterfactual_pairs: bool = False
    t2_pair_margin: float = 0.5
    t2_pair_loss_weight: float = 0.5
    t2_release_warmup_epochs: int = 2
    t2_release_epochs: int = 6
    t2_rank_phase_epochs: int = 0
    t2_open_phase_epochs: int = 0
    t2_query_loss_weight: float = 0.5
    t2_monotonic_loss_weight: float = 0.5
    t2_stateful_loss_weight: float = 0.75
    t2_arrival_loss_weight: float = 0.25
    t2_representation_phase_epochs: int = 0
    t2_unified_phase_epochs: int = 0
    t2_head_warmup_epochs: int = 0
    t2_balanced_a_enabled: bool = False
    t2_a_pretrain_epochs: int = 0
    t2_a_patience: int = 3
    batch_t2_a: int = 8
    t2_a_grad_accumulation: int = 1
    t2_a_loss_weight: float = 1.0
    t2_a_max_balanced_accuracy_drop: float = 0.02
    t2_a_min_class_recall: float = 0.20
    t2_balanced_b_enabled: bool = False
    t2_factor_b_evaluation_enabled: bool = False
    t2_b_pretrain_epochs: int = 0
    t2_b_patience: int = 2
    batch_t2_b: int = 1
    t2_b_grad_accumulation: int = 8
    t2_b_loss_weight: float = 1.0
    t2_b_max_balanced_accuracy_drop: float = 0.02
    t2_b_min_class_recall: float = 0.20
    t2_initialization_checkpoint: str | None = None
    t2_scratch_from_stage_a: bool = False
    t2_adapter_distillation_steps: int = 0
    t2_adapter_global_distillation_weight: float = 1.0
    t2_adapter_local_distillation_weight: float = 0.5
    t2_adapter_internal_distillation_weight: float = 0.0
    t2_adapter_min_rank_1: float = 0.0
    t2_adapter_min_rank_3: float = 0.0
    t2_adapter_reconstruction_tolerance_samples: int = 0
    t2_relation_distillation_epochs: int = 0
    t2_rank_recovery_epochs: int = 0
    t2_distillation_temperature: float = 2.0
    t2_rank_distillation_weight: float = 0.0
    t2_hard_negative_loss_weight: float = 0.0
    t2_b_distillation_weight: float = 0.0
    unified_joint_loss_weight: float = 1.0
    unified_bridge_loss_weight: float = 0.1
    unified_official_loss_weight: float = 0.2
    unified_factor_a_loss_weight: float = 0.0
    unified_factor_b_loss_weight: float = 0.0
    unified_rank_loss_weight: float = 0.0
    unified_rank_margin_loss_weight: float = 0.0
    unified_t1_max_eer_increase: float = 0.01
    evaluate_test_each_epoch: bool = False
    checkpoint_gate_warmup_epochs: int = 10
    checkpoint_min_hierarchical_accuracy: float = 0.5
    checkpoint_min_rank_1: float = 0.303125
    checkpoint_min_rank_3: float = 0.0
    checkpoint_min_source_absent_accuracy: float = 0.296875
    checkpoint_min_rf_accuracy: float = 0.95
    checkpoint_require_bayesian_improvement: bool = False
    checkpoint_factor_gates_enabled: bool = True
    output_dir: str = "models/development/runs/fold_0_seed_42"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = _load_yaml(Path(path))
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
