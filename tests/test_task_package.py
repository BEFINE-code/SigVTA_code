import json

import pytest
import torch

from dvsrc.config import ExperimentConfig, ModelConfig
from dvsrc.model import DVSRNet
from dvsrc.task_package import (
    PACKAGE_SCHEMA,
    TaskInferenceModel,
    export_task_package,
    load_task_package,
)
from dvsrc.trainer import Trainer


def _config() -> ModelConfig:
    return ModelConfig(
        variant="t2_abc_v1",
        t1_variant="v5r1",
        pretrained=False,
        fusion="sequence_only",
        hidden_dim=32,
        conformer_layers=1,
        conformer_heads=4,
        conformer_ffn=64,
        sequence_tokens=8,
        local_tokens=4,
        dropout=0.0,
    )


def _batch(protocol: str) -> dict[str, torch.Tensor | str]:
    return {
        "protocol": protocol,
        "sequence": torch.randn(6, 48, 10),
        "sequence_mask": torch.ones(6, 48, dtype=torch.bool),
        "image": torch.zeros(6, 3, 32, 32),
        "anchors": torch.zeros(6, 8, 6),
        "anchor_mask": torch.ones(6, 8, dtype=torch.bool),
        "query_index": torch.tensor([5]),
        "set_index": torch.tensor([[0, 1, 2, 3, 4]]),
        "set_mask": torch.ones(1, 5, dtype=torch.bool),
        "target_index": torch.tensor([2]),
        "episode_type_index": torch.tensor([0]),
    }


def _checkpoint(path, model: DVSRNet, config: ModelConfig) -> None:
    torch.save({
        "stage": "stage_b",
        "epoch": 3,
        "model": model.state_dict(),
        "config": ExperimentConfig(model=config).to_dict(),
    }, path)


def test_v23_configs_use_independent_same_architecture_encoders():
    t1 = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_dual_abc_v23_t1.yaml")
    t2 = ExperimentConfig.from_yaml(".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_dual_abc_v23.yaml")

    assert t1.model.variant == "v5r1"
    assert (t2.model.variant, t2.model.t1_variant) == ("t2_abc_v1", "v5r1")
    assert t1.model.hidden_dim == t2.model.hidden_dim == 256
    assert t1.model.conformer_layers == t2.model.conformer_layers == 6
    assert t1.model.image_backbone == t2.model.image_backbone == "convnext_tiny"
    assert t1.model.fusion == t2.model.fusion == "ms_caf"
    assert t2.train.stage_b_strict_t1_isolation is True
    assert t2.train.stage_b_train_t1 is False
    assert t2.train.stage_b_freeze_encoder is True
    assert t2.train.t2_initialization_checkpoint is None
    assert t2.train.evaluate_test_each_epoch is False


def test_v23_5090_profiles_preserve_effective_batches_and_raise_parallelism():
    local_t1 = ExperimentConfig.from_yaml(
        ".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_dual_abc_v23_t1.yaml"
    )
    server_t1 = ExperimentConfig.from_yaml(
        ".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_dual_abc_v23_5090_t1.yaml"
    )
    local_t2 = ExperimentConfig.from_yaml(
        ".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_dual_abc_v23.yaml"
    )
    server_t2 = ExperimentConfig.from_yaml(
        ".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_dual_abc_v23_5090.yaml"
    )

    assert server_t1.model.variant == local_t1.model.variant == "v5r1"
    assert server_t2.model.variant == local_t2.model.variant == "t2_abc_v1"
    assert server_t1.model.image_chunk_size == server_t2.model.image_chunk_size == 12
    assert server_t1.data.num_workers == server_t2.data.num_workers == 8
    assert server_t1.data.memory_cache_items == server_t2.data.memory_cache_items == 1024
    assert server_t2.train.stage_b_workers_per_loader == 4

    assert (
        server_t1.train.stage_a_batch_t1_1v1
        * server_t1.train.stage_a_grad_accumulation
        == local_t1.train.stage_a_batch_t1_1v1
        * local_t1.train.stage_a_grad_accumulation
    )
    assert (
        server_t1.train.stage_a_batch_t1_5v1
        * server_t1.train.stage_a_grad_accumulation
        == local_t1.train.stage_a_batch_t1_5v1
        * local_t1.train.stage_a_grad_accumulation
    )
    assert (
        server_t2.train.batch_t2 * server_t2.train.grad_accumulation
        == local_t2.train.batch_t2 * local_t2.train.grad_accumulation
    )
    assert (
        server_t2.train.batch_t2_b * server_t2.train.t2_b_grad_accumulation
        == local_t2.train.batch_t2_b * local_t2.train.t2_b_grad_accumulation
    )


def test_v24_rank_continuation_directly_optimizes_and_selects_ranking():
    config = ExperimentConfig.from_yaml(
        ".Trash/workspace_reorg_20260907_131226/benchmark/configs/unified_dual_abc_v24_rank_5090.yaml"
    )

    assert config.model.variant == "t2_abc_v1"
    assert config.train.stage_b_train_t1 is False
    assert config.train.stage_b_strict_t1_isolation is True
    assert config.train.unified_rank_loss_weight == pytest.approx(1.50)
    assert config.train.unified_rank_margin_loss_weight == pytest.approx(0.50)
    assert config.train.unified_factor_b_loss_weight == pytest.approx(0.50)
    assert config.train.selection_policy == "unified_abc_v14"
    assert config.train.evaluate_test_each_epoch is False


def test_task_packages_match_full_training_model_outputs(tmp_path):
    torch.manual_seed(52)
    config = _config()
    model = DVSRNet(config).eval()
    assert model.t2_encoder is not None
    model.t2_encoder.load_state_dict(model.encoder.state_dict())
    t2_checkpoint = tmp_path / "best.pt"
    _checkpoint(t2_checkpoint, model, config)
    t1_config = _config()
    t1_config.variant = "v5r1"
    t1_config.t1_variant = None
    t1_model_source = DVSRNet(t1_config).eval()
    t1_checkpoint = tmp_path / "best_t1.pt"
    _checkpoint(t1_checkpoint, t1_model_source, t1_config)
    calibration = tmp_path / "t1_calibration.json"
    calibration.write_text(json.dumps({"temperature": 1.25}), encoding="utf-8")

    t1_package = tmp_path / "best_t1_inference.pt"
    t2_package = tmp_path / "best_t2_inference.pt"
    t1_report = export_task_package(
        t1_checkpoint, "t1", t1_package, calibration_path=calibration,
    )
    t2_report = export_task_package(
        t2_checkpoint, "t2", t2_package, initialized_from_t1=t1_checkpoint,
    )
    t1_model, t1_metadata = load_task_package(t1_package)
    t2_model, t2_metadata = load_task_package(t2_package)

    t1_batch = _batch("t1_5v1")
    t2_batch = _batch("t2")
    with torch.no_grad():
        full_t1 = t1_model_source(t1_batch)
        packaged_t1 = t1_model(t1_batch)
        full_t2 = model(t2_batch)
        packaged_t2 = t2_model(t2_batch)

    torch.testing.assert_close(packaged_t1["case_logit"], full_t1["case_logit"])
    torch.testing.assert_close(
        packaged_t2["official_probability"], full_t2["official_probability"],
    )
    assert t1_report["schema"] == t2_report["schema"] == PACKAGE_SCHEMA
    assert t1_metadata["calibration"] == {"temperature": 1.25}
    assert t2_metadata["provenance"]["initialized_from_t1"]["sha256"]
    assert not any(name.startswith("t2_") for name, _ in t1_model.named_parameters())
    assert not any(name.startswith("t1") for name, _ in t2_model.named_parameters())


def test_v23_stage_b_copies_t1_encoder_then_trains_only_t2_branch(tmp_path):
    t1_config = _config()
    t1_config.variant = "v5r1"
    t1_config.t1_variant = None
    torch.manual_seed(53)
    t1_model = DVSRNet(t1_config)
    checkpoint = tmp_path / "best_t1.pt"
    _checkpoint(checkpoint, t1_model, t1_config)

    experiment = ExperimentConfig(model=_config())
    experiment.train.output_dir = str(tmp_path / "t2")
    experiment.train.t2_head_warmup_epochs = 2
    experiment.train.stage_b_freeze_encoder = True
    experiment.train.stage_b_train_t1 = False
    experiment.train.stage_b_strict_t1_isolation = True
    trainer = Trainer.__new__(Trainer)
    trainer.config = experiment
    trainer.device = torch.device("cpu")
    trainer.model = DVSRNet(experiment.model)
    trainer.output = tmp_path / "t2"
    trainer.output.mkdir()

    result = trainer.initialize_isolated_stage_b(checkpoint)

    assert trainer.model.t2_encoder is not None
    for key, value in t1_model.encoder.state_dict().items():
        torch.testing.assert_close(trainer.model.encoder.state_dict()[key], value)
        torch.testing.assert_close(trainer.model.t2_encoder.state_dict()[key], value)
    assert result["report"]["transferred_t2_encoder_tensors"] > 0

    trainer._prepare_stage_b_modules(stage_epoch=0)
    assert not any(parameter.requires_grad for parameter in trainer.model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.t1.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.t2_encoder.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.t2.parameters())

    trainer._prepare_stage_b_modules(stage_epoch=2)
    assert all(parameter.requires_grad for parameter in trainer.model.t2_encoder.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.t2.parameters())


def test_task_package_rejects_wrong_protocol_and_incomplete_state(tmp_path):
    config = _config()
    with pytest.raises(ValueError, match="cannot execute protocol"):
        TaskInferenceModel(config, "t1")(_batch("t2"))

    model = DVSRNet(config)
    checkpoint = tmp_path / "broken.pt"
    payload = {
        "model": model.state_dict(),
        "config": ExperimentConfig(model=config).to_dict(),
    }
    payload["model"].pop(next(key for key in payload["model"] if key.startswith("t2_encoder.")))
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        export_task_package(checkpoint, "t2", tmp_path / "broken_package.pt")
