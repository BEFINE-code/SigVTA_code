import json
from pathlib import Path

import pytest
import torch

from dvsrc.config import ExperimentConfig, ModelConfig
from dvsrc.model import DVSRNet
from dvsrc.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "ablation/t2/configs"


def small_t2_config() -> ExperimentConfig:
    model = ModelConfig(
        variant="t2_abc_v1",
        t1_variant="v5r1",
        pretrained=False,
        image_backbone="resnet18",
        hidden_dim=32,
        conformer_layers=6,
        conformer_heads=4,
        conformer_ffn=64,
        sequence_tokens=8,
        local_tokens=4,
        dropout=0.0,
    )
    config = ExperimentConfig(model=model)
    config.train.stage_b_train_t1 = False
    config.train.stage_b_freeze_encoder = True
    config.train.stage_b_strict_t1_isolation = True
    config.train.t2_head_warmup_epochs = 2
    return config


def trainer_for(config: ExperimentConfig) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model = DVSRNet(config.model)
    return trainer


def test_v3_ablation_matrix_uses_transfer_baseline_and_seed_42():
    matrix = json.loads((ROOT / "ablation/t2/matrix.json").read_text(encoding="utf-8"))

    assert matrix["seeds"] == [42]
    assert [setting["id"] for setting in matrix["settings"]] == [
        "C1", "C2", "T1", "M1", "M2", "R1",
    ]
    assert matrix["baseline_id"] == "F0"
    assert len(matrix["settings"]) * len(matrix["seeds"]) == 6
    assert matrix["baseline"]["checkpoint"] == "models/runs/v3_0_t2_seed_42/best_t2.pt"
    assert (ROOT / matrix["baseline"]["checkpoint"]).is_file()
    assert [setting["slug"] for setting in matrix["settings"]] == [
        "C1_head_only_with_t1_transfer",
        "C2_late_partial_with_t1_transfer",
        "T1_no_t1_transfer",
        "M1_sequence_only_with_t1_transfer",
        "M2_image_only_with_t1_transfer",
        "R1_no_pim_no_ocsr_with_t1_transfer",
    ]
    assert {
        setting["id"]: setting["t2_copy_encoder_from_t1"]
        for setting in matrix["settings"]
    } == {
        "C1": True,
        "C2": True,
        "T1": False,
        "M1": True,
        "M2": True,
        "R1": True,
    }


def test_paper_matrix_tracks_transfer_baseline_and_contextual_modules():
    matrix = json.loads(
        (ROOT / "ablation/t2/PAPER_ABLATION_MATRIX.json").read_text(encoding="utf-8")
    )

    assert matrix["seed"] == 42
    assert [setting["id"] for setting in matrix["settings"]] == [
        "F0", "C1", "C2", "T1", "M1", "M2", "R1", "F1", "R2", "R3",
    ]
    assert [item["id"] for item in matrix["excluded_from_paper"]] == ["L1", "L2"]
    assert [setting["id"] for setting in matrix["settings"]
            if setting["group"] == "transfer_descriptive"] == [
        "F0", "C1", "C2", "T1", "M1", "M2", "R1",
    ]
    assert [setting["id"] for setting in matrix["settings"]
            if setting["group"] == "no_transfer_context"] == ["F1", "R2", "R3"]
    assert matrix["main_model_test"] == {
        "id": "F0",
        "status": "completed_once",
        "result": "models/docs/snapshots/v3_0/test_t2_release_metrics.json",
        "checkpoint": "models/runs/v3_0_t2_seed_42/best_t2.pt",
        "checkpoint_sha256": (
            "953024d86d32a740f89c8870df38a53b17b2637129932d8d709fc6a7e969be71"
        ),
    }
    for setting in matrix["settings"]:
        result = ROOT / setting["result"]
        assert (result if setting["id"] == "F0" else result / "validation_metrics.json").is_file()
        if setting.get("strict_transfer_rerun_config"):
            assert (ROOT / setting["strict_transfer_rerun_config"]).is_file()


def test_formal_baseline_freeze_paths_exist():
    freeze = json.loads(
        (ROOT / "ablation/t2/FINAL_BASELINE_FREEZE.json").read_text(encoding="utf-8")
    )

    assert freeze["t2_copy_encoder_from_t1"] is True
    assert (ROOT / freeze["main_model"]["checkpoint"]).is_file()
    assert (ROOT / freeze["main_model"]["test_result"]).is_file()
    for identity, result in freeze["validation_result_mapping"].items():
        path = ROOT / result
        assert (path if identity == "F0" else path / "validation_metrics.json").is_file()
    supplementary = freeze["initialization_ablation"]["supplementary_test_comparison"]
    assert supplementary["strict_single_variable"] is False
    assert (ROOT / supplementary["no_transfer_result"]).is_file()


def test_module_training_matrix_is_no_transfer_seed_42():
    matrix = json.loads(
        (ROOT / "ablation/t2/module_matrix.json").read_text(encoding="utf-8")
    )

    assert matrix["seeds"] == [42]
    assert [setting["id"] for setting in matrix["settings"]] == ["F1", "R2", "R3"]
    assert all(not setting["t2_copy_encoder_from_t1"] for setting in matrix["settings"])


def test_default_module_rerun_matrix_uses_t1_transfer_seed_42():
    matrix = json.loads(
        (ROOT / "ablation/t2/module_matrix_transfer.json").read_text(encoding="utf-8")
    )

    assert matrix["seeds"] == [42]
    assert matrix["comparison_baseline"]["id"] == "F0"
    assert matrix["comparison_baseline"]["t2_copy_encoder_from_t1"] is True
    assert [setting["id"] for setting in matrix["settings"]] == ["F1", "R2", "R3"]
    assert all(setting["t2_copy_encoder_from_t1"] for setting in matrix["settings"])
    for setting in matrix["settings"]:
        config = ExperimentConfig.from_yaml(ROOT / setting["config"])
        assert config.train.t2_copy_encoder_from_t1 is True


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("base_5090.yaml", {"train.t2_copy_encoder_from_t1": True}),
        ("C1_head_only.yaml", {"train.t2_encoder_trainability": "head_only"}),
        ("C2_late_partial.yaml", {"train.t2_encoder_trainability": "late_partial"}),
        ("T1_no_t1_transfer.yaml", {"train.t2_copy_encoder_from_t1": False}),
        ("M1_sequence_only.yaml", {"model.fusion": "sequence_only"}),
        ("M2_image_only.yaml", {"model.fusion": "image_only"}),
        ("F1_late_fusion.yaml", {
            "model.fusion": "late", "train.t2_copy_encoder_from_t1": False,
        }),
        ("F1_late_fusion_with_t1_transfer.yaml", {
            "model.fusion": "late", "train.t2_copy_encoder_from_t1": True,
        }),
        (
            "R2_no_pim.yaml",
            {
                "model.use_pim": False, "model.use_ocsr": True,
                "train.t2_copy_encoder_from_t1": False,
            },
        ),
        ("R2_no_pim_with_t1_transfer.yaml", {
            "model.use_pim": False, "model.use_ocsr": True,
            "train.t2_copy_encoder_from_t1": True,
        }),
        (
            "R3_no_ocsr.yaml",
            {
                "model.use_pim": True, "model.use_ocsr": False,
                "train.t2_copy_encoder_from_t1": False,
            },
        ),
        ("R3_no_ocsr_with_t1_transfer.yaml", {
            "model.use_pim": True, "model.use_ocsr": False,
            "train.t2_copy_encoder_from_t1": True,
        }),
    ],
)
def test_v3_ablation_configs_keep_common_contract(name, expected):
    config = ExperimentConfig.from_yaml(CONFIGS / name)

    assert config.model.variant == "t2_abc_v1"
    assert config.train.stage_b_epochs == 16
    assert config.train.selection_policy == "unified_abc_v14"
    assert config.train.unified_official_loss_weight == 0.0
    assert config.train.checkpoint_factor_gates_enabled is False
    assert config.train.evaluate_test_each_epoch is False
    assert config.train.t2_factor_b_evaluation_enabled is True
    assert config.train.t2_freeze_inactive_modalities is True
    assert config.train.t2_freeze_inactive_relation_modules is True
    for dotted_name, value in expected.items():
        section, field = dotted_name.split(".")
        assert getattr(getattr(config, section), field) == value


def test_head_only_and_late_partial_apply_exact_t2_encoder_trainability():
    head_config = small_t2_config()
    head_config.train.t2_encoder_trainability = "head_only"
    head_trainer = trainer_for(head_config)
    head_trainer._prepare_stage_b_modules(stage_epoch=2)
    assert not any(parameter.requires_grad for parameter in head_trainer.model.t2_encoder.parameters())
    assert all(parameter.requires_grad for parameter in head_trainer.model.t2.parameters())

    late_config = small_t2_config()
    late_config.train.t2_encoder_trainability = "late_partial"
    late_trainer = trainer_for(late_config)
    late_trainer._prepare_stage_b_modules(stage_epoch=2)
    names = {
        name for name, parameter in late_trainer.model.named_parameters()
        if parameter.requires_grad
    }
    assert any(name.startswith("t2_encoder.sequence.blocks.4.") for name in names)
    assert any(name.startswith("t2_encoder.sequence.blocks.5.") for name in names)
    assert any(name.startswith("t2_encoder.sequence.norm.") for name in names)
    assert any(name.startswith("t2_encoder.image.stage3_projection.") for name in names)
    assert any(name.startswith("t2_encoder.image.stage4_projection.") for name in names)
    assert any(name.startswith("t2_encoder.image.global_projection.") for name in names)
    assert any(name.startswith("t2_encoder.fusion.") for name in names)
    assert not any(name.startswith("t2_encoder.sequence.blocks.3.") for name in names)
    assert not any(name.startswith("t2_encoder.image.features.") for name in names)


def test_no_t1_transfer_keeps_seeded_t2_encoder_initialization(tmp_path):
    target_config = small_t2_config()
    target_config.model.fusion = "sequence_only"
    target_config.train.t2_copy_encoder_from_t1 = False
    source_model_config = ModelConfig(**target_config.model.__dict__)
    source_model_config.variant = "v5r1"
    source_model_config.t1_variant = None

    torch.manual_seed(10)
    source = DVSRNet(source_model_config)
    with torch.no_grad():
        next(source.encoder.parameters()).fill_(0.25)
    checkpoint = tmp_path / "best_t1.pt"
    torch.save({
        "epoch": 3,
        "model": source.state_dict(),
        "config": ExperimentConfig(model=source_model_config).to_dict(),
    }, checkpoint)

    torch.manual_seed(11)
    trainer = trainer_for(target_config)
    trainer.device = torch.device("cpu")
    trainer.output = tmp_path / "run"
    trainer.output.mkdir()
    fresh_t2 = next(trainer.model.t2_encoder.parameters()).detach().clone()

    result = trainer.initialize_isolated_stage_b(checkpoint)

    torch.testing.assert_close(
        next(trainer.model.encoder.parameters()), next(source.encoder.parameters()),
    )
    torch.testing.assert_close(next(trainer.model.t2_encoder.parameters()), fresh_t2)
    assert result["report"]["transferred_t2_encoder_tensors"] == 0
    assert result["report"]["t2_copy_encoder_from_t1"] is False


def test_relation_ablation_freezes_the_disabled_pim_and_ocsr_parameters():
    config = small_t2_config()
    config.model.use_pim = False
    config.model.use_ocsr = False
    config.train.t2_freeze_inactive_relation_modules = True
    trainer = trainer_for(config)

    trainer._prepare_stage_b_modules(stage_epoch=2)

    trainable = {
        name for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }
    assert not any(name.startswith("t2.relation.pim.") for name in trainable)
    assert not any(name.startswith("t2.relation.ocsr.") for name in trainable)
    assert any(name.startswith("t2.relation.simple_rank.") for name in trainable)
    assert any(name.startswith("t2.relation.simple_exist.") for name in trainable)


def test_individual_relation_ablations_freeze_only_inactive_paths():
    no_pim = small_t2_config()
    no_pim.model.use_pim = False
    no_pim.model.use_ocsr = True
    no_pim.train.t2_freeze_inactive_relation_modules = True
    no_pim_trainer = trainer_for(no_pim)
    no_pim_trainer._prepare_stage_b_modules(stage_epoch=2)
    no_pim_trainable = {
        name for name, parameter in no_pim_trainer.model.named_parameters()
        if parameter.requires_grad
    }
    assert not any(name.startswith("t2.relation.pim.") for name in no_pim_trainable)
    assert any(name.startswith("t2.relation.ocsr.") for name in no_pim_trainable)
    assert not any(name.startswith("t2.relation.simple_rank.") for name in no_pim_trainable)

    no_ocsr = small_t2_config()
    no_ocsr.model.use_pim = True
    no_ocsr.model.use_ocsr = False
    no_ocsr.train.t2_freeze_inactive_relation_modules = True
    no_ocsr_trainer = trainer_for(no_ocsr)
    no_ocsr_trainer._prepare_stage_b_modules(stage_epoch=2)
    no_ocsr_trainable = {
        name for name, parameter in no_ocsr_trainer.model.named_parameters()
        if parameter.requires_grad
    }
    assert any(name.startswith("t2.relation.pim.") for name in no_ocsr_trainable)
    assert not any(name.startswith("t2.relation.ocsr.") for name in no_ocsr_trainable)
    assert any(name.startswith("t2.relation.simple_rank.") for name in no_ocsr_trainable)


def test_late_fusion_freezes_unused_ms_caf_parameters():
    config = small_t2_config()
    config.model.fusion = "late"
    config.train.t2_freeze_inactive_modalities = True
    trainer = trainer_for(config)

    trainer._prepare_stage_b_modules(stage_epoch=2)

    trainable = {
        name for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }
    assert any(name.startswith("t2_encoder.sequence.") for name in trainable)
    assert any(name.startswith("t2_encoder.image.") for name in trainable)
    assert any(name.startswith("t2_encoder.late.") for name in trainable)
    assert not any(name.startswith("t2_encoder.fusion.") for name in trainable)
