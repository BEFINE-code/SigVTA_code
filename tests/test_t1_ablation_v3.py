import argparse
import csv
import json
from pathlib import Path

import pytest
import torch

from ablation.shared.scripts import t1_ablation_control as control
from dvsrc.config import ExperimentConfig, ModelConfig, TrainConfig
from dvsrc.model import T1Head


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "ablation/t1/v3/t1_matrix.json"


def test_t1_matrix_has_eight_single_seed_settings():
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))

    assert matrix["baseline_id"] == "TF0"
    assert matrix["seeds"] == [42]
    assert matrix["training_test_access"] is False
    assert [setting["id"] for setting in matrix["settings"]] == [
        "TF0", "TM1", "TM2", "TF1", "TA1", "TQ1", "TV1", "TE1",
    ]
    assert matrix["split_digest"] == (
        "ba64c09c53119be31f357c089987f706eb3266e4348d8398ff5d88e83dc0c48d"
    )


def test_t1_configs_change_exactly_one_component():
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    base = ExperimentConfig.from_yaml(
        ROOT / "ablation/t1/v3/configs/TF0_full.yaml"
    )
    fields = (
        "fusion", "use_tsa", "use_qrsa",
        "use_t1_residual_verification", "use_t1_evidence_anchor",
    )
    expected = {
        "TM1": ("fusion", "sequence_only"),
        "TM2": ("fusion", "image_only"),
        "TF1": ("fusion", "late"),
        "TA1": ("use_tsa", False),
        "TQ1": ("use_qrsa", False),
        "TV1": ("use_t1_residual_verification", False),
        "TE1": ("use_t1_evidence_anchor", False),
    }
    for setting in matrix["settings"]:
        config = ExperimentConfig.from_yaml(ROOT / setting["config"])
        assert config.train.stage_a_finalize_test is False
        assert config.train.stage_a_evaluate_test_each_epoch is False
        assert config.train.stage_a_selection_policy == "dual_t1"
        differences = [
            name for name in fields
            if getattr(config.model, name) != getattr(base.model, name)
        ]
        if setting["id"] == "TF0":
            assert differences == []
        else:
            name, value = expected[setting["id"]]
            assert differences == [name]
            assert getattr(config.model, name) == value


def test_t1_head_explicit_switches_preserve_default_behavior():
    default = ModelConfig(
        variant="v5r1", t1_variant="v5r1", pretrained=False, hidden_dim=32,
        conformer_heads=4, conformer_ffn=64,
    )
    default_head = T1Head(default, default.t1_variant)
    assert default_head.residual_one_to_one is not None
    assert default_head.evidence_anchored_set is not None

    no_residual = ModelConfig(
        variant="v5r1", t1_variant="v5r1", pretrained=False, hidden_dim=32,
        conformer_heads=4, conformer_ffn=64,
        use_t1_residual_verification=False,
    )
    assert T1Head(no_residual, no_residual.t1_variant).residual_one_to_one is None

    no_anchor = ModelConfig(
        variant="v5r1", t1_variant="v5r1", pretrained=False, hidden_dim=32,
        conformer_heads=4, conformer_ffn=64, use_t1_evidence_anchor=False,
    )
    assert T1Head(no_anchor, no_anchor.t1_variant).evidence_anchored_set is None


def test_stage_a_finalize_test_defaults_to_existing_behavior():
    assert TrainConfig().stage_a_finalize_test is True
    ablation = ExperimentConfig.from_yaml(
        ROOT / "ablation/t1/v3/configs/TF0_full.yaml"
    )
    assert ablation.train.stage_a_finalize_test is False
    assert ablation.data.num_workers == 4
    assert ablation.data.memory_cache_items == 256


def test_t1_controller_validates_every_matrix_setting(tmp_path, monkeypatch):
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(control, "MATRIX_PATH", MATRIX_PATH)
    for setting in matrix["settings"]:
        config = control.config_for(setting, 42, tmp_path / setting["slug"])
        control.validate_contract(config, setting)


def test_finalize_t1_validation_freezes_checkpoint_without_test(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    validation = {
        "t1_1v1": {"overall": {"accuracy": 0.9, "eer": 0.1}},
        "t1_5v1": {"overall": {"accuracy": 0.95, "eer": 0.05}},
    }
    torch.save({
        "validation": validation,
        "config": {"train": {"stage_a_selection_policy": "dual_t1"}},
        "stage_epoch": 3,
    }, output / "best_t1.pt")
    (output / "t1_training_status.json").write_text(
        json.dumps({"running": False, "test_executed": False}), encoding="utf-8",
    )

    control.finalize_validation(argparse.Namespace(output=str(output)))

    record = json.loads(
        (output / "validation_checkpoint.json").read_text(encoding="utf-8")
    )
    assert record["validation_only"] is True
    assert record["selection_policy"] == "dual_t1"
    assert json.loads(
        (output / "validation_metrics.json").read_text(encoding="utf-8")
    ) == validation
    assert not (output / "test_metrics.json").exists()


def test_finalize_t1_validation_rejects_early_test_access(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    torch.save({
        "validation": {"t1_1v1": {}, "t1_5v1": {}},
        "config": {"train": {"stage_a_selection_policy": "dual_t1"}},
    }, output / "best_t1.pt")
    (output / "t1_training_status.json").write_text(
        json.dumps({"running": False, "test_executed": False}), encoding="utf-8",
    )
    (output / "test_metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Test metrics exist"):
        control.finalize_validation(argparse.Namespace(output=str(output)))


def test_begin_t1_test_resumes_same_frozen_access(tmp_path, monkeypatch, capsys):
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    setting = matrix["settings"][0]
    output = tmp_path / "ablation/t1/v3/runs" / setting["slug"] / "seed_42"
    monkeypatch.setattr(control, "ROOT", tmp_path)
    checkpoint = tmp_path / "best_t1.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = control.sha256_file(checkpoint)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [{
        "id": setting["id"], "seed": 42, "config": setting["config"],
        "output": str(output), "checkpoint": str(checkpoint), "sha256": digest,
    }]}), encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    started = output / "test_access_started.json"
    started.write_text(json.dumps({
        "experiment_id": setting["id"], "seed": 42,
        "checkpoint_sha256": digest,
    }), encoding="utf-8")
    monkeypatch.setattr(control, "MATRIX_PATH", MATRIX_PATH)

    try:
        control.begin_test(argparse.Namespace(
            id=setting["id"], seed=42, manifest=str(manifest),
        ))
        assert capsys.readouterr().out.splitlines()[0] == "RESUME"
    finally:
        started.unlink(missing_ok=True)
        for path in (output, output.parent, output.parent.parent):
            try:
                path.rmdir()
            except OSError:
                pass


def test_watchdog_matches_cli_before_output_path():
    script = (
        ROOT / "ablation/shared/scripts/watch_t1_ablation_20m.sh"
    ).read_text(encoding="utf-8")
    assert 'pgrep -f "dvsrc.cli train-t1.*${project_root}"' in script
    assert 'pgrep -f "dvsrc.cli evaluate-t1.*${project_root}"' in script
    assert "t1_watchdog_v2.lock" in script
    assert script.count("9>&-") == 2


def test_published_t1_ablation_results_are_complete_and_weight_free():
    results = ROOT / "ablation/t1/v3/results_seed42"
    run_dirs = sorted((results / "runs").glob("*/seed_42"))

    assert len(run_dirs) == 8
    for run_dir in run_dirs:
        assert (run_dir / "validation_checkpoint.json").is_file()
        assert (run_dir / "validation_metrics.json").is_file()
        assert (run_dir / "test_release_manifest.json").is_file()
        assert (run_dir / "test_metrics.json").is_file()
    assert not list(results.rglob("*.pt"))
    assert not [path for path in results.rglob("*") if path.is_symlink()]


def test_published_t1_ablation_summary_covers_every_split_and_protocol():
    results = ROOT / "ablation/t1/v3/results_seed42"
    with (results / "t1_ablation_metrics_seed42.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8 * 2 * 2
    assert {row["id"] for row in rows} == {
        "TF0", "TM1", "TM2", "TF1", "TA1", "TQ1", "TV1", "TE1",
    }
    assert {row["split"] for row in rows} == {"validation", "test"}
    assert {row["protocol"] for row in rows} == {"1v1", "5v1"}

    manifest = json.loads(
        (results / "evidence/t1_test_checkpoint_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["entries"]) == 8
    assert len({entry["id"] for entry in manifest["entries"]}) == 8
    assert all(len(entry["sha256"]) == 64 for entry in manifest["entries"])
