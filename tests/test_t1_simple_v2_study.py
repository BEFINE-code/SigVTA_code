import copy

import pytest
import torch

from ablation.t1.simple_v2 import run as control
from ablation.t1.simple_v2 import supervise as watch
from dvsrc.calibration import Calibrator
from dvsrc.utils import atomic_json


def test_eight_configs_are_isolated_and_declared():
    control.validate_configs()
    matrix = control.matrix()
    assert len(matrix["settings"]) == 8
    assert "S2C1" not in {s["id"] for s in matrix["settings"]}
    assert matrix["baseline_id"] == "S2F0"
    assert len([s for s in matrix["settings"] if s["group"] == "diagnostic"]) == 1
    for setting in matrix["settings"]:
        config = control.config_for(setting)
        assert config.model.use_tsa and config.train.t1_genuine_identity_only
        assert "ablation/t1/runs" in config.train.output_dir.replace("\\", "/")
    old = control.read(control.ROOT / "ablation/t1/simple_v1/matrix.json")
    assert {s["id"] for s in old["settings"]}.isdisjoint(s["id"] for s in matrix["settings"])


def test_undeclared_change_rejected(monkeypatch):
    original = control.config_for
    def changed(setting):
        config = original(setting)
        config.model.use_tsa = False
        return config
    monkeypatch.setattr(control, "config_for", changed)
    with pytest.raises(ValueError):
        control.validate_configs()


@pytest.fixture
def finished(tmp_path, monkeypatch):
    matrix = control.matrix()
    configs = {s["id"]: control.config_for(s) for s in matrix["settings"]}
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "STUDY", tmp_path / "ablation/t1")
    atomic_json(control.STUDY / "matrix.json", matrix)
    atomic_json(tmp_path / matrix["benchmark_root"] / "audit_report.json",
                {"ok": True, "split_digest": matrix["split_digest"]})
    for split in ("train", "val", "test"):
        for protocol in ("t1_1v1", "t1_5v1"):
            atomic_json(tmp_path / matrix["benchmark_root"] / "episodes/fold_0" / f"{split}_{protocol}.jsonl", {})
    def config_for(setting):
        cfg = copy.deepcopy(configs[setting["id"]])
        cfg.train.output_dir = str(control.run_dir(setting).resolve())
        return cfg
    monkeypatch.setattr(control, "config_for", config_for)
    monkeypatch.setattr(control, "require_gpu", lambda: None)
    atomic_json(tmp_path / next(iter(configs.values())).data.dataset_root / "manifest.json", {})
    metrics = {p: {"overall": {"accuracy": .9, "eer": .1, "roc_auc": .95}}
               for p in ("t1_1v1", "t1_5v1")}
    for setting in matrix["settings"]:
        output = control.run_dir(setting)
        atomic_json(output / "run_manifest.json", {"identity": control.identity(setting)})
        atomic_json(output / "t1_training_status.json", {"running": False, "test_executed": False})
        atomic_json(output / "t1_calibration.json", Calibrator().__dict__)
        atomic_json(output / "validation_metrics.json", metrics)
        torch.save({"config": config_for(setting).to_dict(), "validation": metrics,
                    "stage_epoch": 2}, output / "best_t1.pt")
        control.finalize_validation(setting)
    return matrix, metrics


def test_all_eight_required_before_freeze(finished):
    matrix, _ = finished
    assert watch.IDS == tuple(s["id"] for s in matrix["settings"])
    assert watch.phase(control.STUDY) == "freeze"
    setting = matrix["settings"][0]
    path = control.run_dir(setting) / "validation_checkpoint.json"
    record = control.read(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        control.freeze()
    atomic_json(path, record)
    assert len(control.freeze()["entries"]) == 8
    assert control.freeze() == control.freeze()
    assert watch.phase(control.STUDY) == "test"


@pytest.mark.parametrize("target", ["checkpoint", "calibration", "episodes", "dataset"])
def test_changed_frozen_inputs_block_resume(finished, target):
    matrix, _ = finished
    control.freeze()
    setting = matrix["settings"][0]
    output = control.run_dir(setting)
    paths = {"checkpoint": output / "best_t1.pt", "calibration": output / "t1_calibration.json",
             "episodes": control.ROOT / matrix["benchmark_root"] / "episodes/fold_0/train_t1_1v1.jsonl",
             "dataset": control.ROOT / control.config_for(setting).data.dataset_root / "manifest.json"}
    paths[target].write_bytes(b"changed fixture")
    with pytest.raises(ValueError):
        control.test([setting])


def test_frozen_test_once_and_export_no_weights(finished, monkeypatch):
    matrix, metrics = finished
    with pytest.raises(ValueError, match="frozen"):
        control.test(matrix["settings"])
    control.freeze()
    calls = []
    class FakeTrainer:
        def __init__(self, config):
            pass
        def load_checkpoint(self, path):
            calls.append(path)
        def evaluate_split(self, split, calibrator, export, protocols):
            assert split == "test"
            assert calibrator.__dict__ == Calibrator().__dict__
            return metrics
        def fit_t1_calibration(self):
            pytest.fail("Test must never refit calibration")
    monkeypatch.setattr("dvsrc.trainer.Trainer", FakeTrainer)
    control.test(matrix["settings"])
    control.test(matrix["settings"])
    assert len(calls) == 8
    assert watch.phase(control.STUDY) == "export"
    control.export()
    destination = control.ROOT / matrix["results_root"]
    summary = control.read(destination / "summary.json")
    assert len(summary["rows"]) == 48
    assert summary["test_reused_after_v1"] is True
    assert summary["test_completed"] == 8
    assert watch.phase(control.STUDY) == "complete"
    assert not list(destination.rglob("*.pt"))
    assert not any(p.is_symlink() for p in destination.rglob("*"))
    with pytest.raises(FileExistsError):
        control.export()


def test_train_resumes_only_own_last_and_keeps_logs(finished, monkeypatch):
    matrix, _ = finished
    setting = matrix["settings"][0]
    output = control.run_dir(setting)
    (output / "validation_checkpoint.json").unlink()
    (output / "last_t1.pt").write_bytes(b"fixture")
    (output / "stdout.log").write_bytes(b"previous\n")
    atomic_json(output / "t1_training_status.json", {"running": True, "test_executed": False})
    def launch(command, **kwargs):
        assert command[-2:] == ["--resume", str(output / "last_t1.pt")]
        kwargs["stdout"].write(b"resumed\n")
        atomic_json(output / "t1_training_status.json", {"running": False, "test_executed": False})
    monkeypatch.setattr(control.subprocess, "run", launch)
    control.train([setting])
    control.train([setting])
    assert (output / "stdout.log").read_bytes() == b"previous\nresumed\n"


def test_new_study_execution_lock(finished):
    with control.study_lock():
        with pytest.raises(OSError):
            with control.study_lock():
                pytest.fail("Duplicate execution acquired lock")
    with control.study_lock():
        pass
