import copy

import pytest
import torch

from ablation.t1.simple_v1 import run as control
from dvsrc.calibration import Calibrator
from dvsrc.config import ModelConfig
from dvsrc.model import GlobalPairRelation, SignatureEncoding, T1Head
from dvsrc.utils import atomic_json


def test_new_matrix_is_separate_and_tsa_always_enabled():
    control.validate_configs()
    m = control.matrix()
    assert len(m["settings"]) == 7
    assert m["runs_root"] == "ablation/t1/simple_v1/runs"
    assert m["freeze_manifest"].startswith("ablation/t1/simple_v1/")
    for setting in m["settings"]:
        config = control.config_for(setting)
        assert config.model.use_tsa
        assert config.model.t1_variant == "simple_v1"
        assert not config.train.stage_a_finalize_test
    old = control.read(control.ROOT / "ablation/t1/v3/t1_matrix.json")
    assert len(old["settings"]) == 8
    assert {s["id"] for s in old["settings"]}.isdisjoint(s["id"] for s in m["settings"])


@pytest.mark.parametrize("field,value", [("use_tsa", False), ("use_qrsa", True)])
def test_rejects_undeclared_model_changes(monkeypatch, field, value):
    original = control.config_for
    def altered(setting):
        config = original(setting)
        setattr(config.model, field, value)
        return config
    monkeypatch.setattr(control, "config_for", altered)
    with pytest.raises(ValueError):
        control.validate_configs()


@pytest.mark.parametrize("mode", ["diff_only", "product_only"])
def test_single_global_feature_reaches_projection(mode):
    torch.manual_seed(42)
    relation = GlobalPairRelation(8, 0, mode)
    a, b = torch.randn(2, 8, requires_grad=True), torch.randn(2, 8, requires_grad=True)
    seen = []
    handle = relation.mlp[0].register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    pair, logits = relation(a, b)
    handle.remove()
    torch.testing.assert_close(seen[0], (a - b).abs() if mode == "diff_only" else a * b)
    assert relation.mlp[0].in_features == 8
    logits.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_mean_arm_removes_mlp_and_preserves_single_reference():
    def encoding(n=None):
        shape = (2, 8) if n is None else (2, n, 8)
        z = torch.randn(shape)
        return SignatureEncoding(z, z.unsqueeze(-2), torch.ones(*shape[:-1], 1, dtype=torch.bool),
                                 torch.zeros(*shape[:-1], 1, 2), z, z, {})
    config = ModelConfig(hidden_dim=8, t1_variant="simple_v1", t1_set_aggregation="mean")
    head = T1Head(config).eval()
    assert head.five_to_one is None
    for n in (1, 5):
        result = head(encoding(n), encoding(), torch.ones(2, n, dtype=torch.bool))
        torch.testing.assert_close(result["case_logit"], result["pair_logits"].mean(dim=1))


@pytest.fixture
def finished_study(tmp_path, monkeypatch):
    m = control.matrix()
    configs = {s["id"]: control.config_for(s) for s in m["settings"]}
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "STUDY", tmp_path / "ablation/t1/simple_v1")
    atomic_json(control.STUDY / "matrix.json", m)
    atomic_json(tmp_path / m["benchmark_root"] / "audit_report.json",
                {"ok": True, "split_digest": m["split_digest"]})
    def config_for(setting):
        cfg = copy.deepcopy(configs[setting["id"]])
        cfg.train.output_dir = str(control.run_dir(setting).resolve())
        return cfg
    monkeypatch.setattr(control, "config_for", config_for)
    monkeypatch.setattr(control, "require_gpu", lambda: None)
    metrics = {p: {"overall": {"accuracy": 0.9, "eer": 0.1, "roc_auc": 0.95}}
               for p in ("t1_1v1", "t1_5v1")}
    for setting in m["settings"]:
        output = control.run_dir(setting)
        atomic_json(output / "run_manifest.json", {"identity": control.identity(setting)})
        atomic_json(output / "t1_training_status.json", {"running": False, "test_executed": False})
        atomic_json(output / "t1_calibration.json", Calibrator().__dict__)
        atomic_json(output / "validation_metrics.json", metrics)
        torch.save({"config": config_for(setting).to_dict(), "validation": metrics,
                    "stage_epoch": 2}, output / "best_t1.pt")
        control.finalize_validation(setting)
    return m, metrics


def test_requires_every_new_validation_and_freeze_is_idempotent(finished_study):
    m, _ = finished_study
    record = control.run_dir(m["settings"][-1]) / "validation_checkpoint.json"
    saved = control.read(record)
    record.unlink()
    with pytest.raises(FileNotFoundError):
        control.freeze()
    assert not (control.ROOT / m["freeze_manifest"]).exists()
    atomic_json(record, saved)
    assert control.freeze() == control.freeze()


@pytest.mark.parametrize("filename", ["best_t1.pt", "t1_calibration.json"])
def test_changed_frozen_artifact_is_rejected(finished_study, filename):
    m, _ = finished_study
    control.freeze()
    (control.run_dir(m["settings"][0]) / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        control.freeze()


def test_test_requires_manifest_and_uses_frozen_calibration(finished_study, monkeypatch):
    m, metrics = finished_study
    with pytest.raises(ValueError, match="must be frozen"):
        control.test(m["settings"])
    control.freeze()
    calls = []
    class FakeTrainer:
        def __init__(self, config):
            self.config = config
        def load_checkpoint(self, path):
            assert path.name == "best_t1.pt"
        def evaluate_split(self, split, calibrator, export, protocols):
            assert split == "test" and export
            assert calibrator.__dict__ == Calibrator().__dict__
            calls.append(self.config.train.output_dir)
            return metrics
        def fit_t1_calibration(self):
            pytest.fail("Test must not refit the frozen calibrator")
    monkeypatch.setattr("dvsrc.trainer.Trainer", FakeTrainer)
    control.test(m["settings"])
    control.test(m["settings"])
    assert len(calls) == 7
    control.export()
    target = control.ROOT / m["results_root"]
    assert not list(target.rglob("*.pt"))
    assert not [p for p in target.rglob("*") if p.is_symlink()]
    assert len(control.read(target / "summary.json")["rows"]) == 42
    assert len(list(target.rglob("test_release_manifest.json"))) == 7
    with pytest.raises(FileExistsError):
        control.export()


def test_interrupted_test_checks_frozen_artifact_again(finished_study):
    m, _ = finished_study
    frozen = control.freeze()
    setting = m["settings"][0]
    atomic_json(control.run_dir(setting) / "test/test_access_started.json",
                {"frozen_record": frozen["entries"][0]})
    (control.run_dir(setting) / "t1_calibration.json").write_text("{}")
    with pytest.raises(ValueError, match="calibration changed"):
        control.test([setting])


def test_train_resumes_own_checkpoint_and_appends_logs(finished_study, monkeypatch):
    m, _ = finished_study
    setting = m["settings"][0]
    output = control.run_dir(setting)
    (output / "validation_checkpoint.json").unlink()
    atomic_json(output / "t1_training_status.json", {"running": True, "test_executed": False})
    (output / "last_t1.pt").write_bytes(b"resume fixture")
    (output / "stdout.log").write_bytes(b"previous\n")
    calls = []
    def subprocess_run(command, cwd, stdout, stderr, check):
        calls.append(command)
        assert command[-2:] == ["--resume", str(output / "last_t1.pt")]
        assert "train-t1" in command
        stdout.write(b"continued\n")
        atomic_json(output / "t1_training_status.json", {"running": False, "test_executed": False})
    monkeypatch.setattr(control.subprocess, "run", subprocess_run)
    control.train([setting])
    control.train([setting])
    assert len(calls) == 1
    assert (output / "stdout.log").read_bytes() == b"previous\ncontinued\n"


def test_study_lock_prevents_concurrent_execution(finished_study):
    with control.study_lock():
        with pytest.raises(OSError):
            with control.study_lock():
                pytest.fail("Second execution acquired the study lock")
    with control.study_lock():
        pass
