import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "simple_supervisor", Path(__file__).resolve().parents[1] / "ablation/t1/simple_v1/supervise.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def test_phase_requires_all_validation_before_test(tmp_path):
    assert watch.phase(tmp_path) == "train"
    for item in watch.IDS:
        run = tmp_path / "runs" / item / "seed_42"
        run.mkdir(parents=True)
        (run / "validation_checkpoint.json").write_text("{}")
    assert watch.phase(tmp_path) == "freeze"
    (tmp_path / "test_checkpoint_manifest.json").write_text("{}")
    assert watch.phase(tmp_path) == "test"
    for item in watch.IDS:
        output = tmp_path / "runs" / item / "seed_42/test"
        output.mkdir()
        (output / "test_release_manifest.json").write_text("{}")
    assert watch.phase(tmp_path) == "export"
    output = tmp_path / "results_seed42"
    output.mkdir()
    (output / "summary.json").write_text(json.dumps({"status": "complete",
        "validation_completed": 7, "test_completed": 7, "weights_included": False}))
    assert watch.phase(tmp_path) == "complete"
    (output / "summary.json").write_text("{}")
    with pytest.raises(ValueError):
        watch.phase(tmp_path)


def test_only_workspace_exact_workers():
    args = ["python", "-m", "dvsrc.cli", "train-t1"]
    assert watch.matches_worker(args, str(watch.ROOT))
    assert not watch.matches_worker(args, "/another/project")
    assert not watch.matches_worker(["bash", "-c", "dvsrc.cli train-t1"], str(watch.ROOT))
    assert not watch.matches_worker(["python", "ablation/t1/simple_v1/run.py", "status"], str(watch.ROOT))


@pytest.mark.parametrize("phase,age,gpu,expected", [
    ("train", 1900, 0, True), ("test", 1900, 4, True),
    ("train", 1900, 50, False), ("train", 1700, 0, False),
    ("train", 1900, None, False), ("export", 1900, 0, False)])
def test_stall_requires_stale_log_and_idle_gpu(phase, age, gpu, expected):
    assert watch.stalled({"phase": phase, "log_age_seconds": age, "gpu_utilization": gpu}) == expected
