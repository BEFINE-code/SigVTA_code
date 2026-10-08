import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dvsrc.config import DataConfig, ExperimentConfig
from dvsrc.data import (
    FEATURE_NAMES,
    RenderConfig,
    SignaturePreprocessor,
    build_point_features,
    channel_indices,
    resample_arc_length,
    zero_feature_channels,
)


ROOT = Path(__file__).resolve().parents[1]


def test_channel_zeroing_keeps_width_and_nulls_named_dims():
    features = np.arange(20, dtype=np.float32).reshape(2, 10)
    indices = channel_indices(("t_rel", "delta_t", "log1p_speed"))
    out = zero_feature_channels(features, indices)
    assert out.shape == (2, 10)
    assert np.all(out[:, indices] == 0)
    kept = [i for i in range(10) if i not in indices]
    np.testing.assert_array_equal(out[:, kept], features[:, kept])
    np.testing.assert_array_equal(features[:, 0], [0, 10])


def test_arc_length_resample_imposes_constant_dt_and_duration():
    raw = np.asarray([
        [0, 0, 0, 0.2, 0, 0, 1],
        [10, 3, 4, 0.8, 500, 53, 1],
        [1000, 6, 8, 0.4, 10, 53, 1],
    ], dtype=np.float32)
    aligned = resample_arc_length(raw, n_points=5, duration_ms=4000)
    assert aligned.shape == (5, 7)
    np.testing.assert_allclose(aligned[-1, 0] - aligned[0, 0], 4000)
    dt = np.diff(aligned[:, 0])
    np.testing.assert_allclose(dt, dt[0])
    np.testing.assert_allclose(aligned[0, 1:3], [0, 0])
    np.testing.assert_allclose(aligned[-1, 1:3], [6, 8])
    assert aligned[-1, 0] / (len(aligned) - 1) == pytest.approx(1000)


def test_yaml_zero_channels_and_time_align_load():
    config = ExperimentConfig.from_yaml(ROOT / "ablation/t1_sf_hack/configs/H-T2.yaml")
    assert config.data.zero_channels == ("t_rel", "delta_t", "log1p_speed")
    assert config.data.time_align is None
    aligned = ExperimentConfig.from_yaml(ROOT / "ablation/t1_sf_hack/configs/H-N2.yaml")
    assert aligned.data.time_align == "arc_length"
    assert aligned.data.time_align_constant_pressure is True
    assert aligned.data.image_cache_root == "datasets/cache/t1_sf_hack/aligned_png"
    with pytest.raises(ValueError, match="time_align"):
        DataConfig(time_align="warp")


def test_preprocessor_zeros_time_channels_before_clip(tmp_path):
    class Store:
        def get(self, sample_id):
            return {"sha256": sample_id}

        def load_csv(self, sample_id):
            t = np.arange(8, dtype=np.float32) * 10
            return np.stack([t, t, t, np.ones(8), np.ones(8) * 100, np.zeros(8), np.ones(8)], axis=1)

    stats = {"mean": [0] * 10, "std": [1] * 10, "clip_sigma": 8}
    preprocessor = SignaturePreprocessor(
        Store(), stats, tmp_path, RenderConfig(width=32, height=16, margin=2, supersample=1),
        cache_images=False, memory_cache=False, zero_channels=("t_rel", "delta_t"),
    )
    prepared = preprocessor.prepare("a")
    features = prepared.sequence.numpy()
    assert features.shape[1] == 10
    assert np.all(features[:, channel_indices(("t_rel", "delta_t"))] == 0)
    raw = Store().load_csv("a")
    original = np.clip((build_point_features(raw) - 0) / 1, -8, 8)
    assert not np.allclose(original[:, 0], 0)


def test_time_align_changes_cache_key_and_point_count(tmp_path):
    class Store:
        def __init__(self):
            self.samples = {
                "a": {"sample_id": "a", "sha256": "a", "writer_id": "001",
                      "label": "genuine", "duration_ms": 8000},
                "b": {"sample_id": "b", "sha256": "b", "writer_id": "001",
                      "label": "genuine", "duration_ms": 2000},
            }

        def get(self, sample_id):
            return self.samples[sample_id]

        def load_csv(self, sample_id):
            n = 6 if sample_id == "a" else 4
            t = np.arange(n, dtype=np.float32) * (80 if sample_id == "a" else 10)
            x = np.linspace(0, 10, n, dtype=np.float32)
            return np.stack([t, x, x, np.ones(n) * 0.5, np.ones(n), np.zeros(n), np.ones(n)], axis=1)

    stats = {"mean": [0] * 10, "std": [1] * 10, "clip_sigma": 8}
    render = RenderConfig(width=32, height=16, margin=2, supersample=1)
    plain = SignaturePreprocessor(Store(), stats, tmp_path / "plain", render, cache_images=False, memory_cache=False)
    aligned = SignaturePreprocessor(
        Store(), stats, tmp_path / "aligned", render, cache_images=False, memory_cache=False,
        time_align="arc_length", time_align_points=8, time_align_duration_ms=4000,
    )
    assert plain._cache_paths("a")[2] != aligned._cache_paths("a")[2]
    prepared = aligned.prepare("a")
    assert len(prepared.sequence) == 8
    raw = aligned._raw_for_sample("a")
    assert raw[-1, 0] == pytest.approx(4000)


def _load_sf_hack_run():
    path = ROOT / "ablation/t1_sf_hack/run.py"
    spec = importlib.util.spec_from_file_location("t1_sf_hack_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sf_hack_matrix_is_isolated():
    control = _load_sf_hack_run()
    control.validate_configs()
    matrix = control.matrix()
    assert [s["id"] for s in matrix["settings"]] == ["H-T1", "H-T2", "H-T3", "H-N1", "H-N2"]
    assert matrix["runs_root"].startswith("ablation/t1_sf_hack/")
    assert "ablation/t1/runs" not in matrix["runs_root"]
    config = control.config_for(matrix["settings"][0])
    assert config.data.zero_channels == ("t_rel", "delta_t")
    assert config.train.stage_a_batch_t1_1v1 == 8
    ht2 = control.config_for(matrix["settings"][1])
    assert ht2.train.stage_a_batch_t1_1v1 == 32
    assert ht2.train.stage_a_grad_accumulation == 2
    assert FEATURE_NAMES[0] == "t_rel"
