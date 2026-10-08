import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dvsrc.data import (
    DynamicRGBRenderConfig,
    DynamicRGBRenderer,
    RenderConfig,
    SignaturePreprocessor,
    SignatureStore,
    T2SingleSplitBuilder,
    audit_benchmark,
    audit_t1_single_benchmark,
    audit_t2_single_benchmark,
    build_raw_sequence,
    uniform_subsample,
)
from dvsrc.utils import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_each_writer_is_test_once():
    rotations = json.loads(
        (ROOT / ".Trash/workspace_reorg_20260907_131226/benchmark/datasets/legacy_main/folds/fold_rotation.json").read_text()
    )
    test_writers = [writer for rotation in rotations.values() for writer in rotation["test"]]
    assert len(test_writers) == 79
    assert len(set(test_writers)) == 79
    for rotation in rotations.values():
        train, val, test = map(set, (rotation["train"], rotation["val"], rotation["test"]))
        assert not (train & val or train & test or val & test)


def test_primary_t2_has_all_three_episode_types():
    rows = read_jsonl(
        ROOT / ".Trash/workspace_reorg_20260907_131226/benchmark/datasets/legacy_main/episodes/fold_0/test_t2.jsonl"
    )
    assert {row["episode_type"] for row in rows} == {"source_present", "source_absent", "rf_no_source"}
    assert {row["candidate_count"] for row in rows} == {8}
    writers = {row["target_writer_id"] for row in rows}
    for writer in writers:
        positions = {row["target_index"] for row in rows
                     if row["target_writer_id"] == writer and row["episode_type"] == "source_present"}
        assert positions == set(range(8))


def test_primary_t2_is_binary_balanced_without_reducing_unknown_episodes():
    for split in ("train", "val", "test"):
        rows = read_jsonl(
            ROOT / f".Trash/workspace_reorg_20260907_131226/benchmark/datasets/legacy_main/episodes/fold_0/{split}_t2.jsonl"
        )
        counts = Counter(row["episode_type"] for row in rows)
        assert counts["source_present"] == counts["source_absent"] + counts["rf_no_source"]
        assert counts["source_absent"] * 3 == counts["rf_no_source"] * 2


def test_l20_scope_is_explicitly_restricted():
    rows = read_jsonl(
        ROOT
        / ".Trash/workspace_reorg_20260907_131226/benchmark/datasets/legacy_main/episodes/fold_0/test_t2_l20_restricted.jsonl"
    )
    assert {row["episode_type"] for row in rows} == {"source_present", "rf_no_source"}
    assert {row["candidate_count"] for row in rows} == {20}


def test_t1_forgery_rows_retain_forger_for_lofo_audit():
    rows = read_jsonl(
        ROOT / ".Trash/workspace_reorg_20260907_131226/benchmark/datasets/legacy_main/episodes/fold_0/train_t1_1v1.jsonl"
    )
    forged = [row for row in rows if row["attack_type"] in {"RF", "SF"}]
    assert forged
    assert {row["forger_id"] for row in forged} == {"F01", "F02", "F03", "F04"}


def test_audit_passes():
    report = audit_benchmark(
        ROOT / "datasets/signatures",
        ROOT / ".Trash/workspace_reorg_20260907_131226/benchmark/datasets/legacy_main",
        full_hash=False,
    )
    assert report["ok"]


def test_t2_single_split_counts_and_audit_pass():
    root = ROOT / "datasets/protocols/t2"
    expected = {
        "train": (10560, {"source_present": 5280, "source_absent": 2640, "rf_no_source": 2640}),
        "val": (2304, {"source_present": 1152, "source_absent": 576, "rf_no_source": 576}),
        "test": (2304, {"source_present": 1152, "source_absent": 576, "rf_no_source": 576}),
    }
    for split, (total, distribution) in expected.items():
        rows = read_jsonl(root / f"episodes/fold_0/{split}_t2.jsonl")
        assert len(rows) == total
        assert dict(Counter(row["episode_type"] for row in rows)) == distribution
        assert len({(row["query_id"], row["candidate_pool_hash"]) for row in rows}) == total
    assert audit_t2_single_benchmark(
        ROOT / "datasets/signatures", root,
    )["ok"]


def test_t1_single_split_metadata_matches_file_roles():
    root = ROOT / "datasets/protocols/t1"
    for split in ("train", "val", "test"):
        for reference_count in (1, 5):
            protocol = f"t1_{reference_count}v1"
            rows = read_jsonl(
                root / f"episodes/fold_0/{split}_t1_{reference_count}v1.jsonl"
            )
            assert {row["split"] for row in rows} == {split}
            assert {row["protocol"] for row in rows} == {protocol}
            assert all(
                row["episode_id"].startswith(
                    f"S0-{split.upper()}-T1-{reference_count}V1-"
                )
                for row in rows
            )
            assert all(
                row["query_group_id"]
                == f"{split}:{protocol}:{row['target_writer_id']}:{row['query_id']}"
                for row in rows
            )
    assert audit_t1_single_benchmark(
        ROOT / "datasets/signatures", root,
    )["ok"]


def test_t2_single_split_builder_accepts_l5_and_rejects_invalid_candidate_counts(tmp_path):
    store = SimpleNamespace(samples={})
    builder = T2SingleSplitBuilder(store, tmp_path / "l5", candidate_count=5)

    assert builder.candidate_count == 5
    with pytest.raises(ValueError, match="between 2 and 19"):
        T2SingleSplitBuilder(store, tmp_path / "invalid", candidate_count=20)


def test_t2_l5_split_keeps_episode_ratio_and_near_balances_source_positions(tmp_path):
    samples = {}
    for index in range(1, 21):
        sample_id = f"001-NW-{index}"
        samples[sample_id] = {
            "sample_id": sample_id,
            "writer_id": "001",
            "state": "NW",
            "collection_batch": "batch_1",
        }
    for index in range(1, 9):
        sample_id = f"001-SF-{index}"
        samples[sample_id] = {
            "sample_id": sample_id,
            "writer_id": "001",
            "state": "SF",
            "reference_nw_index": index,
            "forger_id": "F01",
            "collection_batch": "batch_1",
        }
    for index in range(1, 13):
        sample_id = f"001-RF-{index}"
        samples[sample_id] = {
            "sample_id": sample_id,
            "writer_id": "001",
            "state": "RF",
            "forger_id": "F01",
            "collection_batch": "batch_1",
        }

    builder = T2SingleSplitBuilder(
        SimpleNamespace(samples=samples), tmp_path / "l5", candidate_count=5,
    )
    rows = builder._build_split("test", ["001"])

    assert len(rows) == 192
    assert Counter(row["episode_type"] for row in rows) == {
        "source_present": 96,
        "source_absent": 48,
        "rf_no_source": 48,
    }
    assert {row["candidate_count"] for row in rows} == {5}
    assert {len(row["candidate_ids"]) for row in rows} == {5}
    positions = Counter(
        row["target_index"] for row in rows if row["episode_type"] == "source_present"
    )
    assert set(positions) == set(range(5))
    assert sorted(positions.values()) == [19, 19, 19, 19, 20]


def test_signature_store_reads_utf8_header_on_windows(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\u65f6\u95f4(ms),X\u5750\u6807(mm),Y\u5750\u6807(mm),\u538b\u529b,\u901f\u5ea6,\u65b9\u5411\u89d2,\u843d\u7b14\u72b6\u6001\n"
        "0,1,2,3,4,5,1\n",
        encoding="utf-8",
    )
    manifest = {"samples": [{"sample_id": "sample", "csv_path": "sample.csv"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    array = SignatureStore(tmp_path).load_csv("sample")

    assert array.shape == (1, 7)
    assert array.tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0]]


def test_prepared_signature_memory_cache_is_lru_bounded(tmp_path):
    class Store:
        def __init__(self):
            base = np.arange(4, dtype=np.float32)
            self.rows = {
                sample_id: np.stack(
                    [base, base + offset, base, base + 1, base + 2, base, np.ones(4)], axis=1,
                )
                for sample_id, offset in (("a", 0), ("b", 1))
            }

        def get(self, sample_id):
            return {"sha256": sample_id}

        def load_csv(self, sample_id):
            return self.rows[sample_id]

    preprocessor = SignaturePreprocessor(
        Store(), {"mean": [0] * 10, "std": [1] * 10, "clip_sigma": 8}, tmp_path,
        RenderConfig(width=32, height=16, margin=2, supersample=1),
        cache_images=False, memory_cache=True, memory_cache_items=1,
    )

    first = preprocessor.prepare("a")
    assert preprocessor.prepare("a") is first
    preprocessor.prepare("b")

    assert list(preprocessor._prepared_cache) == ["b"]


def test_cache_publish_reuses_identical_file_from_another_worker(tmp_path, monkeypatch):
    target = tmp_path / "cache.png"
    temporary = tmp_path / "cache.worker.tmp.png"
    payload = b"deterministic-rendered-png"
    target.write_bytes(payload)
    temporary.write_bytes(payload)

    def deny_replace(self, destination):
        raise PermissionError("simulated Windows cache reader")

    monkeypatch.setattr(type(temporary), "replace", deny_replace)
    digest = SignaturePreprocessor._publish_cached_png(temporary, target)

    assert digest == hashlib.sha256(payload).hexdigest()
    assert target.read_bytes() == payload
    assert not temporary.exists()


def test_dynamic_rgb_renderer_encodes_pressure_speed_and_time():
    raw = np.asarray([
        [0, 0, 0, 0.10, 0, 0, 1],
        [10, 1, 1, 0.30, 200, 45, 1],
        [20, 2, 2, 0.70, 500, 45, 1],
        [30, 3, 3, 1.00, 800, 45, 1],
    ], dtype=np.float32)
    renderer = DynamicRGBRenderer(DynamicRGBRenderConfig(
        width=32, height=32, margin=2, line_width=1, supersample=1,
    ))

    image, metadata = renderer.render(raw)
    pixels, _ = renderer.transform(raw)
    array = np.asarray(image)
    early = array[round(pixels[0, 1]), round(pixels[0, 0])]
    late = array[round(pixels[-1, 1]), round(pixels[-1, 0])]

    assert image.mode == "RGB"
    assert metadata["flip_y"] is True
    assert pixels[0, 1] > pixels[-1, 1]
    assert np.all(late > early)
    assert np.all(array[0, 0] == 0)


def test_raw_sequence_excludes_dataset_speed_and_direction():
    raw = np.asarray([
        [0, 0, 0, 0.25, 10, -90, 1],
        [10, 1, 2, 0.75, 20, 90, 1],
    ], dtype=np.float32)
    changed = raw.copy()
    changed[:, 4] = [999, 1234]
    changed[:, 5] = [13, 27]
    first = build_raw_sequence(raw)
    second = build_raw_sequence(changed)

    assert first.shape == (2, 5)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[:, 0], [0, 10 / 60000])
    np.testing.assert_allclose(first[:, 1:3], [[0, 0], [0.01, 0.02]])


def test_uniform_subsample_preserves_both_ends():
    raw = np.arange(70, dtype=np.float32).reshape(10, 7)
    sampled = uniform_subsample(raw, 4)

    assert len(sampled) == 4
    np.testing.assert_array_equal(sampled[0], raw[0])
    np.testing.assert_array_equal(sampled[-1], raw[-1])
