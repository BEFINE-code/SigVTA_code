import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_diagnose():
    path = ROOT / "ablation/t1_sf_hack/diagnose_duration.py"
    spec = importlib.util.spec_from_file_location("t1_sf_hack_duration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_diagnose = _load_diagnose()
attach_duration = _diagnose.attach_duration
duration_matched_subset = _diagnose.duration_matched_subset
far_with_ci = _diagnose.far_with_ci
wilson_interval = _diagnose.wilson_interval


def test_wilson_interval_for_zero_events_is_not_zero_risk():
    low, high = wilson_interval(0, 384)
    assert low == 0
    assert 0.005 < high < 0.02


def test_duration_matching_keeps_only_sf_inside_writer_iqr():
    samples = {
        "G1": {"sample_id": "G1", "writer_id": "001", "label": "genuine", "state": "NW",
               "duration_ms": 4000, "row_count": 400},
        "G2": {"sample_id": "G2", "writer_id": "001", "label": "genuine", "state": "NW",
               "duration_ms": 6000, "row_count": 600},
        "SF_slow": {"sample_id": "SF_slow", "writer_id": "001", "label": "forged", "state": "SF",
                    "duration_ms": 20000, "row_count": 2000},
        "SF_in": {"sample_id": "SF_in", "writer_id": "001", "label": "forged", "state": "SF",
                  "duration_ms": 5000, "row_count": 500},
    }
    episodes = [
        {"label": 1, "attack_type": "genuine", "query_id": "G1", "reference_ids": ["G2"],
         "target_writer_id": "001"},
        {"label": 0, "attack_type": "SF", "query_id": "SF_slow", "reference_ids": ["G1"],
         "target_writer_id": "001"},
        {"label": 0, "attack_type": "SF", "query_id": "SF_in", "reference_ids": ["G1"],
         "target_writer_id": "001"},
    ]
    rows = attach_duration(episodes, samples)
    matched = duration_matched_subset(rows)
    assert {row["query_id"] for row in matched if row["attack_type"] == "SF"} == {"SF_in"}
    scored = [{**row, "score": 0.0} for row in matched if row["attack_type"] == "SF"]
    assert far_with_ci(scored, 0.5)["n"] == 1
