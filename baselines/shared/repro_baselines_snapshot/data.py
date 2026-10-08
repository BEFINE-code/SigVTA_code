from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_DIGEST = "ba64c09c53119be31f357c089987f706eb3266e4348d8398ff5d88e83dc0c48d"
T2_CANDIDATE_COUNT = 5
EXPECTED_COUNTS = {
    "train_t1_1v1.jsonl": 17600,
    "train_t1_5v1.jsonl": 17600,
    "val_t1_1v1.jsonl": 1920,
    "val_t1_5v1.jsonl": 1920,
    "test_t1_1v1.jsonl": 1920,
    "test_t1_5v1.jsonl": 1920,
    "train_t2.jsonl": 10560,
    "val_t2.jsonl": 2304,
    "test_t2.jsonl": 2304,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class BenchmarkRepository:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        final_root = self.root / "experiment_final"
        self.t1_benchmark = final_root / "benchmarks" / "benchmark_t1_final_v1"
        self.t2_benchmark = final_root / "benchmarks" / "benchmark_t2_l5_final_v1"
        self.benchmark = self.t1_benchmark
        self.dataset = final_root / "data" / "sig_database"
        self.t1_episodes_root = self.t1_benchmark / "episodes" / "fold_0"
        self.t2_episodes_root = self.t2_benchmark / "episodes" / "fold_0"
        self.split = read_json(self.t1_benchmark / "folds" / "single_split.json")

    def verify(self) -> dict[str, Any]:
        t1_audit = read_json(self.t1_benchmark / "audit_report.json")
        t2_audit = read_json(self.t2_benchmark / "t2_audit_report.json")
        if not t1_audit.get("ok") or not t2_audit.get("ok"):
            raise RuntimeError("Frozen benchmark audit is not clean")
        digests = {
            self.split.get("split_digest"),
            t1_audit.get("split_digest"),
            t2_audit.get("split_digest"),
        }
        if digests != {SPLIT_DIGEST}:
            raise RuntimeError(f"Unexpected split digests: {sorted(str(value) for value in digests)}")
        observed = {**t1_audit["episodes_checked"], **t2_audit["episodes_checked"]}
        if observed != EXPECTED_COUNTS:
            raise RuntimeError(f"Unexpected episode counts: {observed}")
        writers = {name: set(values) for name, values in self.split["splits"].items()}
        if writers["train"] & writers["val"] or writers["train"] & writers["test"] or writers["val"] & writers["test"]:
            raise RuntimeError("Writer split is not disjoint")
        if t2_audit.get("candidate_count") != T2_CANDIDATE_COUNT:
            raise RuntimeError(f"Unexpected T2 candidate count: {t2_audit.get('candidate_count')}")
        return {
            "split_digest": SPLIT_DIGEST,
            "episode_counts": observed,
            "writer_counts": {name: len(values) for name, values in writers.items()},
        }

    def writers(self, split: str) -> set[str]:
        return set(self.split["splits"][split])

    def episodes(self, task: str, split: str, *, final: bool = False) -> list[dict[str, Any]]:
        if split == "test" and not final:
            raise PermissionError("Test episodes require final=True")
        if task not in {"t1_1v1", "t1_5v1", "t2"}:
            raise ValueError(f"Unknown task: {task}")
        suffix = task if task.startswith("t1_") else "t2"
        episodes_root = self.t1_episodes_root if task.startswith("t1_") else self.t2_episodes_root
        rows = read_jsonl(episodes_root / f"{split}_{suffix}.jsonl")
        if task == "t2" and any(len(row["candidate_ids"]) != T2_CANDIDATE_COUNT for row in rows):
            raise RuntimeError(f"{split}/t2 does not use {T2_CANDIDATE_COUNT} candidates")
        allowed = self.writers(split)
        if {row["target_writer_id"] for row in rows} - allowed:
            raise RuntimeError(f"{split}/{task} contains writers outside the frozen split")
        return rows


class SignatureStore:
    def __init__(self, dataset_root: str | Path) -> None:
        self.root = Path(dataset_root).resolve()
        manifest = read_json(self.root / "manifest.json")
        self.samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
        if len(self.samples) != manifest["counts"]["samples"]:
            raise RuntimeError("Dataset manifest contains duplicate or missing sample IDs")

    def load(self, sample_id: str) -> np.ndarray:
        sample = self.samples[sample_id]
        values = np.genfromtxt(
            self.root / sample["csv_path"], delimiter=",", skip_header=1,
            dtype=np.float32, encoding="utf-8",
        )
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != 7 or not np.isfinite(values).all():
            raise ValueError(f"Invalid CSV for {sample_id}: {values.shape}")
        return values
