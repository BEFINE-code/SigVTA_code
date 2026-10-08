from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from dtaidistance import dtw_ndim

from .data import SignatureStore


def trajectory(raw: np.ndarray, max_points: int = 512) -> np.ndarray:
    points = raw[raw[:, 6] > 0.5][:, 1:5]
    if not len(points):
        points = raw[:, 1:5]
    points = np.asarray(points, dtype=np.float64).copy()
    points[:, :2] -= points[:, :2].mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points[:, :2], axis=1).max()
    points[:, :2] /= max(float(scale), 1e-6)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]
    return np.ascontiguousarray(points)


def dtw_distance(first: np.ndarray, second: np.ndarray, window_fraction: float = 0.2) -> float:
    window = max(abs(len(first) - len(second)), int(max(len(first), len(second)) * window_fraction))
    distance = dtw_ndim.distance_fast(
        first,
        second,
        window=max(window, 1),
        inner_dist="euclidean",
    )
    return float(distance / max(len(first) + len(second), 1))


class DTWScorer:
    def __init__(self, store: SignatureStore, cache_path: str | Path | None = None) -> None:
        self.store = store
        self.cache_path = Path(cache_path) if cache_path else None
        self.trajectories: dict[str, np.ndarray] = {}
        self.scores: dict[tuple[str, str], float] = {}
        if self.cache_path and self.cache_path.is_file():
            cached = np.load(self.cache_path, allow_pickle=False)
            pairs = cached["pairs"]
            values = cached["scores"]
            self.scores = {
                (str(pair[0]), str(pair[1])): float(value)
                for pair, value in zip(pairs, values)
            }

    def representation(self, sample_id: str) -> np.ndarray:
        if sample_id not in self.trajectories:
            self.trajectories[sample_id] = trajectory(self.store.load(sample_id))
        return self.trajectories[sample_id]

    def _save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        pairs = sorted(self.scores)
        np.savez_compressed(
            self.cache_path,
            pairs=np.asarray(pairs, dtype="U64"),
            scores=np.asarray([self.scores[pair] for pair in pairs], dtype=np.float64),
        )

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        keys = [tuple(sorted(pair)) for pair in pairs]
        missing = sorted(set(keys) - self.scores.keys())
        if missing:
            started = time.perf_counter()
            for index, pair in enumerate(missing, start=1):
                self.scores[pair] = -dtw_distance(
                    self.representation(pair[0]), self.representation(pair[1]),
                )
                if index % 2000 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"DTW {index}/{len(missing)} new pairs ({elapsed:.1f}s)", flush=True)
            self._save()
        return np.asarray([self.scores[key] for key in keys], dtype=np.float64)
