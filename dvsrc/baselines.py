from __future__ import annotations

from typing import Any

import numpy as np

from .data import SignatureStore


def _trajectory(raw: np.ndarray, max_points: int = 512) -> np.ndarray:
    points = raw[raw[:, 6] > 0.5][:, 1:5]
    if not len(points):
        points = raw[:, 1:5]
    points = points.copy()
    points[:, :2] -= points[:, :2].mean(0, keepdims=True)
    scale = np.linalg.norm(points[:, :2], axis=1).max()
    points[:, :2] /= max(scale, 1e-6)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]
    return points


def _dtw_path_cost(costs: np.ndarray, window: int) -> float:
    n, m = costs.shape
    accumulated = np.full((n + 1, m + 1), np.inf)
    accumulated[0, 0] = 0.0
    for diagonal in range(2, n + m + 1):
        rows = np.arange(max(1, diagonal - m), min(n, diagonal - 1) + 1)
        columns = diagonal - rows
        valid = np.abs(rows - columns) <= window
        rows, columns = rows[valid], columns[valid]
        if not len(rows):
            continue
        previous = np.minimum(
            np.minimum(accumulated[rows, columns - 1], accumulated[rows - 1, columns]),
            accumulated[rows - 1, columns - 1],
        )
        accumulated[rows, columns] = costs[rows - 1, columns - 1] + previous
    return float(accumulated[n, m] / max(n + m, 1))


def dtw_distance(a: np.ndarray, b: np.ndarray, window_fraction: float = 0.2) -> float:
    costs = np.sqrt(np.square(a[:, None, :] - b[None, :, :]).sum(axis=-1))
    n, m = costs.shape
    window = max(abs(n - m), int(max(n, m) * window_fraction))
    return float(_dtw_path_cost(costs, window))


class DTWBaseline:
    def __init__(self, store: SignatureStore):
        self.store = store
        self.cache: dict[str, np.ndarray] = {}
        self.pair_cache: dict[tuple[str, str], float] = {}

    def representation(self, sample_id: str) -> np.ndarray:
        if sample_id not in self.cache:
            self.cache[sample_id] = _trajectory(self.store.load_csv(sample_id))
        return self.cache[sample_id]

    def score_pair(self, first: str, second: str) -> float:
        key = tuple(sorted((first, second)))
        if key not in self.pair_cache:
            self.pair_cache[key] = -dtw_distance(self.representation(first), self.representation(second))
        return self.pair_cache[key]

    def predict_t1(self, episode: dict[str, Any]) -> float:
        scores = [self.score_pair(reference, episode["query_id"]) for reference in episode["reference_ids"]]
        return float(np.mean(scores))

    def scores_t2(self, episode: dict[str, Any]) -> np.ndarray:
        return np.asarray([self.score_pair(candidate, episode["query_id"])
                           for candidate in episode["candidate_ids"]])

    def predict_t2(self, episode: dict[str, Any], unknown_threshold: float) -> list[float]:
        scores = self.scores_t2(episode)
        shifted = np.exp(scores - scores.max())
        rank = shifted / shifted.sum()
        exist = 1 / (1 + np.exp(-(scores.max() - unknown_threshold)))
        return (np.r_[exist * rank, 1 - exist]).tolist()


def global_features(raw: np.ndarray) -> np.ndarray:
    time, x, y, pressure, speed, direction, pen = raw.T
    active = pen > 0.5
    points = np.stack([x, y], 1)
    delta = np.diff(points, axis=0)
    step = np.linalg.norm(delta, axis=1)
    active_step = step[(active[1:] & active[:-1])]
    span = np.ptp(points[active] if active.any() else points, axis=0)
    displacement = np.linalg.norm(points[-1] - points[0])
    angle = np.deg2rad(direction)
    transitions = np.abs(np.diff(pen)).sum()
    histogram, _ = np.histogram(np.mod(direction[active], 360), bins=8, range=(0, 360), density=True)
    values = [
        np.log1p(max(time[-1] - time[0], 0)), np.log1p(len(raw)), span[0], span[1],
        span[0] / max(span[1], 1e-4), active_step.sum() if len(active_step) else 0,
        displacement, displacement / max(active_step.sum(), 1e-4) if len(active_step) else 0,
        active.mean(), np.log1p(transitions), pressure.mean(), pressure.std(),
        *np.quantile(pressure, [0.1, 0.5, 0.9]), np.log1p(np.maximum(speed, 0)).mean(),
        np.log1p(np.maximum(speed, 0)).std(), *np.quantile(np.log1p(np.maximum(speed, 0)), [0.1, 0.5, 0.9]),
        np.sin(angle).mean(), np.cos(angle).mean(), *histogram.tolist(),
    ]
    return np.asarray(values, dtype=np.float32)


class GlobalFeatureBaseline:
    def __init__(self, store: SignatureStore):
        self.store = store
        self.cache = {sample_id: global_features(store.load_csv(sample_id)) for sample_id in store.samples}
        self.mean = np.zeros_like(next(iter(self.cache.values())))
        self.std = np.ones_like(self.mean)

    def fit_normalization(self, writer_ids: set[str]) -> None:
        matrix = np.stack([self.cache[sample_id] for sample_id, sample in self.store.samples.items()
                           if sample["writer_id"] in writer_ids])
        self.mean, self.std = matrix.mean(0), matrix.std(0).clip(min=1e-5)

    def representation(self, sample_id: str) -> np.ndarray:
        return (self.cache[sample_id] - self.mean) / self.std

    def similarity(self, first: str, second: str) -> float:
        difference = self.representation(first) - self.representation(second)
        return float(-np.sqrt(np.mean(difference * difference)))

    def score_t1(self, episode: dict[str, Any]) -> float:
        return float(np.mean([self.similarity(reference, episode["query_id"])
                              for reference in episode["reference_ids"]]))

    def scores_t2(self, episode: dict[str, Any]) -> np.ndarray:
        return np.asarray([self.similarity(candidate, episode["query_id"])
                           for candidate in episode["candidate_ids"]])
