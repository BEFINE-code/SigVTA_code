from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .data import SignatureStore


def global_features(raw: np.ndarray) -> np.ndarray:
    time, x, y, pressure, speed, direction, pen = raw.T
    active = pen > 0.5
    points = np.stack([x, y], axis=1)
    active_points = points[active] if active.any() else points
    delta = np.diff(points, axis=0)
    step = np.linalg.norm(delta, axis=1)
    active_step = step[active[1:] & active[:-1]]
    span = np.ptp(active_points, axis=0)
    displacement = np.linalg.norm(points[-1] - points[0])
    path_length = active_step.sum() if len(active_step) else 0.0
    angle = np.deg2rad(direction)
    transitions = np.abs(np.diff(pen)).sum()
    histogram, _ = np.histogram(
        np.mod(direction[active], 360) if active.any() else np.mod(direction, 360),
        bins=8, range=(0, 360), density=True,
    )
    log_speed = np.log1p(np.maximum(speed, 0))
    values = [
        np.log1p(max(time[-1] - time[0], 0)), np.log1p(len(raw)), span[0], span[1],
        span[0] / max(span[1], 1e-4), path_length, displacement,
        displacement / max(path_length, 1e-4), active.mean(), np.log1p(transitions),
        pressure.mean(), pressure.std(), *np.quantile(pressure, [0.1, 0.5, 0.9]),
        log_speed.mean(), log_speed.std(), *np.quantile(log_speed, [0.1, 0.5, 0.9]),
        np.sin(angle).mean(), np.cos(angle).mean(), *histogram.tolist(),
    ]
    return np.asarray(values, dtype=np.float32)


class FeatureStore:
    def __init__(self, store: SignatureStore, cache_path: str | Path | None = None) -> None:
        self.store = store
        self.cache_path = Path(cache_path) if cache_path else None
        self.ids: list[str] = []
        self.matrix = np.empty((0, 0), dtype=np.float32)
        self.index: dict[str, int] = {}
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def build(self) -> None:
        if self.cache_path and self.cache_path.is_file():
            cached = np.load(self.cache_path, allow_pickle=False)
            self.ids = [str(value) for value in cached["ids"]]
            self.matrix = cached["features"].astype(np.float32)
        else:
            self.ids = sorted(self.store.samples)
            self.matrix = np.stack([global_features(self.store.load(sample_id)) for sample_id in self.ids])
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    self.cache_path,
                    ids=np.asarray(self.ids, dtype="U64"),
                    features=self.matrix,
                )
        self.index = {sample_id: index for index, sample_id in enumerate(self.ids)}

    def fit_normalization(self, writer_ids: set[str]) -> None:
        indices = [
            self.index[sample_id] for sample_id, sample in self.store.samples.items()
            if sample["writer_id"] in writer_ids
        ]
        train = self.matrix[indices]
        self.mean = train.mean(axis=0)
        self.std = train.std(axis=0).clip(min=1e-5)

    def vectors(self, sample_ids: Iterable[str]) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Feature normalization must be fit on Train writers first")
        indices = [self.index[sample_id] for sample_id in sample_ids]
        return (self.matrix[indices] - self.mean) / self.std

    def pair_matrix(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        first = self.vectors(pair[0] for pair in pairs)
        second = self.vectors(pair[1] for pair in pairs)
        difference = first - second
        l1 = np.mean(np.abs(difference), axis=1, keepdims=True)
        l2 = np.sqrt(np.mean(difference * difference, axis=1, keepdims=True))
        cosine = np.sum(first * second, axis=1, keepdims=True) / (
            np.linalg.norm(first, axis=1, keepdims=True)
            * np.linalg.norm(second, axis=1, keepdims=True)
        ).clip(min=1e-6)
        return np.concatenate(
            [np.abs(difference), difference * difference, first * second, l1, l2, cosine],
            axis=1,
        ).astype(np.float32)
