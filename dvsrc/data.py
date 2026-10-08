from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .config import DataConfig
from .utils import atomic_json, read_jsonl, sha256_file, stable_hash, write_jsonl


CSV_COLUMNS = ("time", "x", "y", "pressure", "speed", "direction", "pen")
FEATURE_NAMES = (
    "t_rel", "delta_t", "x_centered", "y_centered", "pressure",
    "log1p_speed", "sin_direction", "cos_direction", "pen_state", "pen_transition",
)
RAW_SEQUENCE_NAMES = (
    "elapsed_time_60s", "x_from_origin_100mm", "y_from_origin_100mm", "pressure", "pen_state",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
RAW_SEQUENCE_INDEX = {name: index for index, name in enumerate(RAW_SEQUENCE_NAMES)}
TIME_SCALE_SECONDS = 60.0
POSITION_SCALE_MM = 100.0


@dataclass(frozen=True)
class RenderConfig:
    width: int = 512
    height: int = 256
    margin: int = 12
    line_width: int = 2
    supersample: int = 2
    background: int = 255
    foreground: int = 0

    @property
    def digest(self) -> str:
        return stable_hash(self.__dict__)


@dataclass(frozen=True)
class DynamicRGBRenderConfig:
    """Versioned reconstruction of the baseline's black-background RGB view."""

    width: int = 448
    height: int = 448
    margin: int = 16
    line_width: int = 4
    supersample: int = 2
    speed_cap_mm_s: float = 800.0
    renderer_version: str = "dynamic_rgb_v2"

    @property
    def digest(self) -> str:
        return stable_hash(self.__dict__)


@dataclass
class PreparedSignature:
    sequence: torch.Tensor
    image: torch.Tensor
    anchors: torch.Tensor
    anchor_mask: torch.Tensor
    sample_id: str


class SignatureStore:
    """Authoritative sample lookup; audit metadata never enters model inputs."""

    def __init__(self, dataset_root: str | Path, verify_hashes: bool = False):
        self.root = Path(dataset_root).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.metadata = manifest
        self.samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
        if len(self.samples) != len(manifest["samples"]):
            raise ValueError("Duplicate sample_id in authoritative manifest")
        if verify_hashes:
            self.validate_all_hashes()

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, sample_id: str) -> dict[str, Any]:
        return self.samples[sample_id]

    def path(self, sample_id: str) -> Path:
        return self.root / self.samples[sample_id]["csv_path"]

    def load_csv(self, sample_id: str) -> np.ndarray:
        array = np.genfromtxt(
            self.path(sample_id), delimiter=",", skip_header=1, dtype=np.float32, encoding="utf-8"
        )
        if array.ndim == 1:
            array = array[None, :]
        if array.shape[1] != 7 or not np.isfinite(array).all():
            raise ValueError(f"Invalid CSV values for {sample_id}: shape={array.shape}")
        return array

    def validate_sha256(self, sample_id: str) -> None:
        expected = self.samples[sample_id]["sha256"]
        actual = sha256_file(self.path(sample_id))
        if actual != expected:
            raise ValueError(f"CSV hash mismatch for {sample_id}: {actual} != {expected}")

    def validate_all_hashes(self) -> None:
        for sample_id in self.samples:
            self.validate_sha256(sample_id)

    def export_benchmark_manifest(self, path: str | Path) -> None:
        fields = (
            "sample_id", "csv_path", "sha256", "writer_id", "label", "state",
            "forger_id", "reference_nw_index", "collection_batch", "session",
            "row_count", "duration_ms", "quality_flags",
        )
        write_jsonl(path, ({key: sample.get(key) for key in fields} for sample in self.samples.values()))


def build_point_features(raw: np.ndarray) -> np.ndarray:
    time, x, y, pressure, speed, direction, pen = raw.T
    duration = max(float(time[-1] - time[0]), 1.0)
    t_rel = (time - time[0]) / duration
    dt = np.diff(time, prepend=time[0]) / 1000.0
    angle = np.deg2rad(direction)
    transition = np.abs(np.diff(pen, prepend=pen[0]))
    return np.stack(
        [t_rel, dt, x - x.mean(), y - y.mean(), pressure, np.log1p(np.maximum(speed, 0)),
         np.sin(angle), np.cos(angle), pen, transition],
        axis=1,
    ).astype(np.float32)


def build_raw_sequence(raw: np.ndarray, time_scale_seconds: float = TIME_SCALE_SECONDS,
                       position_scale_mm: float = POSITION_SCALE_MM) -> np.ndarray:
    """Build fixed-unit primitive inputs without derived speed, direction, or transitions."""
    time = raw[:, 0]
    elapsed = (time - time[0]) / (1000.0 * time_scale_seconds)
    active = raw[:, 6] > 0.5
    origin = raw[np.flatnonzero(active)[0] if active.any() else 0, 1:3]
    xy_from_origin = (raw[:, 1:3] - origin) / position_scale_mm
    pressure = np.clip(raw[:, 3], 0.0, 1.0)
    pen = active.astype(np.float32)
    return np.column_stack([elapsed, xy_from_origin, pressure, pen]).astype(np.float32)


def uniform_subsample(raw: np.ndarray, max_points: int) -> np.ndarray:
    """Keep the complete time span when a diagnostic config limits point count."""
    if len(raw) <= max_points:
        return raw
    indices = np.linspace(0, len(raw) - 1, max_points).round().astype(np.int64)
    return raw[indices]


def channel_indices(names: Sequence[str], feature_names: Sequence[str] = FEATURE_NAMES) -> tuple[int, ...]:
    lookup = {name: index for index, name in enumerate(feature_names)}
    unknown = [name for name in names if name not in lookup]
    if unknown:
        raise ValueError(f"Unknown feature channels: {unknown}")
    return tuple(lookup[name] for name in names)


def zero_feature_channels(features: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if not indices:
        return features
    output = np.array(features, copy=True)
    output[:, list(indices)] = 0.0
    return output


def resample_arc_length(
    raw: np.ndarray,
    n_points: int,
    duration_ms: float,
    constant_pressure: bool = False,
) -> np.ndarray:
    """Destroy original timing by resampling (x, y) on arc length with constant dt."""
    if n_points < 2:
        raise ValueError("Arc-length resampling requires at least 2 points")
    if duration_ms <= 0:
        raise ValueError("Aligned duration must be positive")
    xy = raw[:, 1:3].astype(np.float64)
    step = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(np.maximum(step, 0.0))])
    total = float(cumulative[-1])
    sample = np.linspace(0.0, total, n_points) if total > 1e-8 else np.zeros(n_points)
    x = np.interp(sample, cumulative, raw[:, 1])
    y = np.interp(sample, cumulative, raw[:, 2])
    pressure = (
        np.full(n_points, float(np.clip(raw[:, 3], 0.0, 1.0).mean()))
        if constant_pressure else np.interp(sample, cumulative, np.clip(raw[:, 3], 0.0, 1.0))
    )
    pen = (np.interp(sample, cumulative, raw[:, 6]) >= 0.5).astype(np.float32)
    dt_ms = duration_ms / (n_points - 1)
    time = np.arange(n_points, dtype=np.float64) * dt_ms
    delta = np.diff(np.stack([x, y], axis=1), axis=0, prepend=np.stack([x[:1], y[:1]], axis=1))
    dt_s = max(dt_ms / 1000.0, 1e-6)
    speed = np.sqrt((delta ** 2).sum(axis=1)) / dt_s
    speed[0] = speed[1] if n_points > 1 else 0.0
    direction = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    direction[0] = direction[1] if n_points > 1 else 0.0
    return np.column_stack([time, x, y, pressure, speed, direction, pen]).astype(np.float32)


def genuine_median_duration_ms(
    store: "SignatureStore",
    writer_id: str,
    exclude_sample_id: str | None = None,
) -> float:
    durations = [
        float(sample["duration_ms"])
        for sample in store.samples.values()
        if sample["writer_id"] == writer_id
        and sample["label"] == "genuine"
        and sample["sample_id"] != exclude_sample_id
    ]
    if not durations:
        raise ValueError(f"No genuine durations for writer {writer_id}")
    return float(np.median(durations))


def compute_normalization(store: SignatureStore, writer_ids: set[str], clip_sigma: float = 8.0) -> dict[str, Any]:
    total = 0
    mean = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    m2 = np.zeros_like(mean)
    for sample in store.samples.values():
        if sample["writer_id"] not in writer_ids:
            continue
        values = build_point_features(store.load_csv(sample["sample_id"])).astype(np.float64)
        batch_n = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - mean
        new_total = total + batch_n
        mean += delta * batch_n / new_total
        m2 += batch_m2 + delta * delta * total * batch_n / new_total
        total = new_total
    std = np.sqrt(m2 / max(total - 1, 1)).clip(min=1e-6)
    # Binary and bounded trigonometric channels retain their physical scales.
    passthrough = [0, 6, 7, 8, 9]
    mean[passthrough] = 0.0
    std[passthrough] = 1.0
    return {
        "feature_names": list(FEATURE_NAMES), "count": total,
        "mean": mean.tolist(), "std": std.tolist(), "clip_sigma": clip_sigma,
        "writer_ids": sorted(writer_ids),
    }


class TrajectoryRenderer:
    image_mode = "L"

    def __init__(self, config: RenderConfig):
        self.config = config

    def transform(self, raw: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        cfg = self.config
        xy = raw[:, 1:3].astype(np.float32)
        pen_xy = xy[raw[:, 6] > 0.5]
        bounds = pen_xy if len(pen_xy) else xy
        minimum, maximum = bounds.min(axis=0), bounds.max(axis=0)
        span = np.maximum(maximum - minimum, 1e-4)
        usable_w, usable_h = cfg.width - 2 * cfg.margin, cfg.height - 2 * cfg.margin
        scale = min(usable_w / span[0], usable_h / span[1])
        offset = np.array(
            [(cfg.width - span[0] * scale) / 2 - minimum[0] * scale,
             (cfg.height - span[1] * scale) / 2 - minimum[1] * scale], dtype=np.float32,
        )
        pixels = xy * scale + offset
        transform = {
            "scale": float(scale), "offset_x": float(offset[0]), "offset_y": float(offset[1]),
            "width": float(cfg.width), "height": float(cfg.height),
        }
        return pixels, transform

    def render(self, raw: np.ndarray) -> tuple[Image.Image, dict[str, float]]:
        cfg = self.config
        pixels, transform = self.transform(raw)
        factor = cfg.supersample
        image = Image.new("L", (cfg.width * factor, cfg.height * factor), color=cfg.background)
        draw = ImageDraw.Draw(image)
        pen = raw[:, 6] > 0.5
        start = None
        for index, active in enumerate(pen):
            if active and start is None:
                start = index
            at_end = index == len(pen) - 1
            if start is not None and ((not active) or at_end):
                end = index + 1 if active and at_end else index
                points = [(float(x * factor), float(y * factor)) for x, y in pixels[start:end]]
                if len(points) == 1:
                    x, y = points[0]
                    radius = max(1, cfg.line_width * factor // 2)
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=cfg.foreground)
                elif points:
                    draw.line(points, fill=cfg.foreground, width=cfg.line_width * factor, joint="curve")
                start = None
        if factor > 1:
            image = image.resize((cfg.width, cfg.height), Image.Resampling.LANCZOS)
        return image, transform


class DynamicRGBRenderer:
    """Render pressure, speed, and elapsed time on a black RGB canvas."""

    image_mode = "RGB"

    def __init__(self, config: DynamicRGBRenderConfig):
        self.config = config

    def transform(self, raw: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool]]:
        cfg = self.config
        xy = raw[:, 1:3].astype(np.float32)
        pen_xy = xy[raw[:, 6] > 0.5]
        bounds = pen_xy if len(pen_xy) else xy
        minimum, maximum = bounds.min(axis=0), bounds.max(axis=0)
        span = np.maximum(maximum - minimum, 1e-4)
        usable_w, usable_h = cfg.width - 2 * cfg.margin, cfg.height - 2 * cfg.margin
        scale = min(usable_w / span[0], usable_h / span[1])
        x = (xy[:, 0] - minimum[0]) * scale + (cfg.width - span[0] * scale) / 2
        y = (maximum[1] - xy[:, 1]) * scale + (cfg.height - span[1] * scale) / 2
        pixels = np.stack([x, y], axis=1).astype(np.float32)
        transform: dict[str, float | bool] = {
            "scale": float(scale),
            "minimum_x": float(minimum[0]), "maximum_y": float(maximum[1]),
            "offset_x": float((cfg.width - span[0] * scale) / 2),
            "offset_y": float((cfg.height - span[1] * scale) / 2),
            "width": float(cfg.width), "height": float(cfg.height), "flip_y": True,
        }
        return pixels, transform

    def render(self, raw: np.ndarray) -> tuple[Image.Image, dict[str, float | bool]]:
        cfg = self.config
        pixels, transform = self.transform(raw)
        factor = cfg.supersample
        image = Image.new("RGB", (cfg.width * factor, cfg.height * factor), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        active = raw[:, 6] > 0.5
        time = raw[:, 0]
        duration = max(float(time[-1] - time[0]), 1.0)
        elapsed = np.clip((time - time[0]) / duration, 0.0, 1.0)
        pressure = np.clip(raw[:, 3], 0.0, 1.0)
        speed = np.clip(raw[:, 4] / cfg.speed_cap_mm_s, 0.0, 1.0)

        def color(index: int) -> tuple[int, int, int]:
            return tuple(int(round(255 * value)) for value in (pressure[index], speed[index], elapsed[index]))

        for index in range(len(raw)):
            if not active[index]:
                continue
            point = tuple(float(value * factor) for value in pixels[index])
            if index == 0 or not active[index - 1]:
                radius = max(1, cfg.line_width * factor // 2)
                draw.ellipse((point[0] - radius, point[1] - radius,
                              point[0] + radius, point[1] + radius), fill=color(index))
                continue
            previous = tuple(float(value * factor) for value in pixels[index - 1])
            segment_color = tuple((a + b) // 2 for a, b in zip(color(index - 1), color(index)))
            draw.line((previous, point), fill=segment_color, width=cfg.line_width * factor)
        if factor > 1:
            image = image.resize((cfg.width, cfg.height), Image.Resampling.LANCZOS)
        return image, transform


def temporal_anchors(raw: np.ndarray, transform: dict[str, Any], stride: int = 8) -> tuple[np.ndarray, np.ndarray]:
    anchors: list[list[float]] = []
    valid: list[bool] = []
    width, height = transform["width"], transform["height"]
    for start in range(0, len(raw), stride):
        window = raw[start:start + stride]
        points = window[window[:, 6] > 0.5, 1:3]
        if not len(points):
            anchors.append([0.0] * 6)
            valid.append(False)
            continue
        center, std = points.mean(axis=0), points.std(axis=0)
        span = points.max(axis=0) - points.min(axis=0)
        if transform.get("flip_y", False):
            center = np.asarray([
                (center[0] - transform["minimum_x"]) * transform["scale"] + transform["offset_x"],
                (transform["maximum_y"] - center[1]) * transform["scale"] + transform["offset_y"],
            ], dtype=np.float32)
        else:
            center = center * transform["scale"] + np.array([transform["offset_x"], transform["offset_y"]])
        std = std * transform["scale"]
        span = span * transform["scale"]
        anchors.append([
            float(center[0] / max(width - 1, 1) * 2 - 1),
            float(center[1] / max(height - 1, 1) * 2 - 1),
            float(std[0] / width), float(std[1] / height),
            float(span[0] / width), float(span[1] / height),
        ])
        valid.append(True)
    return np.asarray(anchors, dtype=np.float32), np.asarray(valid, dtype=np.bool_)


class SignaturePreprocessor:
    def __init__(self, store: SignatureStore, stats: dict[str, Any] | None, cache_root: str | Path,
                 render_config: RenderConfig | DynamicRGBRenderConfig, max_points: int = 6144,
                 cache_images: bool = True, memory_cache: bool = True, memory_cache_items: int = 256,
                 input_pipeline: str = "legacy_v1", raw_time_scale_seconds: float = TIME_SCALE_SECONDS,
                 raw_position_scale_mm: float = POSITION_SCALE_MM,
                 zero_channels: Sequence[str] = (), time_align: str | None = None,
                 time_align_points: int = 256, time_align_constant_pressure: bool = False,
                 time_align_duration_ms: float | None = None):
        self.store = store
        self.input_pipeline = input_pipeline
        supported = {"legacy_v1", "rgb_legacy_ablation", "gray_raw_ablation", "raw_rgb_v2"}
        if input_pipeline not in supported:
            raise ValueError(f"Unsupported input pipeline: {input_pipeline}")
        self.legacy_sequence = input_pipeline in {"legacy_v1", "rgb_legacy_ablation"}
        self.rgb_image = input_pipeline in {"rgb_legacy_ablation", "raw_rgb_v2"}
        if self.legacy_sequence:
            if stats is None:
                raise ValueError(f"{input_pipeline} requires fold normalization")
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            self.std = np.asarray(stats["std"], dtype=np.float32)
            self.clip = float(stats.get("clip_sigma", 8.0))
        else:
            self.mean = self.std = None
            self.clip = 0.0
        if self.rgb_image:
            if not isinstance(render_config, DynamicRGBRenderConfig):
                raise ValueError(f"{input_pipeline} requires DynamicRGBRenderConfig")
            self.renderer = DynamicRGBRenderer(render_config)
        else:
            if not isinstance(render_config, RenderConfig):
                raise ValueError(f"{input_pipeline} requires RenderConfig")
            self.renderer = TrajectoryRenderer(render_config)
        self.cache_root = Path(cache_root)
        self.raw_time_scale_seconds = raw_time_scale_seconds
        self.raw_position_scale_mm = raw_position_scale_mm
        self.max_points = max_points
        self.cache_images = cache_images
        self.memory_cache = memory_cache
        self.memory_cache_items = max(0, memory_cache_items)
        names = FEATURE_NAMES if self.legacy_sequence else RAW_SEQUENCE_NAMES
        self.zero_channels = tuple(zero_channels)
        self.zero_indices = channel_indices(self.zero_channels, names) if self.zero_channels else ()
        if time_align not in {None, "arc_length"}:
            raise ValueError(f"Unsupported time_align: {time_align}")
        self.time_align = time_align
        self.time_align_points = int(time_align_points)
        self.time_align_constant_pressure = bool(time_align_constant_pressure)
        self.time_align_duration_ms = time_align_duration_ms
        self._prepared_cache: OrderedDict[str, PreparedSignature] = OrderedDict()

    def _align_digest(self) -> str:
        if self.time_align is None:
            return ""
        return stable_hash({
            "time_align": self.time_align,
            "time_align_points": self.time_align_points,
            "time_align_constant_pressure": self.time_align_constant_pressure,
            "time_align_duration_ms": self.time_align_duration_ms,
        })

    def _aligned_duration(self, sample_id: str) -> float:
        if self.time_align_duration_ms is not None:
            return float(self.time_align_duration_ms)
        writer_id = self.store.get(sample_id)["writer_id"]
        return genuine_median_duration_ms(self.store, writer_id, exclude_sample_id=sample_id)

    def _raw_for_sample(self, sample_id: str) -> np.ndarray:
        complete = self.store.load_csv(sample_id)
        if self.time_align == "arc_length":
            complete = resample_arc_length(
                complete, self.time_align_points, self._aligned_duration(sample_id),
                constant_pressure=self.time_align_constant_pressure,
            )
        if self.legacy_sequence:
            return complete[:self.max_points]
        return uniform_subsample(complete, self.max_points)

    def _cache_paths(self, sample_id: str) -> tuple[Path, Path, str]:
        sample = self.store.get(sample_id)
        key = hashlib.sha256(
            (sample["sha256"] + self.renderer.config.digest + self._align_digest()).encode()
        ).hexdigest()
        return self.cache_root / f"{key}.png", self.cache_root / f"{key}.json", key

    def _read_cached_image(self, png_path: Path, metadata_path: Path, key: str) -> tuple[Image.Image, dict[str, float]] | None:
        if not png_path.exists() or not metadata_path.exists():
            return None
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if meta.get("cache_key") != key or sha256_file(png_path) != meta.get("png_sha256"):
                return None
            with Image.open(png_path) as cached:
                image = cached.convert(self.renderer.image_mode)
            return image, meta["transform"]
        except (OSError, PermissionError, json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def _publish_cached_png(temporary_png: Path, png_path: Path) -> str:
        """Publish one deterministic cache file despite concurrent Windows workers."""
        expected_sha256 = sha256_file(temporary_png)
        for attempt in range(8):
            if png_path.exists():
                try:
                    if sha256_file(png_path) == expected_sha256:
                        temporary_png.unlink(missing_ok=True)
                        return expected_sha256
                except (OSError, PermissionError):
                    pass
            try:
                temporary_png.replace(png_path)
                return expected_sha256
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.025 * (attempt + 1))
        raise RuntimeError("unreachable cache publication state")

    def _image(self, sample_id: str, raw: np.ndarray) -> tuple[Image.Image, dict[str, float]]:
        png_path, metadata_path, key = self._cache_paths(sample_id)
        cached = self._read_cached_image(png_path, metadata_path, key)
        if cached is not None:
            return cached
        image, transform = self.renderer.render(raw)
        if self.cache_images:
            png_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_png = png_path.with_suffix(f".{os.getpid()}.tmp.png")
            image.save(temporary_png, format="PNG", optimize=True)
            png_sha256 = self._publish_cached_png(temporary_png, png_path)
            meta = {
                "sample_id": sample_id, "csv_sha256": self.store.get(sample_id)["sha256"],
                "renderer_sha256": self.renderer.config.digest, "cache_key": key,
                "png_sha256": png_sha256, "width": image.width, "height": image.height,
                "transform": transform, "input_pipeline": self.input_pipeline,
            }
            atomic_json(metadata_path, meta)
        return image, transform

    def prepare(self, sample_id: str) -> PreparedSignature:
        if self.memory_cache and sample_id in self._prepared_cache:
            prepared = self._prepared_cache[sample_id]
            self._prepared_cache.move_to_end(sample_id)
            return prepared
        raw = self._raw_for_sample(sample_id)
        image, transform = self._image(sample_id, raw)
        anchors, anchor_mask = temporal_anchors(raw, transform)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        if self.legacy_sequence:
            assert self.mean is not None and self.std is not None
            features = np.clip((build_point_features(raw) - self.mean) / self.std, -self.clip, self.clip)
        else:
            features = build_raw_sequence(
                raw, time_scale_seconds=self.raw_time_scale_seconds,
                position_scale_mm=self.raw_position_scale_mm,
            )
        features = zero_feature_channels(features, self.zero_indices)
        if self.rgb_image:
            image_tensor = torch.as_tensor(pixels, dtype=torch.float32).permute(2, 0, 1)
        else:
            image_tensor = torch.as_tensor(1.0 - pixels, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        prepared = PreparedSignature(
            sequence=torch.as_tensor(features, dtype=torch.float32), image=image_tensor,
            anchors=torch.as_tensor(anchors, dtype=torch.float32),
            anchor_mask=torch.as_tensor(anchor_mask, dtype=torch.bool), sample_id=sample_id,
        )
        if self.memory_cache and self.memory_cache_items:
            self._prepared_cache[sample_id] = prepared
            while len(self._prepared_cache) > self.memory_cache_items:
                self._prepared_cache.popitem(last=False)
        return prepared


def _writer_folds(writers: list[str], seed: int) -> list[list[str]]:
    ordered = list(writers)
    random.Random(seed).shuffle(ordered)
    sizes = [16, 16, 16, 16, 15]
    folds, cursor = [], 0
    for size in sizes:
        folds.append(sorted(ordered[cursor:cursor + size]))
        cursor += size
    return folds


def _sample_index(sample: dict[str, Any]) -> int:
    return int(sample["sample_id"].rsplit("-", 1)[-1])


class EpisodeBuilder:
    def __init__(self, store: SignatureStore, benchmark_root: str | Path, seed: int = 20260723):
        self.store = store
        self.root = Path(benchmark_root)
        self.seed = seed
        self.by_writer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in store.samples.values():
            self.by_writer[sample["writer_id"]].append(sample)
        self.writers = sorted(self.by_writer)

    def build(self) -> dict[str, Any]:
        self.store.export_benchmark_manifest(self.root / "samples/signature_manifest.jsonl")
        folds = _writer_folds(self.writers, self.seed)
        rotations: dict[str, dict[str, list[str]]] = {}
        for fold_id in range(5):
            test_idx, val_idx = fold_id, (fold_id + 1) % 5
            train = sorted(writer for index, fold in enumerate(folds) if index not in {test_idx, val_idx} for writer in fold)
            rotations[str(fold_id)] = {"train": train, "val": folds[val_idx], "test": folds[test_idx]}
        atomic_json(self.root / "folds/writer_folds.json", {"seed": self.seed, "folds": folds})
        atomic_json(self.root / "folds/fold_rotation.json", rotations)
        renderer = RenderConfig()
        atomic_json(self.root / "renderer/renderer.json", renderer.__dict__)
        (self.root / "renderer/renderer.sha256").write_text(renderer.digest + "\n", encoding="ascii")
        dynamic_renderer = DynamicRGBRenderConfig()
        atomic_json(self.root / "renderer/dynamic_rgb_v2.json", dynamic_renderer.__dict__)
        (self.root / "renderer/dynamic_rgb_v2.sha256").write_text(dynamic_renderer.digest + "\n", encoding="ascii")

        counts: dict[str, dict[str, int]] = {}
        for fold_id, split_map in rotations.items():
            counts[fold_id] = {}
            train_writers = set(split_map["train"])
            stats = compute_normalization(self.store, train_writers)
            atomic_json(self.root / f"folds/fold_{fold_id}_normalization.json", stats)
            for split, writers in split_map.items():
                rng = random.Random(self.seed + int(fold_id) * 1009 + {"train": 11, "val": 17, "test": 23}[split])
                t1_1 = self._build_t1(writers, fold_id, split, rng, references=1)
                t1_5 = self._build_t1(writers, fold_id, split, rng, references=5)
                t2_8 = self._build_t2(writers, fold_id, split, rng, candidate_count=8, include_absent=True)
                t2_4 = self._build_t2(writers, fold_id, split, rng, candidate_count=4, include_absent=True)
                t2_20 = self._build_t2(writers, fold_id, split, rng, candidate_count=20, include_absent=False)
                folder = self.root / f"episodes/fold_{fold_id}"
                files = {
                    f"{split}_t1_1v1.jsonl": t1_1, f"{split}_t1_5v1.jsonl": t1_5,
                    f"{split}_t2.jsonl": t2_8, f"{split}_t2_l4.jsonl": t2_4,
                    f"{split}_t2_l20_restricted.jsonl": t2_20,
                }
                for name, rows in files.items():
                    write_jsonl(folder / name, rows)
                    counts[fold_id][name] = len(rows)
        summary = {
            "schema_version": "1.1.0", "seed": self.seed, "counts": counts,
            "t2_primary_distribution": "source_present:source_absent:rf_no_source=2:1:1",
            "t2_l20_scope": "source_present_and_rf_only; source_absent is impossible with 20 NW and source excluded",
        }
        atomic_json(self.root / "benchmark_manifest.json", summary)
        return summary

    def _genuine(self, writer: str) -> list[dict[str, Any]]:
        return sorted((s for s in self.by_writer[writer] if s["label"] == "genuine"), key=lambda s: s["sample_id"])

    def _references(self, writer: str, query: dict[str, Any], count: int, rng: random.Random) -> list[dict[str, Any]]:
        pool = [s for s in self._genuine(writer) if s["sha256"] != query["sha256"]]
        if count == 1:
            return [rng.choice(pool)]
        by_state = defaultdict(list)
        for sample in pool:
            by_state[sample["state"]].append(sample)
        selected = [rng.choice(by_state[state]) for state in ("NW", "HPG", "SW", "VSW")]
        remaining = [s for s in pool if s not in selected]
        selected.append(rng.choice(remaining))
        rng.shuffle(selected)
        return selected

    def _build_t1(self, writers: Sequence[str], fold_id: str, split: str, rng: random.Random,
                  references: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        protocol = "t1_1v1" if references == 1 else "t1_5v1"
        for writer in writers:
            genuine = self._genuine(writer)
            positives: list[dict[str, Any]] = []
            for state in ("NW", "HPG", "SW", "VSW"):
                candidates = [s for s in genuine if s["state"] == state]
                rng.shuffle(candidates)
                positives.extend(candidates[:10])
            other_genuine = [s for other in writers if other != writer for s in self._genuine(other)]
            rng.shuffle(other_genuine)
            zero = other_genuine[:20]
            forged = [s for s in self.by_writer[writer] if s["state"] in {"RF", "SF"}]
            cases = [(s, 1, "genuine") for s in positives]
            cases += [(s, 0, "zero_effort") for s in zero]
            cases += [(s, 0, s["state"]) for s in sorted(forged, key=lambda x: x["sample_id"])]
            for query, label, attack in cases:
                refs = self._references(writer, query, references, rng)
                index = len(rows)
                rows.append({
                    "episode_id": f"F{fold_id}-{split.upper()}-T1-{references}V1-{index:06d}",
                    "fold_id": int(fold_id), "split": split, "protocol": protocol,
                    "reference_ids": [r["sample_id"] for r in refs], "query_id": query["sample_id"],
                    "target_writer_id": writer, "label": label, "attack_type": attack,
                    "forger_id": query.get("forger_id"),
                    "reference_states": [r["state"] for r in refs], "query_state": query["state"],
                    "condition_relation": "same" if all(r["state"] == query["state"] for r in refs) else "cross_or_mixed",
                    "sampling_seed": self.seed, "statistical_unit": writer,
                })
        rng.shuffle(rows)
        return rows

    def _build_t2(self, writers: Sequence[str], fold_id: str, split: str, rng: random.Random,
                  candidate_count: int, include_absent: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for writer in writers:
            nw = sorted((s for s in self.by_writer[writer] if s["state"] == "NW"), key=_sample_index)
            nw_by_index = {_sample_index(s): s for s in nw}
            sf = sorted((s for s in self.by_writer[writer] if s["state"] == "SF"), key=lambda s: s["sample_id"])
            rf = sorted((s for s in self.by_writer[writer] if s["state"] == "RF"), key=lambda s: s["sample_id"])
            position_offset = rng.randrange(candidate_count)
            target_present = len(rf) + (len(sf) if include_absent else 0)
            repeats, extra = divmod(target_present, len(sf))
            present_index = 0
            for sf_offset, query in enumerate(sf):
                source = nw_by_index[int(query["reference_nw_index"])]
                decoys = [s for s in nw if s["sample_id"] != source["sample_id"]]
                for repeat in range(repeats + int(sf_offset < extra)):
                    chosen = rng.sample(decoys, candidate_count - 1)
                    rng.shuffle(chosen)
                    source_position = (present_index + position_offset) % candidate_count
                    chosen.insert(source_position, source)
                    rows.append(self._t2_row(
                        fold_id, split, query, chosen, source_position, "source_present", writer, repeat,
                    ))
                    present_index += 1
                if include_absent:
                    chosen = rng.sample(decoys, candidate_count)
                    rng.shuffle(chosen)
                    rows.append(self._t2_row(fold_id, split, query, chosen, -1, "source_absent", writer, sf_offset))
            for rf_offset, query in enumerate(rf):
                chosen = rng.sample(nw, candidate_count)
                rng.shuffle(chosen)
                rows.append(self._t2_row(fold_id, split, query, chosen, -1, "rf_no_source", writer, rf_offset))
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            row["episode_id"] = f"F{fold_id}-{split.upper()}-T2-L{candidate_count}-{index:06d}"
        return rows

    def _t2_row(self, fold_id: str, split: str, query: dict[str, Any], candidates: list[dict[str, Any]],
                target: int, episode_type: str, writer: str, repeat: int) -> dict[str, Any]:
        return {
            "episode_id": "", "fold_id": int(fold_id), "split": split, "protocol": "t2_source_ranking",
            "query_id": query["sample_id"], "candidate_ids": [s["sample_id"] for s in candidates],
            "target_index": target, "target_type": "candidate" if target >= 0 else "unknown",
            "episode_type": episode_type, "candidate_count": len(candidates), "target_writer_id": writer,
            "forger_id": query.get("forger_id"), "reference_nw_index": query.get("reference_nw_index"),
            "collection_batch": query.get("collection_batch"), "sampling_seed": self.seed,
            "candidate_collection_batches": [sample.get("collection_batch") for sample in candidates],
            "repeated_query_group": query["sample_id"], "statistical_unit": writer, "repeat_index": repeat,
        }


class T1SingleSplitBuilder:
    """Materialize the frozen 70:15:15 T1 protocol without zero-effort episodes."""

    # Preserve the frozen writer assignment: shuffled writers were allocated
    # as Train, Test, then Validation before their semantic roles were finalized.
    SPLIT_SIZES = {"train": 55, "test": 12, "val": 12}
    REPEATS = {
        "train": {"genuine": 2, "RF": 8, "SF": 8},
        "val": {"genuine": 1, "RF": 4, "SF": 4},
        "test": {"genuine": 1, "RF": 4, "SF": 4},
    }

    def __init__(self, store: SignatureStore, benchmark_root: str | Path, seed: int = 20260731):
        self.store = store
        self.root = Path(benchmark_root)
        self.seed = seed
        self.by_writer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in store.samples.values():
            self.by_writer[sample["writer_id"]].append(sample)
        self.writers = sorted(self.by_writer)
        if len(self.writers) != sum(self.SPLIT_SIZES.values()):
            raise ValueError(
                f"T1 single-split protocol requires 79 writers, found {len(self.writers)}"
            )

    def build(self) -> dict[str, Any]:
        self.store.export_benchmark_manifest(self.root / "samples/signature_manifest.jsonl")
        shuffled = list(self.writers)
        random.Random(self.seed).shuffle(shuffled)
        split_map: dict[str, list[str]] = {}
        cursor = 0
        for split, size in self.SPLIT_SIZES.items():
            split_map[split] = shuffled[cursor:cursor + size]
            cursor += size
        split_record = {
            "schema_version": "1.0.0",
            "algorithm": "python_random.Random(seed).shuffle(sorted_writer_ids)",
            "seed": self.seed,
            "source_writer_count": len(self.writers),
            "requested_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
            "realized_counts": self.SPLIT_SIZES,
            "realized_ratio": {
                split: len(writers) / len(self.writers) for split, writers in split_map.items()
            },
            "shuffled_writer_order": shuffled,
            "splits": split_map,
            "split_digest": stable_hash(split_map),
        }
        atomic_json(self.root / "folds/single_split.json", split_record)
        atomic_json(self.root / "folds/fold_rotation.json", {"0": split_map})
        atomic_json(
            self.root / "folds/fold_0_normalization.json",
            compute_normalization(self.store, set(split_map["train"])),
        )
        renderer = RenderConfig()
        atomic_json(self.root / "renderer/renderer.json", renderer.__dict__)
        (self.root / "renderer/renderer.sha256").write_text(renderer.digest + "\n", encoding="ascii")
        dynamic_renderer = DynamicRGBRenderConfig()
        atomic_json(self.root / "renderer/dynamic_rgb_v2.json", dynamic_renderer.__dict__)
        (self.root / "renderer/dynamic_rgb_v2.sha256").write_text(
            dynamic_renderer.digest + "\n", encoding="ascii"
        )

        counts: dict[str, int] = {}
        for split, writers in split_map.items():
            for reference_count in (1, 5):
                rows = self._build_t1(writers, split, reference_count)
                name = f"{split}_t1_{reference_count}v1.jsonl"
                write_jsonl(self.root / "episodes/fold_0" / name, rows)
                counts[name] = len(rows)

        summary = {
            "schema_version": "2.0.0-t1-single-split",
            "task": "T1 online signature verification",
            "seed": self.seed,
            "writer_split": {split: len(writers) for split, writers in split_map.items()},
            "writer_split_digest": split_record["split_digest"],
            "episode_counts": counts,
            "total_episodes": sum(counts.values()),
            "protocols": ["t1_1v1", "t1_5v1"],
            "zero_effort_included": False,
            "query_policy": "within target writer; genuine positive, RF/SF negative",
            "reference_policy": (
                "target-writer genuine references; no state quota; distinct IDs and content; "
                "unique reference configurations per query"
            ),
            "per_writer": {
                "train": {"genuine": 160, "RF": 96, "SF": 64, "total": 320},
                "val": {"genuine": 80, "RF": 48, "SF": 32, "total": 160},
                "test": {"genuine": 80, "RF": 48, "SF": 32, "total": 160},
            },
            "class_ratio": {"positive": 0.5, "negative": 0.5},
            "attack_ratio_all_episodes": {"genuine": 0.5, "RF": 0.3, "SF": 0.2},
            "statistical_unit": "writer; repeated query/reference configurations are clustered",
        }
        atomic_json(self.root / "benchmark_manifest.json", summary)
        return summary

    def _genuine(self, writer: str) -> list[dict[str, Any]]:
        return sorted(
            (sample for sample in self.by_writer[writer] if sample["label"] == "genuine"),
            key=lambda sample: sample["sample_id"],
        )

    def _queries(self, writer: str, attack_type: str) -> list[dict[str, Any]]:
        if attack_type == "genuine":
            return self._genuine(writer)
        return sorted(
            (sample for sample in self.by_writer[writer] if sample["state"] == attack_type),
            key=lambda sample: sample["sample_id"],
        )

    @staticmethod
    def _choose_references(
        pool: list[dict[str, Any]], count: int, usage: Counter[str],
        seen: set[tuple[str, ...]], rng: random.Random,
    ) -> list[dict[str, Any]]:
        if len(pool) < count:
            raise ValueError(f"Need {count} references, found {len(pool)}")
        candidates: list[tuple[tuple[int, int, float], list[dict[str, Any]], tuple[str, ...]]] = []
        if count == 1:
            for sample in pool:
                key = (sample["sample_id"],)
                if key not in seen:
                    candidates.append(((usage[key[0]], usage[key[0]], rng.random()), [sample], key))
        else:
            for _ in range(256):
                selected = rng.sample(pool, count)
                key = tuple(sorted(sample["sample_id"] for sample in selected))
                if key in seen:
                    continue
                counts = [usage[sample_id] for sample_id in key]
                candidates.append(((sum(counts), max(counts), rng.random()), selected, key))
        if not candidates:
            raise ValueError("Unable to construct a new unique reference configuration")
        _, selected, key = min(candidates, key=lambda item: item[0])
        seen.add(key)
        for sample in selected:
            usage[sample["sample_id"]] += 1
        rng.shuffle(selected)
        return selected

    def _build_t1(self, writers: Sequence[str], split: str, reference_count: int) -> list[dict[str, Any]]:
        protocol = f"t1_{reference_count}v1"
        split_offset = {"train": 11, "val": 17, "test": 23}[split]
        protocol_offset = 101 if reference_count == 1 else 503
        rng = random.Random(self.seed + split_offset + protocol_offset)
        rows: list[dict[str, Any]] = []
        for writer in writers:
            genuine = self._genuine(writer)
            usage: Counter[str] = Counter()
            seen_by_query: dict[str, set[tuple[str, ...]]] = defaultdict(set)
            for attack_type in ("genuine", "RF", "SF"):
                queries = self._queries(writer, attack_type)
                repeat_count = self.REPEATS[split][attack_type]
                for query in queries:
                    pool = [sample for sample in genuine if sample["sha256"] != query["sha256"]]
                    for repeat_index in range(repeat_count):
                        references = self._choose_references(
                            pool, reference_count, usage,
                            seen_by_query[query["sample_id"]], rng,
                        )
                        reference_ids = [sample["sample_id"] for sample in references]
                        rows.append({
                            "episode_id": "",
                            "fold_id": 0,
                            "split": split,
                            "protocol": protocol,
                            "reference_ids": reference_ids,
                            "query_id": query["sample_id"],
                            "target_writer_id": writer,
                            "label": int(attack_type == "genuine"),
                            "attack_type": attack_type,
                            "forger_id": query.get("forger_id"),
                            "reference_states": [sample["state"] for sample in references],
                            "query_state": query["state"],
                            "condition_relation": (
                                "same" if all(sample["state"] == query["state"] for sample in references)
                                else "cross_or_mixed"
                            ),
                            "sampling_seed": self.seed,
                            "statistical_unit": writer,
                            "query_group_id": f"{split}:{protocol}:{writer}:{query['sample_id']}",
                            "reference_repeat_index": repeat_index,
                            "reference_set_hash": stable_hash(sorted(reference_ids)),
                        })
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            row["episode_id"] = f"S0-{split.upper()}-T1-{reference_count}V1-{index:06d}"
        return rows


def audit_t1_single_benchmark(
    dataset_root: str | Path, benchmark_root: str | Path, full_hash: bool = False,
) -> dict[str, Any]:
    store = SignatureStore(dataset_root, verify_hashes=full_hash)
    root = Path(benchmark_root)
    split_record = json.loads((root / "folds/single_split.json").read_text(encoding="utf-8"))
    splits = {name: set(writers) for name, writers in split_record["splits"].items()}
    errors: list[str] = []
    if {name: len(writers) for name, writers in splits.items()} != T1SingleSplitBuilder.SPLIT_SIZES:
        errors.append("writer split counts are not 55/12/12")
    if set.union(*splits.values()) != set(store.metadata["writers"] if "writers" in store.metadata else {
        sample["writer_id"] for sample in store.samples.values()
    }):
        errors.append("writer split union does not equal all dataset writers")
    if any(splits[left] & splits[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        errors.append("writer leakage between train/val/test")
    expected_per_writer = {"train": 320, "val": 160, "test": 160}
    expected_attack = {
        "train": {"genuine": 160, "RF": 96, "SF": 64},
        "val": {"genuine": 80, "RF": 48, "SF": 32},
        "test": {"genuine": 80, "RF": 48, "SF": 32},
    }
    checked: dict[str, int] = {}
    file_stats: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        for reference_count in (1, 5):
            path = root / "episodes/fold_0" / f"{split}_t1_{reference_count}v1.jsonl"
            rows = read_jsonl(path)
            checked[path.name] = len(rows)
            ids_seen: set[str] = set()
            configurations: set[tuple[str, str]] = set()
            per_writer = defaultdict(Counter)
            reference_usage = Counter()
            for row in rows:
                episode_id = row["episode_id"]
                protocol = f"t1_{reference_count}v1"
                if episode_id in ids_seen:
                    errors.append(f"{path.name}: duplicate episode_id {episode_id}")
                ids_seen.add(episode_id)
                writer = row["target_writer_id"]
                per_writer[writer][row["attack_type"]] += 1
                if writer not in splits[split]:
                    errors.append(f"{episode_id}: target writer outside {split}")
                if row.get("split") != split:
                    errors.append(f"{episode_id}: row split does not match {path.name}")
                if row.get("protocol") != protocol:
                    errors.append(f"{episode_id}: protocol does not match {path.name}")
                if not episode_id.startswith(f"S0-{split.upper()}-T1-{reference_count}V1-"):
                    errors.append(f"{episode_id}: episode ID does not match {path.name}")
                expected_query_group = f"{split}:{protocol}:{writer}:{row['query_id']}"
                if row.get("query_group_id") != expected_query_group:
                    errors.append(f"{episode_id}: query group does not match {path.name}")
                if row["attack_type"] == "zero_effort":
                    errors.append(f"{episode_id}: zero-effort is forbidden")
                query = store.get(row["query_id"])
                references = [store.get(sample_id) for sample_id in row["reference_ids"]]
                if len(references) != reference_count:
                    errors.append(f"{episode_id}: expected {reference_count} references")
                if len({sample["sample_id"] for sample in references}) != reference_count:
                    errors.append(f"{episode_id}: repeated reference ID")
                if len({sample["sha256"] for sample in references}) != reference_count:
                    errors.append(f"{episode_id}: repeated reference content")
                if any(sample["writer_id"] != writer or sample["label"] != "genuine" for sample in references):
                    errors.append(f"{episode_id}: invalid target-writer genuine reference")
                if query["sha256"] in {sample["sha256"] for sample in references}:
                    errors.append(f"{episode_id}: query content appears in references")
                expected_label = int(query["label"] == "genuine")
                if query["writer_id"] != writer or row["label"] != expected_label:
                    errors.append(f"{episode_id}: query/label is not a within-writer T1 case")
                expected_type = "genuine" if expected_label else query["state"]
                if row["attack_type"] != expected_type or expected_type not in {"genuine", "RF", "SF"}:
                    errors.append(f"{episode_id}: attack type mismatch")
                key = (row["query_id"], row["reference_set_hash"])
                if key in configurations:
                    errors.append(f"{episode_id}: duplicate query/reference configuration")
                configurations.add(key)
                if row["reference_set_hash"] != stable_hash(sorted(row["reference_ids"])):
                    errors.append(f"{episode_id}: invalid reference_set_hash")
                reference_usage.update(row["reference_ids"])
            for writer in splits[split]:
                if sum(per_writer[writer].values()) != expected_per_writer[split]:
                    errors.append(f"{path.name}: {writer} episode count mismatch")
                if dict(per_writer[writer]) != expected_attack[split]:
                    errors.append(f"{path.name}: {writer} attack distribution mismatch")
            file_stats[path.name] = {
                "episodes": len(rows),
                "writers": len(per_writer),
                "positive": sum(row["label"] == 1 for row in rows),
                "negative": sum(row["label"] == 0 for row in rows),
                "attack_counts": dict(Counter(row["attack_type"] for row in rows)),
                "reference_usage_min": min(reference_usage.values()),
                "reference_usage_max": max(reference_usage.values()),
            }
    report = {
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors[:100],
        "writer_split": {name: len(writers) for name, writers in splits.items()},
        "split_digest": split_record["split_digest"],
        "episodes_checked": checked,
        "files": file_stats,
    }
    atomic_json(root / "audit_report.json", report)
    if errors:
        raise ValueError(f"T1 benchmark audit failed with {len(errors)} errors; see audit_report.json")
    return report


class T2SingleSplitBuilder:
    """Materialize T2 episodes on the already-frozen T1 writer split."""

    PRESENT_PER_SF = 12
    ABSENT_PER_SF = 6
    RF_PER_QUERY = 4
    DEFAULT_CANDIDATE_COUNT = 8
    EXPECTED_PER_WRITER = {
        "source_present": 96,
        "source_absent": 48,
        "rf_no_source": 48,
    }

    def __init__(
        self,
        store: SignatureStore,
        benchmark_root: str | Path,
        seed: int = 20260731,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT,
        source_benchmark_root: str | Path | None = None,
        swap_val_test: bool = False,
    ):
        if not 2 <= candidate_count <= 19:
            raise ValueError("T2 candidate_count must be between 2 and 19 when Absent pools are enabled")
        if swap_val_test and source_benchmark_root is None:
            raise ValueError("swap_val_test requires source_benchmark_root")
        self.store = store
        self.root = Path(benchmark_root)
        self.seed = seed
        self.candidate_count = candidate_count
        self.source_benchmark_root = (
            Path(source_benchmark_root) if source_benchmark_root is not None else None
        )
        self.swap_val_test = swap_val_test
        self.by_writer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in store.samples.values():
            self.by_writer[sample["writer_id"]].append(sample)

    def _materialize_benchmark_context(self) -> None:
        if self.source_benchmark_root is None:
            return
        source = self.source_benchmark_root
        if source.resolve() == self.root.resolve():
            raise ValueError("source_benchmark_root and benchmark_root must be different")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(
                f"Refusing to populate non-empty derived benchmark directory: {self.root}"
            )

        source_split_path = source / "folds/single_split.json"
        if not source_split_path.is_file():
            raise FileNotFoundError(f"Source benchmark split not found: {source_split_path}")
        source_record = json.loads(source_split_path.read_text(encoding="utf-8"))
        source_splits = source_record["splits"]
        split_map = {
            "train": list(source_splits["train"]),
            "val": list(source_splits["test"] if self.swap_val_test else source_splits["val"]),
            "test": list(source_splits["val"] if self.swap_val_test else source_splits["test"]),
        }
        if {name: len(writers) for name, writers in split_map.items()} != T1SingleSplitBuilder.SPLIT_SIZES:
            raise ValueError("Source benchmark does not contain the required 55/12/12 writer split")

        split_record = {
            "schema_version": "1.1.0-t2-derived-split",
            "algorithm": (
                "source split with validation/test roles swapped"
                if self.swap_val_test else "source split copied without role changes"
            ),
            "seed": source_record.get("seed", self.seed),
            "source_writer_count": sum(len(writers) for writers in split_map.values()),
            "source_benchmark": source.as_posix(),
            "source_split_digest": source_record.get("split_digest"),
            "realized_counts": {name: len(writers) for name, writers in split_map.items()},
            "realized_ratio": {
                name: len(writers) / sum(len(group) for group in split_map.values())
                for name, writers in split_map.items()
            },
            "splits": split_map,
            "split_digest": stable_hash(split_map),
        }
        atomic_json(self.root / "folds/single_split.json", split_record)
        atomic_json(self.root / "folds/fold_rotation.json", {"0": split_map})

        copied_files: list[dict[str, Any]] = []
        normalization_source = source / "folds/fold_0_normalization.json"
        normalization_target = self.root / "folds/fold_0_normalization.json"
        if not normalization_source.is_file():
            raise FileNotFoundError(f"Source normalization not found: {normalization_source}")
        normalization_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(normalization_source, normalization_target)
        copied_files.append({
            "source": str(normalization_source.relative_to(source)),
            "destination": str(normalization_target.relative_to(self.root)),
            "sha256": sha256_file(normalization_target),
        })

        role_source = {
            "train": "train",
            "val": "test" if self.swap_val_test else "val",
            "test": "val" if self.swap_val_test else "test",
        }
        for destination_split, source_split in role_source.items():
            for reference_count in (1, 5):
                name = f"{source_split}_t1_{reference_count}v1.jsonl"
                source_path = source / "episodes/fold_0" / name
                target_path = (
                    self.root / "episodes/fold_0"
                    / f"{destination_split}_t1_{reference_count}v1.jsonl"
                )
                if not source_path.is_file():
                    raise FileNotFoundError(f"Source T1 episode file not found: {source_path}")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                copied_files.append({
                    "source": str(source_path.relative_to(source)),
                    "destination": str(target_path.relative_to(self.root)),
                    "sha256": sha256_file(target_path),
                })

        renderer_source = source / "renderer"
        if renderer_source.is_dir():
            shutil.copytree(renderer_source, self.root / "renderer")

        context_manifest = {
            "schema_version": "2.0.0-final-role-copy",
            "source_benchmark": source.as_posix(),
            "source_split_digest": source_record.get("split_digest"),
            "derived_split_digest": split_record["split_digest"],
            "candidate_count": self.candidate_count,
            "copied_files": copied_files,
        }
        if self.swap_val_test:
            context_manifest["historical_role_exchange"] = True
        atomic_json(self.root / "t2_context_manifest.json", context_manifest)

    @staticmethod
    def _unique_sample(
        pool: list[dict[str, Any]], count: int, seen: set[tuple[str, ...]], rng: random.Random,
    ) -> list[dict[str, Any]]:
        for _ in range(4096):
            selected = rng.sample(pool, count)
            key = tuple(sorted(sample["sample_id"] for sample in selected))
            if key not in seen:
                seen.add(key)
                return selected
        raise ValueError("Unable to construct a unique T2 candidate pool")

    def build(self) -> dict[str, Any]:
        self._materialize_benchmark_context()
        split_path = self.root / "folds/single_split.json"
        if not split_path.exists():
            raise FileNotFoundError(
                "T2 single-split construction requires the frozen T1 split at "
                f"{split_path}"
            )
        split_record = json.loads(split_path.read_text(encoding="utf-8"))
        split_map = split_record["splits"]
        if {name: len(writers) for name, writers in split_map.items()} != T1SingleSplitBuilder.SPLIT_SIZES:
            raise ValueError("T2 requires the exact frozen T1 55/12/12 writer split")

        counts: dict[str, int] = {}
        for split, writers in split_map.items():
            rows = self._build_split(split, writers)
            name = f"{split}_t2.jsonl"
            write_jsonl(self.root / "episodes/fold_0" / name, rows)
            counts[name] = len(rows)

        summary = {
            "schema_version": "1.1.0-t2-single-split",
            "task": "T2 conditional source attribution",
            "seed": self.seed,
            "writer_split": {split: len(writers) for split, writers in split_map.items()},
            "writer_split_digest": split_record["split_digest"],
            "candidate_count": self.candidate_count,
            "candidate_state": "NW only",
            "episode_counts": counts,
            "total_episodes": sum(counts.values()),
            "per_writer": {**self.EXPECTED_PER_WRITER, "total": 192},
            "episode_ratio": "source_present:source_absent:rf_no_source=2:1:1",
            "final_label_ratio": "candidate_position:Unknown=1:1",
            "present_policy": (
                f"recorded source NW plus {self.candidate_count - 1} independently sampled NW decoys"
            ),
            "absent_policy": (
                f"{self.candidate_count} independently sampled NW candidates excluding the recorded source"
            ),
            "rf_policy": f"{self.candidate_count} independently sampled NW candidates",
            "pool_identity": "query_id plus unordered candidate membership; order-only changes are not episodes",
            "statistical_unit": "writer; repeated pools for a query are clustered by query_group_id",
        }
        atomic_json(self.root / "t2_benchmark_manifest.json", summary)
        return summary

    def _build_split(self, split: str, writers: Sequence[str]) -> list[dict[str, Any]]:
        split_offset = {"train": 1103, "val": 1709, "test": 2309}[split]
        rng = random.Random(self.seed + split_offset)
        rows: list[dict[str, Any]] = []
        for writer in writers:
            nw = sorted(
                (sample for sample in self.by_writer[writer] if sample["state"] == "NW"),
                key=_sample_index,
            )
            sf = sorted(
                (sample for sample in self.by_writer[writer] if sample["state"] == "SF"),
                key=lambda sample: sample["sample_id"],
            )
            rf = sorted(
                (sample for sample in self.by_writer[writer] if sample["state"] == "RF"),
                key=lambda sample: sample["sample_id"],
            )
            if (len(nw), len(sf), len(rf)) != (20, 8, 12):
                raise ValueError(
                    f"Writer {writer} requires 20 NW, 8 SF, and 12 RF; "
                    f"found {(len(nw), len(sf), len(rf))}"
                )
            nw_by_index = {_sample_index(sample): sample for sample in nw}
            present_index = 0
            position_offset = 0
            if self.EXPECTED_PER_WRITER["source_present"] % self.candidate_count:
                digest = hashlib.sha256(
                    f"{self.seed}:{split}:{writer}:source-position".encode("utf-8")
                ).digest()
                position_offset = int.from_bytes(digest[:8], "big") % self.candidate_count
            for query in sf:
                source = nw_by_index[int(query["reference_nw_index"])]
                decoys = [sample for sample in nw if sample["sample_id"] != source["sample_id"]]
                present_seen: set[tuple[str, ...]] = set()
                absent_seen: set[tuple[str, ...]] = set()
                for repeat_index in range(self.PRESENT_PER_SF):
                    selected = self._unique_sample(
                        decoys, self.candidate_count - 1, present_seen, rng,
                    )
                    source_position = (present_index + position_offset) % self.candidate_count
                    candidates = list(selected)
                    rng.shuffle(candidates)
                    candidates.insert(source_position, source)
                    rows.append(self._row(
                        split, query, candidates, source_position, "source_present",
                        writer, repeat_index,
                    ))
                    present_index += 1
                for repeat_index in range(self.ABSENT_PER_SF):
                    candidates = self._unique_sample(
                        decoys, self.candidate_count, absent_seen, rng,
                    )
                    rng.shuffle(candidates)
                    rows.append(self._row(
                        split, query, candidates, -1, "source_absent", writer, repeat_index,
                    ))
            for query in rf:
                seen: set[tuple[str, ...]] = set()
                for repeat_index in range(self.RF_PER_QUERY):
                    candidates = self._unique_sample(nw, self.candidate_count, seen, rng)
                    rng.shuffle(candidates)
                    rows.append(self._row(
                        split, query, candidates, -1, "rf_no_source", writer, repeat_index,
                    ))

        rng.shuffle(rows)
        for index, row in enumerate(rows):
            row["episode_id"] = f"S0-{split.upper()}-T2-L{self.candidate_count}-{index:06d}"
        return rows

    def _row(
        self, split: str, query: dict[str, Any], candidates: list[dict[str, Any]],
        target_index: int, episode_type: str, writer: str, repeat_index: int,
    ) -> dict[str, Any]:
        candidate_ids = [sample["sample_id"] for sample in candidates]
        return {
            "episode_id": "",
            "fold_id": 0,
            "split": split,
            "protocol": "t2_source_ranking",
            "query_id": query["sample_id"],
            "candidate_ids": candidate_ids,
            "target_index": target_index,
            "target_type": "candidate" if target_index >= 0 else "unknown",
            "episode_type": episode_type,
            "candidate_count": self.candidate_count,
            "target_writer_id": writer,
            "forger_id": query.get("forger_id"),
            "reference_nw_index": query.get("reference_nw_index"),
            "collection_batch": query.get("collection_batch"),
            "candidate_collection_batches": [sample.get("collection_batch") for sample in candidates],
            "sampling_seed": self.seed,
            "query_group_id": f"{split}:t2:{writer}:{query['sample_id']}",
            "repeated_query_group": query["sample_id"],
            "statistical_unit": writer,
            "repeat_index": repeat_index,
            "candidate_pool_hash": stable_hash(sorted(candidate_ids)),
        }


def audit_t2_single_benchmark(
    dataset_root: str | Path,
    benchmark_root: str | Path,
    full_hash: bool = False,
    candidate_count: int | None = None,
) -> dict[str, Any]:
    store = SignatureStore(dataset_root, verify_hashes=full_hash)
    root = Path(benchmark_root)
    manifest_path = root / "t2_benchmark_manifest.json"
    if candidate_count is None:
        candidate_count = (
            int(json.loads(manifest_path.read_text(encoding="utf-8"))["candidate_count"])
            if manifest_path.is_file() else T2SingleSplitBuilder.DEFAULT_CANDIDATE_COUNT
        )
    if not 2 <= candidate_count <= 19:
        raise ValueError("T2 candidate_count must be between 2 and 19")
    split_record = json.loads((root / "folds/single_split.json").read_text(encoding="utf-8"))
    splits = {name: set(writers) for name, writers in split_record["splits"].items()}
    nw_source_lookup = {
        (sample["writer_id"], _sample_index(sample)): sample["sample_id"]
        for sample in store.samples.values() if sample["state"] == "NW"
    }
    errors: list[str] = []
    if {name: len(writers) for name, writers in splits.items()} != T1SingleSplitBuilder.SPLIT_SIZES:
        errors.append("writer split counts are not the frozen T1 55/12/12 split")
    if any(splits[left] & splits[right] for left, right in (
        ("train", "val"), ("train", "test"), ("val", "test"),
    )):
        errors.append("writer leakage between train/val/test")

    checked: dict[str, int] = {}
    file_stats: dict[str, Any] = {}
    expected = T2SingleSplitBuilder.EXPECTED_PER_WRITER
    for split in ("train", "val", "test"):
        path = root / "episodes/fold_0" / f"{split}_t2.jsonl"
        rows = read_jsonl(path)
        checked[path.name] = len(rows)
        episode_ids: set[str] = set()
        query_pools: set[tuple[str, str]] = set()
        per_writer: dict[str, Counter[str]] = defaultdict(Counter)
        position_by_writer: dict[str, Counter[int]] = defaultdict(Counter)
        query_types: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            episode_id = row["episode_id"]
            if episode_id in episode_ids:
                errors.append(f"{path.name}: duplicate episode_id {episode_id}")
            episode_ids.add(episode_id)
            writer = row["target_writer_id"]
            episode_type = row["episode_type"]
            per_writer[writer][episode_type] += 1
            query_types[row["query_id"]][episode_type] += 1
            if writer not in splits[split]:
                errors.append(f"{episode_id}: target writer outside {split}")
            query = store.get(row["query_id"])
            candidates = [store.get(sample_id) for sample_id in row["candidate_ids"]]
            if (
                len(candidates) != candidate_count
                or len({sample["sample_id"] for sample in candidates}) != candidate_count
            ):
                errors.append(
                    f"{episode_id}: candidate pool is not {candidate_count} unique IDs"
                )
            if int(row.get("candidate_count", -1)) != candidate_count:
                errors.append(f"{episode_id}: candidate_count field mismatch")
            if len({sample["sha256"] for sample in candidates}) != len(candidates):
                errors.append(f"{episode_id}: repeated candidate content")
            if any(sample["writer_id"] != writer or sample["state"] != "NW" for sample in candidates):
                errors.append(f"{episode_id}: candidates are not target-writer NW")
            if query["writer_id"] != writer:
                errors.append(f"{episode_id}: query writer mismatch")
            expected_query_state = "RF" if episode_type == "rf_no_source" else "SF"
            if query["state"] != expected_query_state:
                errors.append(f"{episode_id}: query state mismatch")
            expected_hash = stable_hash(sorted(row["candidate_ids"]))
            if row.get("candidate_pool_hash") != expected_hash:
                errors.append(f"{episode_id}: invalid candidate_pool_hash")
            pool_key = (row["query_id"], expected_hash)
            if pool_key in query_pools:
                errors.append(f"{episode_id}: duplicate unordered candidate pool for query")
            query_pools.add(pool_key)

            source_id = None
            if episode_type != "rf_no_source":
                source_index = int(query["reference_nw_index"])
                source_id = nw_source_lookup[(writer, source_index)]
            if episode_type == "source_present":
                target = row["target_index"]
                if target not in range(candidate_count) or row["candidate_ids"][target] != source_id:
                    errors.append(f"{episode_id}: incorrect present source target")
                else:
                    position_by_writer[writer][target] += 1
            elif row["target_index"] != -1:
                errors.append(f"{episode_id}: unknown episode has a candidate target")
            if episode_type == "source_absent" and source_id in row["candidate_ids"]:
                errors.append(f"{episode_id}: source leaked into absent pool")

        for writer in splits[split]:
            if dict(per_writer[writer]) != expected:
                errors.append(f"{path.name}: {writer} episode distribution mismatch")
            position_counts = position_by_writer[writer]
            if (
                set(position_counts) != set(range(candidate_count))
                or sum(position_counts.values()) != expected["source_present"]
                or max(position_counts.values()) - min(position_counts.values()) > 1
            ):
                errors.append(f"{path.name}: {writer} source positions are not near-balanced")
        for query_id, counts in query_types.items():
            state = store.get(query_id)["state"]
            expected_query_counts = (
                {"source_present": 12, "source_absent": 6}
                if state == "SF" else {"rf_no_source": 4}
            )
            if dict(counts) != expected_query_counts:
                errors.append(f"{path.name}: {query_id} repeat distribution mismatch")
        type_counts = Counter(row["episode_type"] for row in rows)
        file_stats[path.name] = {
            "episodes": len(rows),
            "writers": len(per_writer),
            "episode_types": dict(type_counts),
            "unknown": type_counts["source_absent"] + type_counts["rf_no_source"],
            "candidate_targets": type_counts["source_present"],
            "unique_query_pool_memberships": len(query_pools),
        }

    report = {
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors[:100],
        "writer_split": {name: len(writers) for name, writers in splits.items()},
        "split_digest": split_record["split_digest"],
        "candidate_count": candidate_count,
        "episodes_checked": checked,
        "files": file_stats,
    }
    atomic_json(root / "t2_audit_report.json", report)
    if errors:
        raise ValueError(f"T2 benchmark audit failed with {len(errors)} errors; see t2_audit_report.json")
    return report


def t2_balanced_a_query_view(
    episodes: Sequence[dict[str, Any]], seed: int, balance: bool,
) -> list[dict[str, Any]]:
    """Derive one A-only row per unique Query without candidate-pool repetition."""
    first_by_query: dict[str, dict[str, Any]] = {}
    for row in episodes:
        first_by_query.setdefault(row["query_id"], row)
    rows = []
    for query_id, source in first_by_query.items():
        is_rf = source["episode_type"] == "rf_no_source"
        rows.append({
            "episode_id": f"{source['split'].upper()}-T2-A-{query_id}",
            "fold_id": source.get("fold_id", 0),
            "split": source["split"],
            "protocol": "t2_a_query",
            "query_id": query_id,
            "rf_label": int(is_rf),
            "episode_type": "rf_query" if is_rf else "sf_query",
            "target_writer_id": source["target_writer_id"],
            "forger_id": source.get("forger_id"),
            "collection_batch": source.get("collection_batch"),
            "query_group_id": source.get("query_group_id", query_id),
            "statistical_unit": source.get("statistical_unit", source["target_writer_id"]),
        })
    if not balance:
        return sorted(rows, key=lambda row: row["episode_id"])

    balanced: list[dict[str, Any]] = []
    by_writer: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_writer[str(row["target_writer_id"])][int(row["rf_label"])].append(row)
    for writer, labels in sorted(by_writer.items()):
        if not labels[0] or not labels[1]:
            raise ValueError(f"Balanced T2-A view requires SF and RF queries for writer {writer}")
        keep = min(len(labels[0]), len(labels[1]))
        for label in (0, 1):
            ranked = sorted(
                labels[label],
                key=lambda row: hashlib.sha256(
                    f"{seed}:{writer}:{label}:{row['query_id']}".encode("utf-8")
                ).digest(),
            )
            balanced.extend(ranked[:keep])
    return sorted(balanced, key=lambda row: row["episode_id"])


def t2_balanced_b_view(
    episodes: Sequence[dict[str, Any]], seed: int,
) -> list[dict[str, Any]]:
    """Balance Present/Absent within every SF Query for factor-B evaluation."""
    by_query: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in episodes:
        if row["episode_type"] in {"source_present", "source_absent"}:
            by_query[row["query_id"]][row["episode_type"]].append(row)
    selected: list[dict[str, Any]] = []
    for query_id, groups in sorted(by_query.items()):
        present = groups["source_present"]
        absent = groups["source_absent"]
        if not present or not absent:
            raise ValueError(f"Balanced T2-B view requires Present and Absent pools for {query_id}")
        keep = min(len(present), len(absent))
        for name, rows in (("source_present", present), ("source_absent", absent)):
            ranked = sorted(
                rows,
                key=lambda row: hashlib.sha256(
                    f"{seed}:{query_id}:{name}:{row['episode_id']}".encode("utf-8")
                ).digest(),
            )
            selected.extend(ranked[:keep])
    return sorted(selected, key=lambda row: row["episode_id"])


class EpisodeDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.episodes = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.episodes[index]


class EpisodeCollator:
    """Deduplicate samples inside a batch, then pad sequences and episode sets."""

    def __init__(self, preprocessor: SignaturePreprocessor):
        self.preprocessor = preprocessor

    def __call__(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        sample_ids: list[str] = []
        for episode in episodes:
            sample_ids.extend(episode.get("reference_ids", episode.get("candidate_ids", [])))
            sample_ids.append(episode["query_id"])
        unique_ids = list(dict.fromkeys(sample_ids))
        lookup = {sample_id: index for index, sample_id in enumerate(unique_ids)}
        prepared = [self.preprocessor.prepare(sample_id) for sample_id in unique_ids]
        max_length = max(len(item.sequence) for item in prepared)
        max_anchor = max(len(item.anchors) for item in prepared)
        feature_dim = prepared[0].sequence.shape[-1]
        sequences = torch.zeros(len(prepared), max_length, feature_dim)
        sequence_mask = torch.zeros(len(prepared), max_length, dtype=torch.bool)
        anchors = torch.zeros(len(prepared), max_anchor, 6)
        anchor_mask = torch.zeros(len(prepared), max_anchor, dtype=torch.bool)
        for index, item in enumerate(prepared):
            sequences[index, :len(item.sequence)] = item.sequence
            sequence_mask[index, :len(item.sequence)] = True
            anchors[index, :len(item.anchors)] = item.anchors
            anchor_mask[index, :len(item.anchor_mask)] = item.anchor_mask
        images = torch.stack([item.image for item in prepared])
        query_index = torch.tensor([lookup[e["query_id"]] for e in episodes], dtype=torch.long)
        set_ids = [e.get("reference_ids", e.get("candidate_ids", [])) for e in episodes]
        max_set = max(map(len, set_ids))
        set_index = torch.zeros(len(episodes), max_set, dtype=torch.long)
        set_mask = torch.zeros(len(episodes), max_set, dtype=torch.bool)
        for row, ids in enumerate(set_ids):
            set_index[row, :len(ids)] = torch.tensor([lookup[x] for x in ids])
            set_mask[row, :len(ids)] = True
        batch: dict[str, Any] = {
            "sequence": sequences, "sequence_mask": sequence_mask, "image": images,
            "anchors": anchors, "anchor_mask": anchor_mask, "query_index": query_index,
            "set_index": set_index, "set_mask": set_mask, "episode_id": [e["episode_id"] for e in episodes],
            "metadata": episodes, "sample_ids": unique_ids,
        }
        protocol = episodes[0]["protocol"]
        if any(e["protocol"] != protocol for e in episodes):
            raise ValueError("A batch cannot mix protocols")
        batch["protocol"] = protocol
        if protocol.startswith("t1"):
            batch["label"] = torch.tensor([e["label"] for e in episodes], dtype=torch.float32)
        elif protocol == "t2_a_query":
            batch["rf_label"] = torch.tensor(
                [e["rf_label"] for e in episodes], dtype=torch.float32,
            )
        else:
            batch["target_index"] = torch.tensor([e["target_index"] for e in episodes], dtype=torch.long)
            batch["exist_label"] = (batch["target_index"] >= 0).float()
            episode_type_index = {
                "source_present": 0,
                "source_absent": 1,
                "rf_no_source": 2,
            }
            batch["episode_type_index"] = torch.tensor(
                [episode_type_index[e["episode_type"]] for e in episodes], dtype=torch.long,
            )
        return batch


def make_preprocessor(config: DataConfig) -> tuple[SignatureStore, SignaturePreprocessor]:
    store = SignatureStore(config.dataset_root, verify_hashes=config.verify_hashes)
    root = Path(config.benchmark_root)
    legacy_sequence = config.input_pipeline in {"legacy_v1", "rgb_legacy_ablation"}
    rgb_image = config.input_pipeline in {"rgb_legacy_ablation", "raw_rgb_v2"}
    if config.input_pipeline not in {
        "legacy_v1", "rgb_legacy_ablation", "gray_raw_ablation", "raw_rgb_v2",
    }:
        raise ValueError(f"Unsupported input pipeline: {config.input_pipeline}")
    if legacy_sequence:
        stats = json.loads((root / f"folds/fold_{config.fold}_normalization.json").read_text(encoding="utf-8"))
    else:
        stats = None
    if rgb_image:
        render = DynamicRGBRenderConfig(
            width=config.image_width, height=config.image_height,
            margin=config.image_margin, line_width=config.image_line_width,
            supersample=config.image_supersample,
            speed_cap_mm_s=config.image_speed_cap_mm_s,
        )
    else:
        render = RenderConfig(
            width=config.image_width, height=config.image_height,
            margin=config.image_margin, line_width=config.image_line_width,
            supersample=config.image_supersample,
        )
    cache_root = Path(config.image_cache_root) if config.image_cache_root else root / "cache/rendered_png"
    preprocessor = SignaturePreprocessor(
        store, stats, cache_root, render,
        max_points=config.max_points, cache_images=config.cache_images, memory_cache=config.memory_cache,
        memory_cache_items=config.memory_cache_items, input_pipeline=config.input_pipeline,
        raw_time_scale_seconds=config.raw_time_scale_seconds,
        raw_position_scale_mm=config.raw_position_scale_mm,
        zero_channels=config.zero_channels, time_align=config.time_align,
        time_align_points=config.time_align_points,
        time_align_constant_pressure=config.time_align_constant_pressure,
        time_align_duration_ms=config.time_align_duration_ms,
    )
    return store, preprocessor


def audit_benchmark(dataset_root: str | Path, benchmark_root: str | Path, full_hash: bool = False) -> dict[str, Any]:
    store = SignatureStore(dataset_root, verify_hashes=full_hash)
    root = Path(benchmark_root)
    rotations = json.loads((root / "folds/fold_rotation.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    checked = Counter()
    for fold_id, split_map in rotations.items():
        sets = {split: set(writers) for split, writers in split_map.items()}
        if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
            errors.append(f"fold {fold_id}: writer leakage")
        stats = json.loads((root / f"folds/fold_{fold_id}_normalization.json").read_text(encoding="utf-8"))
        if set(stats["writer_ids"]) != sets["train"]:
            errors.append(f"fold {fold_id}: normalization writer leakage")
        for path in sorted((root / f"episodes/fold_{fold_id}").glob("*.jsonl")):
            split = path.name.split("_", 1)[0]
            ids_seen: set[str] = set()
            for row in read_jsonl(path):
                checked[path.name] += 1
                if row["episode_id"] in ids_seen:
                    errors.append(f"{path}: duplicate episode_id {row['episode_id']}")
                ids_seen.add(row["episode_id"])
                if row["target_writer_id"] not in sets[split]:
                    errors.append(f"{row['episode_id']}: target writer outside split")
                members = row.get("reference_ids", row.get("candidate_ids", []))
                if len(members) != len(set(members)):
                    errors.append(f"{row['episode_id']}: repeated sample in set")
                hashes = [store.get(s)["sha256"] for s in members]
                if len(hashes) != len(set(hashes)):
                    errors.append(f"{row['episode_id']}: repeated content in set")
                member_rows = [store.get(sample_id) for sample_id in members]
                if any(sample["writer_id"] != row["target_writer_id"] for sample in member_rows):
                    errors.append(f"{row['episode_id']}: set member writer mismatch")
                if row["protocol"].startswith("t1"):
                    query = store.get(row["query_id"])
                    if any(sample["label"] != "genuine" for sample in member_rows):
                        errors.append(f"{row['episode_id']}: non-genuine T1 reference")
                    if query["sha256"] in hashes:
                        errors.append(f"{row['episode_id']}: query content appears in references")
                    if row["label"] == 1 and (
                        query["writer_id"] != row["target_writer_id"] or query["label"] != "genuine"
                    ):
                        errors.append(f"{row['episode_id']}: invalid positive query")
                    if row["attack_type"] == "zero_effort" and query["writer_id"] == row["target_writer_id"]:
                        errors.append(f"{row['episode_id']}: zero-effort query has target writer")
                else:
                    if any(sample["state"] != "NW" for sample in member_rows):
                        errors.append(f"{row['episode_id']}: non-NW T2 candidate")
                    source_index = row.get("reference_nw_index")
                    source_id = None
                    if source_index is not None:
                        source_id = next(
                            s["sample_id"] for s in store.samples.values()
                            if s["writer_id"] == row["target_writer_id"] and s["state"] == "NW" and _sample_index(s) == source_index
                        )
                    if row["episode_type"] == "source_present":
                        if row["target_index"] < 0 or members[row["target_index"]] != source_id:
                            errors.append(f"{row['episode_id']}: incorrect source target")
                    elif source_id is not None and source_id in members:
                        errors.append(f"{row['episode_id']}: source leaked into absent pool")
    report = {"ok": not errors, "errors": errors[:100], "error_count": len(errors), "episodes_checked": dict(checked)}
    atomic_json(root / "audit_report.json", report)
    if errors:
        raise ValueError(f"Benchmark audit failed with {len(errors)} errors; see audit_report.json")
    return report
