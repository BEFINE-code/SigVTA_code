from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from repro_baselines.data import BenchmarkRepository, SignatureStore


def normalize_official(path: np.ndarray) -> np.ndarray:
    result = np.asarray(path, dtype=np.float64).copy()
    minimum = result[:, :2].min(axis=0)
    maximum = result[:, :2].max(axis=0)
    result[:, :2] = (result[:, :2] - (maximum + minimum) / 2.0) / max(
        float((maximum - minimum).max()), 1e-8,
    )
    result[:, 2] /= max(float(result[:, 2].max()), 1e-8)
    return result


def snr_metrics(
    parameters: np.ndarray, timestamp: np.ndarray, path: np.ndarray,
    speed_x: np.ndarray, speed_y: np.ndarray, reconstruct,
) -> tuple[float, float, float]:
    recon_x, recon_y, recon_speed_x, recon_speed_y = reconstruct(parameters, timestamp)
    velocity = np.hypot(speed_x, speed_y)
    recon_velocity = np.hypot(recon_speed_x, recon_speed_y)
    centered_path = path - path.mean(axis=0)
    recon_x = recon_x - recon_x.mean()
    recon_y = recon_y - recon_y.mean()
    signal_velocity = np.square(speed_x).sum() + np.square(speed_y).sum()
    noise_velocity = np.square(recon_speed_x - speed_x).sum() + np.square(
        recon_speed_y - speed_y,
    ).sum()
    snr_velocity_xy = 10.0 * np.log10(signal_velocity / (noise_velocity + 1e-8))
    snr_velocity = 10.0 * np.log10(
        np.square(velocity).sum() / (np.square(recon_velocity - velocity).sum() + 1e-8),
    )
    signal_path = np.square(centered_path[:, 0]).sum() + np.square(centered_path[:, 1]).sum()
    noise_path = np.square(recon_x - centered_path[:, 0]).sum() + np.square(
        recon_y - centered_path[:, 1],
    ).sum()
    snr_path = 10.0 * np.log10(signal_path / (noise_path + 1e-8))
    return float(snr_velocity_xy), float(snr_velocity), float(snr_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--profile-optimization", action="store_true")
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    sys.path.insert(0, str(official_root / "sigma_lognormal"))
    import scipy
    from scipy import signal

    if not hasattr(scipy, "convolve"):
        scipy.convolve = signal.convolve  # type: ignore[attr-defined]
    from slbox import functions  # type: ignore[import-not-found]
    from slbox import optimization  # type: ignore[import-not-found]
    from slbox.extractionFirstMode import (  # type: ignore[import-not-found]
        paramExtraction, paramReconstruction,
    )

    repo = BenchmarkRepository(args.root)
    store = SignatureStore(repo.dataset)
    train_writers = repo.writers("train")
    sample_ids = sorted(
        sample_id for sample_id, sample in store.samples.items()
        if sample["writer_id"] in train_writers and sample["label"] == "genuine"
    )[: args.count]
    total_started = time.perf_counter()
    for sample_id in sample_ids:
        raw = store.load(sample_id)
        path = normalize_official(raw[:, [1, 2, 3]])
        path = functions.cubicSplineInterp(path, nfs=2)
        timestamp = np.arange(len(path), dtype=np.float64) * 0.005
        started = time.perf_counter()
        parameters, auc, residual, _, target_path, speed_x, speed_y = paramExtraction(
            timestamp, path, lAmbda=15.0, smoothing=True, zeroInit=False,
            saveParam=False, dt=0.005, seqMode=False, localOptim=False,
        )
        snr_velocity_xy, snr_velocity, snr_path = snr_metrics(
            parameters, timestamp, target_path, speed_x, speed_y, paramReconstruction,
        )
        row = {
            "sample_id": sample_id,
            "raw_points": len(raw),
            "interpolated_points": len(path),
            "lognormal_components": len(parameters),
            "residual_points": len(residual),
            "auc_components": len(auc),
            "snr_velocity_xy": snr_velocity_xy,
            "snr_velocity": snr_velocity,
            "snr_path": snr_path,
            "triggers_second_pass": snr_velocity_xy < 15.0 or snr_path < 12.0,
            "triggers_global_optimization": snr_velocity_xy < 18.0 or snr_path < 18.0,
            "seconds": time.perf_counter() - started,
        }
        if args.profile_optimization and row["triggers_global_optimization"]:
            optimization_started = time.perf_counter()
            optimized = optimization.globalOptim(
                timestamp, target_path, np.column_stack([speed_x, speed_y]),
                parameters, dt=0.005,
            )
            row["optimization_seconds"] = time.perf_counter() - optimization_started
            row["optimized_components"] = len(optimized)
        print(row, flush=True)
    print({"count": len(sample_ids), "total_seconds": time.perf_counter() - total_started})


if __name__ == "__main__":
    main()
