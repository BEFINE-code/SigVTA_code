from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from repro_baselines.data import BenchmarkRepository, SignatureStore


def normalize_official(raw: np.ndarray) -> np.ndarray:
    result = np.asarray(raw[:, [1, 2, 3]], dtype=np.float64).copy()
    minimum = result[:, :2].min(axis=0)
    maximum = result[:, :2].max(axis=0)
    result[:, :2] = (result[:, :2] - (maximum + minimum) / 2.0) / max(
        float((maximum - minimum).max()), 1e-8,
    )
    result[:, 2] /= max(float(result[:, 2].max()), 1e-8)
    return result.astype(np.float32)


def _load_official(official_root: str) -> dict[str, Any]:
    import scipy
    from scipy import signal

    if not hasattr(scipy, "convolve"):
        scipy.convolve = signal.convolve  # type: ignore[attr-defined]
    sys.path.insert(0, str(Path(official_root) / "sigma_lognormal"))
    from slbox import functions, optimization  # type: ignore[import-not-found]
    from slbox.extractionFirstMode import (  # type: ignore[import-not-found]
        pad, paramExtraction, paramReconstruction, pathOps,
    )
    return {
        "functions": functions,
        "optimization": optimization,
        "pad": pad,
        "paramExtraction": paramExtraction,
        "paramReconstruction": paramReconstruction,
        "pathOps": pathOps,
    }


def _snr(
    parameters: np.ndarray, timestamp: np.ndarray, path: np.ndarray,
    speed_x: np.ndarray, speed_y: np.ndarray, reconstruct,
) -> tuple[float, float, float]:
    recon_x, recon_y, recon_speed_x, recon_speed_y = reconstruct(parameters, timestamp)
    velocity = np.hypot(speed_x, speed_y)
    recon_velocity = np.hypot(recon_speed_x, recon_speed_y)
    centered_path = path - path.mean(axis=0)
    recon_x = recon_x - recon_x.mean()
    recon_y = recon_y - recon_y.mean()
    signal_xy = np.square(speed_x).sum() + np.square(speed_y).sum()
    noise_xy = np.square(recon_speed_x - speed_x).sum() + np.square(
        recon_speed_y - speed_y,
    ).sum()
    snr_xy = 10.0 * np.log10(signal_xy / (noise_xy + 1e-8))
    snr_velocity = 10.0 * np.log10(
        np.square(velocity).sum() / (np.square(recon_velocity - velocity).sum() + 1e-8),
    )
    signal_path = np.square(centered_path[:, :2]).sum()
    noise_path = np.square(recon_x - centered_path[:, 0]).sum() + np.square(
        recon_y - centered_path[:, 1],
    ).sum()
    return float(snr_xy), float(snr_velocity), float(
        10.0 * np.log10(signal_path / (noise_path + 1e-8)),
    )


def _residual(parameters: np.ndarray, timestamp: np.ndarray, path: np.ndarray, official) -> np.ndarray:
    path_object = official["pathOps"](
        timestamp, path, smoothing=False, pad=official["pad"], zeroInit=False,
    )
    for parameter in parameters:
        path_object.subtractStroke(parameter)
    return path_object.finalPath().copy()


def _extract(job: tuple[str, int, str, np.ndarray, str, str]) -> dict[str, Any]:
    writer, index, sample_id, path, params_root, official_root = job
    output_dir = Path(params_root) / str(int(writer))
    parameter_path = output_dir / f"Pmatrix_G{int(writer)}_{index}.npy"
    residual_path = output_dir / f"residual_G{int(writer)}_{index}.npy"
    if parameter_path.is_file() and residual_path.is_file():
        return {"sample_id": sample_id, "status": "cached"}
    official = _load_official(official_root)
    started = time.perf_counter()
    interpolated = official["functions"].cubicSplineInterp(path, nfs=2)
    timestamp = np.arange(len(interpolated), dtype=np.float64) * 0.005
    parameters, auc, first_residual, vmax, target, speed_x, speed_y = official[
        "paramExtraction"
    ](
        timestamp, interpolated, lAmbda=15.0, smoothing=True, zeroInit=False,
        saveParam=False, dt=0.005, seqMode=False, localOptim=False,
    )
    snr_xy, snr_velocity, snr_path = _snr(
        parameters, timestamp, target, speed_x, speed_y, official["paramReconstruction"],
    )
    second_pass = snr_xy < 15.0 or snr_path < 12.0
    if second_pass:
        extra, extra_auc, _, _, _, _, _ = official["paramExtraction"](
            timestamp, first_residual, lAmbda=25.0, smoothing=False, zeroInit=True,
            saveParam=False, dt=0.005, vm=vmax, seqMode=True,
        )
        parameters = np.concatenate([parameters, extra], axis=0)
        auc = np.concatenate([auc, extra_auc], axis=0)
        order = np.argsort(parameters[:, 1])
        parameters = parameters[order]
        auc = auc[order]
        snr_xy, snr_velocity, snr_path = _snr(
            parameters, timestamp, target, speed_x, speed_y,
            official["paramReconstruction"],
        )
    optimized = snr_xy < 18.0 or snr_path < 18.0
    if optimized:
        candidate = official["optimization"].globalOptim(
            timestamp, target, np.column_stack([speed_x, speed_y]), parameters, dt=0.005,
        )
        candidate_snr = _snr(
            candidate, timestamp, target, speed_x, speed_y,
            official["paramReconstruction"],
        )
        if candidate_snr[2] >= snr_path:
            parameters = candidate[np.argsort(candidate[:, 1])]
            snr_xy, snr_velocity, snr_path = candidate_snr
    residual = _residual(parameters, timestamp, interpolated, official)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(parameter_path, parameters)
    np.save(residual_path, residual)
    return {
        "sample_id": sample_id,
        "status": "extracted",
        "components": len(parameters),
        "snr_velocity_xy": snr_xy,
        "snr_velocity": snr_velocity,
        "snr_path": snr_path,
        "second_pass": second_pass,
        "optimized": optimized,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    official_root = Path(args.official_root).resolve()
    output = Path(args.output).resolve()
    params_root = output / "params"
    output.mkdir(parents=True, exist_ok=True)
    repo = BenchmarkRepository(root)
    store = SignatureStore(repo.dataset)
    train_writers = repo.writers("train")
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id, sample in store.samples.items():
        if sample["writer_id"] in train_writers and sample["label"] == "genuine":
            grouped[sample["writer_id"]].append(sample_id)
    ordered = {
        writer: sorted(sample_ids) for writer, sample_ids in sorted(grouped.items())
    }
    records: list[dict[str, Any]] = []
    signature_dict: dict[int, dict[bool, list[np.ndarray]]] = {}
    jobs: list[tuple[str, int, str, np.ndarray, str, str]] = []
    for writer, sample_ids in ordered.items():
        paths = [normalize_official(store.load(sample_id)) for sample_id in sample_ids]
        signature_dict[int(writer)] = {True: paths, False: []}
        for index, (sample_id, path) in enumerate(zip(sample_ids, paths, strict=True)):
            records.append({"writer_id": writer, "index": index, "sample_id": sample_id})
            jobs.append((writer, index, sample_id, path, str(params_root), str(official_root)))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    started = time.perf_counter()
    extracted: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(_extract, jobs, chunksize=1), start=1):
            extracted.append(result)
            if index % 100 == 0 or index == len(jobs):
                print(f"SynSig2Vec parameters {index}/{len(jobs)}", flush=True)
    complete = len(jobs) == len(records)
    if complete:
        with (output / "train_genuine.pkl").open("wb") as stream:
            pickle.dump(signature_dict, stream, protocol=4)
    commit = subprocess.check_output(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    manifest = {
        "official_repository": "https://github.com/LaiSongxuan/SynSig2Vec",
        "official_commit": commit,
        "official_license": "GPL-3.0",
        "split_digest": repo.verify()["split_digest"],
        "writers": len(ordered),
        "genuine_signatures": len(records),
        "processed_jobs": len(jobs),
        "complete": complete,
        "workers": args.workers,
        "seconds": time.perf_counter() - started,
        "records": records,
        "extraction_summary": {
            "cached": sum(row["status"] == "cached" for row in extracted),
            "extracted": sum(row["status"] == "extracted" for row in extracted),
            "second_pass": sum(row.get("second_pass", False) for row in extracted),
            "optimized": sum(row.get("optimized", False) for row in extracted),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
