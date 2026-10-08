"""P0 duration-shortcut diagnostics for T1 skilled-forgery FAR=0."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "dvsrc").is_dir())
sys.path.insert(0, str(ROOT))

from dvsrc.metrics import binary_summary, t1_metrics
from dvsrc.utils import atomic_json, read_jsonl


PROTOCOLS = ("t1_1v1", "t1_5v1")
DEFAULT_S2F0 = ROOT / "ablation/t1/results_seed42/runs/S2F0/seed_42"


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {sample["sample_id"]: sample for sample in payload["samples"]}


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, q))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def far_with_ci(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    negatives = [row for row in rows if row["label"] == 0]
    n = len(negatives)
    accepted = sum(row["score"] >= threshold for row in negatives)
    low, high = wilson_interval(accepted, n)
    return {
        "n": n,
        "accepted": accepted,
        "far": (accepted / n) if n else float("nan"),
        "far_ci_low": low,
        "far_ci_high": high,
    }


def writer_genuine_stats(samples: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_writer: dict[str, list[float]] = defaultdict(list)
    for sample in samples.values():
        if sample["label"] == "genuine":
            by_writer[sample["writer_id"]].append(float(sample["duration_ms"]))
    stats = {}
    for writer, durations in by_writer.items():
        values = np.asarray(durations, dtype=np.float64)
        stats[writer] = {
            "n": int(len(values)),
            "median": float(np.median(values)),
            "q25": percentile(values, 0.25),
            "q75": percentile(values, 0.75),
        }
    return stats


def duration_table(samples: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_state: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for sample in samples.values():
        by_state[sample["state"]].append((float(sample["duration_ms"]), int(sample["row_count"])))
    table = {}
    for state, rows in sorted(by_state.items()):
        durations = np.asarray([row[0] for row in rows], dtype=np.float64)
        points = np.asarray([row[1] for row in rows], dtype=np.float64)
        table[state] = {
            "n": int(len(rows)),
            "duration_mean": float(durations.mean()),
            "duration_median": float(np.median(durations)),
            "duration_p10": percentile(durations, 0.10),
            "duration_p90": percentile(durations, 0.90),
            "points_mean": float(points.mean()),
            "points_median": float(np.median(points)),
        }
    ratios = []
    genuine_by_writer = writer_genuine_stats(samples)
    for sample in samples.values():
        if sample["state"] != "SF":
            continue
        median = genuine_by_writer[sample["writer_id"]]["median"]
        if median > 0:
            ratios.append(float(sample["duration_ms"]) / median)
    table["SF_over_writer_genuine_median"] = {
        "n": int(len(ratios)),
        "mean": float(np.mean(ratios)) if ratios else float("nan"),
        "median": float(np.median(ratios)) if ratios else float("nan"),
    }
    return table


def attach_duration(episodes: Iterable[dict[str, Any]], samples: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    genuine = writer_genuine_stats(samples)
    rows = []
    for episode in episodes:
        query = samples[episode["query_id"]]
        refs = [samples[sample_id] for sample_id in episode["reference_ids"]]
        writer = episode["target_writer_id"]
        query_duration = float(query["duration_ms"])
        ref_median = float(np.median([float(item["duration_ms"]) for item in refs]))
        row = dict(episode)
        row.update({
            "query_duration_ms": query_duration,
            "query_row_count": int(query["row_count"]),
            "reference_median_duration_ms": ref_median,
            "duration_ratio": query_duration / max(ref_median, 1.0),
            "writer_genuine_median_ms": genuine[writer]["median"],
            "writer_genuine_q25_ms": genuine[writer]["q25"],
            "writer_genuine_q75_ms": genuine[writer]["q75"],
            "in_writer_genuine_iqr": genuine[writer]["q25"] <= query_duration <= genuine[writer]["q75"],
        })
        rows.append(row)
    return rows


def duration_score(row: dict[str, Any], feature: str) -> float:
    value = float(row[feature])
    return -math.log1p(max(value, 0.0))


def fit_threshold(rows: list[dict[str, Any]], feature: str) -> float:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    scores = np.asarray([duration_score(row, feature) for row in rows], dtype=np.float64)
    unique = np.unique(scores)
    candidates = np.r_[
        np.nextafter(unique[0], -np.inf),
        (unique[:-1] + unique[1:]) / 2,
        np.nextafter(unique[-1], np.inf),
    ] if len(unique) else np.asarray([0.0])
    ranked = []
    for threshold in candidates:
        summary = binary_summary(labels, scores, float(threshold))
        ranked.append((summary["accuracy"], -abs(float(threshold)), float(threshold)))
    return max(ranked)[-1]


def evaluate_duration_classifier(
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature: str,
) -> dict[str, Any]:
    threshold = fit_threshold(val_rows, feature)
    def scored(rows):
        return [{**row, "score": duration_score(row, feature), "raw_logit": duration_score(row, feature)} for row in rows]
    val_scored, test_scored = scored(val_rows), scored(test_rows)
    metrics = t1_metrics(test_scored, threshold)
    sf = [row for row in test_scored if row["label"] == 1 or row["attack_type"] == "SF"]
    rf = [row for row in test_scored if row["label"] == 1 or row["attack_type"] == "RF"]
    return {
        "feature": feature,
        "threshold": threshold,
        "validation_accuracy": binary_summary(
            np.asarray([row["label"] for row in val_scored]),
            np.asarray([row["score"] for row in val_scored]),
            threshold,
        )["accuracy"],
        "test": metrics,
        "test_sf_far": far_with_ci([row for row in test_scored if row["attack_type"] == "SF"], threshold),
        "test_rf_far": far_with_ci([row for row in test_scored if row["attack_type"] == "RF"], threshold),
        "test_genuine_vs_SF": t1_metrics(sf, threshold).get("genuine_vs_SF"),
        "test_genuine_vs_RF": t1_metrics(rf, threshold).get("genuine_vs_RF"),
    }


def duration_matched_subset(rows: list[dict[str, Any]], attack: str = "SF") -> list[dict[str, Any]]:
    genuine = [row for row in rows if row["label"] == 1]
    matched = [row for row in rows if row["attack_type"] == attack and row["in_writer_genuine_iqr"]]
    return genuine + matched


def logit_duration_correlation(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_attack: dict[str, dict[str, float]] = {}
    for attack, group in (
        ("genuine", [row for row in rows if row["label"] == 1]),
        ("RF", [row for row in rows if row["attack_type"] == "RF"]),
        ("SF", [row for row in rows if row["attack_type"] == "SF"]),
        ("all", rows),
    ):
        if len(group) < 3:
            continue
        logit = np.asarray([row["raw_logit"] for row in group], dtype=np.float64)
        duration = np.log1p(np.asarray([row["query_duration_ms"] for row in group], dtype=np.float64))
        if np.std(logit) == 0 or np.std(duration) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(logit, duration)[0, 1])
        by_attack[attack] = {
            "n": int(len(group)),
            "logit_mean": float(logit.mean()),
            "duration_median": float(np.median([row["query_duration_ms"] for row in group])),
            "corr_logit_log_duration": corr,
        }
    return by_attack


def forger_table(rows: list[dict[str, Any]], threshold: float) -> dict[str, dict[str, float]]:
    by_forger: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["attack_type"] == "SF":
            by_forger[str(row.get("forger_id") or "unknown")].append(row)
    return {forger: far_with_ci(group, threshold) | {
        "duration_median": float(np.median([row["query_duration_ms"] for row in group])),
    } for forger, group in sorted(by_forger.items())}


def load_split_episodes(benchmark_root: Path, split: str, protocol: str) -> list[dict[str, Any]]:
    return read_jsonl(benchmark_root / "episodes/fold_0" / f"{split}_{protocol}.jsonl")


def load_predictions(path: Path, samples: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return attach_duration(read_jsonl(path), samples)


def evaluate_model_predictions(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    metrics = t1_metrics(rows, threshold)
    matched = duration_matched_subset(rows, "SF")
    matched_rf = duration_matched_subset(rows, "RF")
    return {
        "n": len(rows),
        "overall": metrics["overall"],
        "genuine_vs_SF": metrics.get("genuine_vs_SF"),
        "genuine_vs_RF": metrics.get("genuine_vs_RF"),
        "sf_far": far_with_ci([row for row in rows if row["attack_type"] == "SF"], threshold),
        "rf_far": far_with_ci([row for row in rows if row["attack_type"] == "RF"], threshold),
        "duration_matched_sf": {
            "n_sf": sum(row["attack_type"] == "SF" for row in matched),
            "metrics": t1_metrics(matched, threshold) if len({row["label"] for row in matched}) == 2 else None,
            "sf_far": far_with_ci([row for row in matched if row["attack_type"] == "SF"], threshold),
        },
        "duration_matched_rf": {
            "n_rf": sum(row["attack_type"] == "RF" for row in matched_rf),
            "sf_far": far_with_ci([row for row in matched_rf if row["attack_type"] == "RF"], threshold),
        },
        "logit_duration": logit_duration_correlation(rows),
        "sf_by_forger": forger_table(rows, threshold),
    }


def run_diagnostics(
    dataset_root: Path,
    benchmark_root: Path,
    s2f0_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    samples = load_manifest(dataset_root / "manifest.json")
    calibration = json.loads((s2f0_root / "t1_calibration.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "duration_by_state": duration_table(samples),
        "protocols": {},
    }
    for protocol in PROTOCOLS:
        val = attach_duration(load_split_episodes(benchmark_root, "val", protocol), samples)
        test = attach_duration(load_split_episodes(benchmark_root, "test", protocol), samples)
        classifiers = {
            feature: evaluate_duration_classifier(val, test, feature)
            for feature in ("query_duration_ms", "query_row_count", "duration_ratio")
        }
        pred_path = s2f0_root / "test/predictions" / f"test_{protocol}.jsonl"
        model = None
        if pred_path.is_file():
            predictions = load_predictions(pred_path, samples)
            model = evaluate_model_predictions(predictions, float(calibration[f"{protocol}_threshold"]))
        report["protocols"][protocol] = {
            "n_val": len(val),
            "n_test": len(test),
            "n_test_sf": sum(row["attack_type"] == "SF" for row in test),
            "n_test_sf_in_genuine_iqr": sum(
                row["attack_type"] == "SF" and row["in_writer_genuine_iqr"] for row in test
            ),
            "duration_classifiers": classifiers,
            "s2f0": model,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "duration_hack_report.json", report)
    (output_dir / "duration_hack_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{100 * value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# T1 SF duration-shortcut diagnostics",
        "",
        "SF median duration is several times the genuine median. This report asks whether that leak is enough to reject SF without using spatial form.",
        "",
        "## Duration by state",
        "",
        "| state | n | duration median (ms) | p10 | p90 | points median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for state, row in report["duration_by_state"].items():
        if "duration_median" not in row:
            continue
        lines.append(
            f"| {state} | {row['n']} | {row['duration_median']:.0f} | {row['duration_p10']:.0f} | "
            f"{row['duration_p90']:.0f} | {row['points_median']:.0f} |"
        )
    ratio = report["duration_by_state"]["SF_over_writer_genuine_median"]
    lines += [
        "",
        f"Writer-level SF / genuine-median duration: mean {ratio['mean']:.2f}, median {ratio['median']:.2f}.",
        "",
    ]
    for protocol, block in report["protocols"].items():
        lines += [f"## {protocol}", ""]
        lines += [
            f"Test SF in writer genuine IQR: {block['n_test_sf_in_genuine_iqr']} / {block['n_test_sf']}.",
            "",
            "Single-feature classifiers, threshold fit on Validation accuracy. Higher duration/ratio -> forgery.",
            "",
        ]
        for name, clf in block["duration_classifiers"].items():
            gsf = clf.get("test_genuine_vs_SF") or {}
            lines.append(
                f"- {name}: SF FAR {pct(clf['test_sf_far']['far'])}, "
                f"RF FAR {pct(clf['test_rf_far']['far'])}, "
                f"genuine-vs-SF EER {pct(gsf.get('eer'))}, "
                f"AUC {gsf.get('roc_auc', float('nan')):.4f}"
            )
        lines.append("")
        model = block.get("s2f0")
        if not model:
            lines.append("S2F0 Test predictions were not found.")
            continue
        matched = model["duration_matched_sf"]
        lines += [
            "S2F0 frozen-threshold Test:",
            "",
            f"- SF FAR: {pct(model['sf_far']['far'])} "
            f"(Wilson {pct(model['sf_far']['far_ci_low'])}–{pct(model['sf_far']['far_ci_high'])})",
            f"- RF FAR: {pct(model['rf_far']['far'])}",
            f"- Duration-matched SF remaining: {matched['n_sf']}, FAR {pct(matched['sf_far']['far'])}",
            f"- Corr(logit, log duration) SF: {model['logit_duration'].get('SF', {}).get('corr_logit_log_duration')}",
            "",
            "SF FAR by forger:",
            "",
        ]
        for forger, row in model["sf_by_forger"].items():
            lines.append(
                f"- {forger}: FAR {pct(row['far'])} (n={row['n']}, duration median {row['duration_median']:.0f} ms)"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/signatures")
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "datasets/protocols/t1")
    parser.add_argument("--s2f0-root", type=Path, default=DEFAULT_S2F0)
    parser.add_argument("--output", type=Path, default=ROOT / "ablation/t1_sf_hack/results_duration_p0")
    args = parser.parse_args()
    report = run_diagnostics(args.dataset_root, args.benchmark_root, args.s2f0_root, args.output)
    print(args.output / "duration_hack_report.md")
    for protocol, block in report["protocols"].items():
        clf = block["duration_classifiers"]["duration_ratio"]
        print(
            f"{protocol} duration-ratio SF FAR={clf['test_sf_far']['far']:.4f} "
            f"EER={clf['test_genuine_vs_SF']['eer']:.4f} n={clf['test_sf_far']['n']}"
        )


if __name__ == "__main__":
    main()
