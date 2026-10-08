"""Compact SF-hack status for the 5090 workspace. Run on the server."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/t1_sf_hack")
STUDY = ROOT / "ablation/t1_sf_hack"
IDS = ("H-T2", "H-T3", "H-N1", "H-N2")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def last_line(path: Path):
    if not path.is_file():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-1] if lines else None


def gpu():
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    text = query.stdout.strip()
    try:
        utilization = float(text.splitlines()[0].split(",")[0])
    except (ValueError, IndexError):
        utilization = None
    return text, utilization


def tmux_alive():
    return subprocess.run(["tmux", "has-session", "-t", "t1_sf_hack"], capture_output=True).returncode == 0


def phase():
    runs = [STUDY / "runs" / item / "seed_42" for item in IDS]
    if not all((p / "validation_checkpoint.json").is_file() for p in runs):
        return "train"
    if not (STUDY / "test_checkpoint_manifest.json").is_file():
        return "freeze"
    if not all((p / "test/test_release_manifest.json").is_file() for p in runs):
        return "test"
    summary = STUDY / "results_seed42/summary.json"
    if not summary.is_file():
        return "export"
    return "complete"


def newest_log_age():
    ages = []
    for item in IDS:
        run = STUDY / "runs" / item / "seed_42"
        for name in ("stdout.log", "stderr.log", "t1_training_status.json", "history.jsonl",
                     "last_t1.pt", "test/stdout.log"):
            path = run / name
            if path.exists():
                ages.append(time.time() - path.stat().st_mtime)
    return min(ages) if ages else None


def run_row(item: str):
    run = STUDY / "runs" / item / "seed_42"
    status_path = run / "t1_training_status.json"
    training = read(status_path) if status_path.is_file() else {}
    log = run / "stdout.log"
    return {
        "id": item,
        "validation_complete": (run / "validation_checkpoint.json").is_file(),
        "test_complete": (run / "test/test_release_manifest.json").is_file(),
        "completed_epochs": training.get("completed_epochs"),
        "running": training.get("running"),
        "best_score": training.get("best_score") or training.get("best_selection_score"),
        "early_stop": training.get("early_stop_reason"),
        "last_log": last_line(log),
        "log_age_seconds": None if not log.is_file() else time.time() - log.stat().st_mtime,
        "stderr_tail": last_line(run / "stderr.log"),
    }


def estimate(rows, current):
    minutes_per_epoch = 15
    completed = sum(1 for row in rows if row["validation_complete"])
    remaining_groups = len(IDS) - completed
    active = next((row for row in rows if not row["validation_complete"]), None)
    extra = {"test_export_hours": 1.0}
    if current in {"freeze", "test", "export"}:
        return {"eta_hours_low": 0.5, "eta_hours_high": 1.5, "note": current}
    if current == "complete":
        return {"eta_hours_low": 0, "eta_hours_high": 0, "note": "complete"}
    done = int(active["completed_epochs"] or 0) if active else 0
    this_left = max(10 - done, 0)
    low = (max(4 - done, 0) * minutes_per_epoch + max(remaining_groups - 1, 0) * 4 * minutes_per_epoch) / 60
    high = (this_left * minutes_per_epoch + max(remaining_groups - 1, 0) * 7 * minutes_per_epoch) / 60
    return {
        "active": None if not active else active["id"],
        "completed_groups": completed,
        "completed_epochs_active": done,
        "eta_hours_low": round(low + extra["test_export_hours"] * 0.3, 1),
        "eta_hours_high": round(high + extra["test_export_hours"], 1),
    }


def main():
    current = phase()
    gpu_text, utilization = gpu()
    rows = [run_row(item) for item in IDS]
    state = {
        "time": time.time(),
        "phase": current,
        "tmux": tmux_alive(),
        "gpu": gpu_text,
        "gpu_utilization": utilization,
        "log_age_seconds": newest_log_age(),
        "runs": rows,
        "estimate": estimate(rows, current),
        "export_ready": (STUDY / "results_seed42/summary.json").is_file(),
    }
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
