"""Windows watchdog for the local T1 SF-hack run.

Does not start a second trainer if one is already running. On crash or stall,
restarts `run.py train`, which resumes from last_t1.pt. After training,
continues freeze → test → export.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "dvsrc").is_dir())
STUDY = ROOT / "ablation/t1_sf_hack"
RUNTIME = STUDY / "runtime"
IDS = ("H-T1", "H-T2", "H-T3", "H-N1", "H-N2")
PYTHON = Path(os.environ.get("PYTHON_PATH", sys.executable))
STALE_SECONDS = 1200
MAX_FAILURES = 5
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "cuDNN error")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


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
    value = read(summary)
    if (value.get("status") != "complete" or value.get("validation_completed") != len(IDS)
            or value.get("test_completed") != len(IDS) or value.get("weights_included") is not False):
        raise ValueError("Invalid existing export; manual integrity review required")
    return "complete"


def run_status(item: str):
    run = STUDY / "runs" / item / "seed_42"
    status_path = run / "t1_training_status.json"
    training = read(status_path) if status_path.is_file() else None
    log = run / "stdout.log"
    last = None
    if log.is_file():
        lines = [line.strip() for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        last = lines[-1] if lines else None
    return {
        "id": item,
        "validation_complete": (run / "validation_checkpoint.json").is_file(),
        "test_complete": (run / "test/test_release_manifest.json").is_file(),
        "has_resume": (run / "last_t1.pt").is_file(),
        "completed_epochs": None if not training else training.get("completed_epochs"),
        "best_score": None if not training else training.get("best_score"),
        "running_flag": None if not training else training.get("running"),
        "last_log": last,
        "log_age_seconds": None if not log.is_file() else time.time() - log.stat().st_mtime,
    }


def newest_log_age():
    ages = []
    for item in IDS:
        run = STUDY / "runs" / item / "seed_42"
        if (run / "validation_checkpoint.json").is_file() and not (run / "test").exists():
            continue
        for name in ("stdout.log", "stderr.log", "t1_training_status.json", "history.jsonl",
                     "last_t1.pt", "test/stdout.log", "test/test_metrics.json"):
            path = run / name
            if path.exists():
                ages.append(time.time() - path.stat().st_mtime)
    return min(ages) if ages else None


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


def trainer_pids():
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or "python" not in parts[1].lower():
            continue
        try:
            rows.append({"pid": int(parts[0]), "command": parts[1]})
        except ValueError:
            continue
    return rows


def logs_recently_active(seconds: int = 180):
    age = newest_log_age()
    return age is not None and age < seconds


def kill_tree(pid: int):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=30)


def current_stderr_oom():
    for item in IDS:
        run = STUDY / "runs" / item / "seed_42"
        if (run / "validation_checkpoint.json").is_file():
            continue
        path = run / "stderr.log"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")[-8000:]
        if any(marker in text for marker in OOM_MARKERS):
            return item, text[-500:]
    return None, None


def start_phase(current: str):
    environment = {
        **os.environ,
        "PYTHON_PATH": str(PYTHON),
        "MAX_PARALLEL": "1",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "4"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "4"),
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    RUNTIME.mkdir(parents=True, exist_ok=True)
    log = (RUNTIME / "pipeline.log").open("ab")
    command = [str(PYTHON), "-u", str(STUDY / "run.py"), current]
    return subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log)


def snapshot(current: str, extra=None):
    gpu_text, utilization = gpu()
    state = {
        "time": now(),
        "phase": current,
        "gpu": gpu_text,
        "gpu_utilization": utilization,
        "workers": trainer_pids(),
        "log_age_seconds": newest_log_age(),
        "runs": [run_status(item) for item in IDS],
    }
    if extra:
        state.update(extra)
    write(RUNTIME / "status.json", state)
    return state


def estimate(state):
    active = next((row for row in state["runs"] if not row["validation_complete"]), None)
    completed = sum(1 for row in state["runs"] if row["validation_complete"])
    minutes_per_epoch = 28
    remaining_groups = len(IDS) - completed
    if active and active["completed_epochs"] is not None:
        done = int(active["completed_epochs"])
        remaining_epochs_this = max(10 - done, 0)
        remaining_minutes = remaining_epochs_this * minutes_per_epoch + max(remaining_groups - 1, 0) * 6 * minutes_per_epoch
        return {
            "active": active["id"],
            "completed_groups": completed,
            "completed_epochs_active": done,
            "minutes_per_epoch": minutes_per_epoch,
            "eta_hours_low": round((max(4 - done, 0) * minutes_per_epoch + max(remaining_groups - 1, 0) * 4 * minutes_per_epoch) / 60, 1),
            "eta_hours_high": round(remaining_minutes / 60, 1),
            "last_log": active["last_log"],
        }
    return {
        "active": None if remaining_groups == 0 else IDS[completed],
        "completed_groups": completed,
        "eta_hours_low": round((remaining_groups * 4 * minutes_per_epoch) / 60, 1),
        "eta_hours_high": round((remaining_groups * 8 * minutes_per_epoch) / 60, 1),
    }


def lock_handle():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    path = RUNTIME / "watch.lock"
    handle = path.open("a+b")
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=180)
    args = parser.parse_args()
    os.chdir(ROOT)
    handle = lock_handle()
    failures = 0
    owned = None
    try:
        while True:
            current = phase()
            state = snapshot(current, {"failures": failures, "estimate": None})
            state["estimate"] = estimate(state)
            write(RUNTIME / "status.json", state)
            print(json.dumps({
                "time": state["time"], "phase": current, "gpu": state["gpu"],
                "workers": len(state["workers"]), "failures": failures,
                "estimate": state["estimate"],
            }, ensure_ascii=False), flush=True)
            if current == "complete":
                return
            lock = STUDY / "execution.lock"
            if owned is None and (state["workers"] or logs_recently_active(180) or lock.exists()):
                time.sleep(args.interval)
                continue
            if owned is not None and owned.poll() is None:
                stale = (
                    state["log_age_seconds"] is not None
                    and state["log_age_seconds"] > STALE_SECONDS
                    and (state["gpu_utilization"] is None or state["gpu_utilization"] < 5)
                )
                oom_id, _ = current_stderr_oom()
                if stale or oom_id:
                    incident = {**state, "action": "restart_stale_or_oom", "oom_run": oom_id}
                    write(RUNTIME / f"incident-{time.time_ns()}.json", incident)
                    for worker in state["workers"]:
                        kill_tree(worker["pid"])
                    try:
                        owned.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        pass
                    owned = None
                    failures += 1
                    time.sleep(15)
                    continue
                time.sleep(args.interval)
                continue
            if owned is not None:
                code = owned.returncode
                owned = None
                if code:
                    failures += 1
                    write(RUNTIME / f"incident-{time.time_ns()}.json",
                          {**state, "exit_code": code, "failures": failures})
                    time.sleep(30)
                    continue
                failures = 0
                continue
            if failures >= MAX_FAILURES:
                state["blocked"] = "Five failed attempts; inspect runtime/incident-*.json"
                write(RUNTIME / "status.json", state)
                raise RuntimeError(state["blocked"])
            owned = start_phase(current)
            snapshot(current, {"failures": failures, "owned_pid": owned.pid, "action": f"start {current}"})
            time.sleep(args.interval)
    finally:
        handle.close()


if __name__ == "__main__":
    main()
