"""Linux-only serial supervisor for the T1 SF-hack study."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "dvsrc").is_dir())
STUDY = ROOT / "ablation/t1_sf_hack"
RUNTIME = STUDY / "runtime"
IDS = ("H-T2", "H-T3", "H-N1", "H-N2")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def phase(study=STUDY):
    runs = [study / "runs" / item / "seed_42" for item in IDS]
    if not all((p / "validation_checkpoint.json").is_file() for p in runs):
        return "train"
    if not (study / "test_checkpoint_manifest.json").is_file():
        return "freeze"
    if not all((p / "test/test_release_manifest.json").is_file() for p in runs):
        return "test"
    summary = study / "results_seed42/summary.json"
    if not summary.is_file():
        return "export"
    value = read(summary)
    if (value.get("status") != "complete" or value.get("validation_completed") != len(IDS)
            or value.get("test_completed") != len(IDS) or value.get("weights_included") is not False):
        raise ValueError("Invalid existing export; manual integrity review required")
    return "complete"


def matches_worker(args, cwd, root=ROOT):
    if cwd != str(root):
        return False
    runner = str(root / "ablation/t1_sf_hack/run.py")
    return (("dvsrc.cli" in args and "train-t1" in args)
            or (runner in args and any(p in args for p in ("train", "test", "freeze", "export")))
            or (any(path in args for path in ("ablation/t1_sf_hack/run.py", runner))
                and any(p in args for p in ("train", "test", "freeze", "export"))))


def workers():
    result = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
            cwd = str(proc.joinpath("cwd").resolve(strict=True))
            if matches_worker(args, cwd):
                result.append({"pid": int(proc.name), "args": args})
        except (OSError, UnicodeError):
            continue
    return result


def snapshot(current):
    rows, mtimes = [], []
    for item in IDS:
        run = STUDY / "runs" / item / "seed_42"
        status = run / "t1_training_status.json"
        rows.append({"id": item,
                     "validation_complete": (run / "validation_checkpoint.json").is_file(),
                     "test_complete": (run / "test/test_release_manifest.json").is_file(),
                     "training": read(status) if status.is_file() else None})
        for name in ("stdout.log", "stderr.log", "t1_training_status.json", "history.jsonl",
                     "last_t1.pt", "test/stdout.log", "test/stderr.log", "test/test_metrics.json"):
            path = run / name
            if path.exists():
                mtimes.append(path.stat().st_mtime)
    query = subprocess.run([
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    gpu = query.stdout.strip()
    try:
        utilization = float(gpu.splitlines()[0].split(",")[0])
    except (ValueError, IndexError):
        utilization = None
    return {"time": time.time(), "phase": current, "gpu": gpu,
            "gpu_utilization": utilization, "workers": workers(), "runs": rows,
            "log_age_seconds": time.time() - max(mtimes) if mtimes else 0}


def stalled(state):
    return (state["phase"] in ("train", "test") and state["log_age_seconds"] > 1800
            and state["gpu_utilization"] is not None and state["gpu_utilization"] < 5)


def stop_owned(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=20)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=1800)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("interval must be at least 30 seconds")
    os.chdir(ROOT)
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = 0
        while True:
            current = phase()
            state = snapshot(current)
            write(RUNTIME / "status.json", state)
            print(json.dumps({"time": state["time"], "phase": current,
                              "gpu": state["gpu"], "failures": failures}), flush=True)
            if current == "complete":
                return
            if state["workers"]:
                state["blocked"] = "Existing workspace workers; no duplicate started"
                write(RUNTIME / "status.json", state)
                time.sleep(args.interval)
                continue
            if failures >= 3:
                state["blocked"] = "Three failed attempts; inspect preserved evidence before retry"
                write(RUNTIME / "status.json", state)
                raise RuntimeError(state["blocked"])
            extra = ["--from-id", "H-T2"] if current in {"train", "test"} else []
            command = ["bash", str(STUDY / "run_5090.sh"), current, *extra]
            environment = {**os.environ, "PYTHON_PATH": sys.executable, "MAX_PARALLEL": "1"}
            with (RUNTIME / "pipeline.log").open("ab") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                         stdout=log, stderr=log, start_new_session=True)
                stale_count = 0
                while child.poll() is None:
                    try:
                        child.wait(timeout=args.interval)
                    except subprocess.TimeoutExpired:
                        state = snapshot(current)
                        state["owned_pid"] = child.pid
                        stale_count = stale_count + 1 if stalled(state) else 0
                        state["stale_checks"] = stale_count
                        write(RUNTIME / "status.json", state)
                        print(json.dumps({"time": state["time"], "phase": current,
                                          "gpu": state["gpu"], "stale_checks": stale_count}), flush=True)
                        if stale_count >= 2:
                            write(RUNTIME / f"incident-{time.time_ns()}.json", state)
                            stop_owned(child)
                if child.returncode:
                    failures += 1
                    state = snapshot(current)
                    state.update(exit_code=child.returncode, failures=failures)
                    write(RUNTIME / f"incident-{time.time_ns()}.json", state)
                    time.sleep(args.interval)
                else:
                    failures = 0


if __name__ == "__main__":
    main()
