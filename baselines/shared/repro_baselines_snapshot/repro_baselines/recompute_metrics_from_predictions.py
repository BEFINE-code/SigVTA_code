from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from repro_baselines.evaluation import seven_class_metrics, write_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = root / "runs" / "repro_baselines_final_l5_v3"
    updated = 0
    for path in sorted(runs.glob("*/seed_*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if "t2" not in metrics:
            continue
        predictions = path.parent / "predictions"
        validation_path = predictions / "val_t2.jsonl"
        test_path = predictions / "test_t2.jsonl"
        if not validation_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(f"Missing final T2 predictions below {predictions}")
        previous_test = metrics["t2"]["test"]
        validation = seven_class_metrics(read_jsonl(validation_path))
        test = seven_class_metrics(read_jsonl(test_path))
        test.update({key: value for key, value in previous_test.items() if key.startswith("writer_bootstrap_")})
        metrics["t2"]["validation"] = validation
        metrics["t2"]["test"] = test
        if metrics.get("model") == "synsig2vec":
            manifest_path = runs / "synsig2vec_official" / "prepared" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics["fit"]["prepared_split_digest"] = manifest.get("split_digest")
            metrics["fit"]["prepared_cache_scope"] = (
                "Train-only; reused because the final and legacy Train writer sets are identical"
            )
        write_json(path, metrics)
        updated += 1
    print(f"updated {updated} final-protocol metrics files")


if __name__ == "__main__":
    main()
