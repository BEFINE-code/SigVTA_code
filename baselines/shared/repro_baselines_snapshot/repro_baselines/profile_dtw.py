from __future__ import annotations

import time

from repro_baselines.data import BenchmarkRepository, SignatureStore
from repro_baselines.dtw import DTWScorer


def main() -> None:
    repo = BenchmarkRepository(".")
    repo.verify()
    scorer = DTWScorer(SignatureStore(repo.dataset))
    rows = repo.episodes("t2", "val")
    pairs = sorted({
        tuple(sorted((candidate, row["query_id"])))
        for row in rows
        for candidate in row["candidate_ids"]
    })
    for count in (100, 500):
        started = time.perf_counter()
        scorer.score_pairs(pairs[:count])
        elapsed = time.perf_counter() - started
        print({"pairs": count, "incremental_seconds": elapsed, "seconds_per_requested_pair": elapsed / count})


if __name__ == "__main__":
    main()
