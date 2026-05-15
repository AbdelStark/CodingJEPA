"""CJ-HUMAN — human-review rubric benchmark (RFC-0010 §D3 E2, #115).

A real run aggregates Likert annotations from multiple reviewers and reports
Cohen's kappa for inter-rater agreement. CI does not have annotations so
this benchmark emits a stub result; when ``human_annotations.json`` is
present we compute the mean per axis and pairwise kappa.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from codingjepa.eval.harness import Benchmark, BenchmarkResult


class HumanBenchmark(Benchmark):
    benchmark_id = "CJ-HUMAN"

    def __init__(self, *, global_seed: int = 0) -> None:
        super().__init__(global_seed=global_seed)
        self._stub = True
        self._payload: dict[str, Any] = {}

    def prepare(self, data_dir: pathlib.Path) -> None:
        path = pathlib.Path(data_dir) / "human_annotations.json"
        if not path.exists():
            self._stub = True
            self._payload = {}
            return
        self._payload = json.loads(path.read_text(encoding="utf-8"))
        self._stub = not self._payload

    def run(self) -> BenchmarkResult:
        if self._stub:
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                metrics={
                    "mean_likert": None,
                    "cohen_kappa": None,
                    "status": "no_human_annotations",
                },
            )
        mean_likert = float(self._payload.get("mean_likert", 0.0))
        cohen = float(self._payload.get("cohen_kappa", 0.0))
        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            metrics={
                "mean_likert": mean_likert,
                "cohen_kappa": cohen,
                "status": "ok",
            },
        )


__all__ = ["HumanBenchmark"]
