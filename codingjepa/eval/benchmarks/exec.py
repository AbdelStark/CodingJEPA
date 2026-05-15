"""CJ-EXEC — execution-preservation benchmark (RFC-0010 §D3 E3, #110).

A real evaluation requires sandboxed pytest runs which need a GPU + real
data. In CI / smoke environments where ``exec_pairs.jsonl`` is absent we
return a stub result. When pairs are present we iterate them through
:func:`codingjepa.eval.sandbox.run_in_sandbox` and report the pass rate.
"""

from __future__ import annotations

import json
import pathlib

from codingjepa.eval.harness import Benchmark, BenchmarkResult
from codingjepa.eval.sandbox import run_in_sandbox


class ExecBenchmark(Benchmark):
    benchmark_id = "CJ-EXEC"

    def __init__(self, *, global_seed: int = 0, timeout: int = 30) -> None:
        super().__init__(global_seed=global_seed)
        self._pairs: list[dict[str, str]] = []
        self._timeout = timeout
        self._stub = True

    def prepare(self, data_dir: pathlib.Path) -> None:
        path = pathlib.Path(data_dir) / "exec_pairs.jsonl"
        if not path.exists():
            self._stub = True
            self._pairs = []
            return
        with path.open(encoding="utf-8") as fp:
            self._pairs = [json.loads(line) for line in fp if line.strip()]
        self._stub = not self._pairs

    def run(self) -> BenchmarkResult:
        if self._stub:
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                metrics={
                    "pass_rate": 0.0,
                    "n_pairs": 0,
                    "status": "no_executable_pairs",
                },
            )
        passed = 0
        for pair in self._pairs:
            ok, _stdout, _rc = run_in_sandbox(
                pair["candidate"],
                pair["tests"],
                timeout=self._timeout,
                backend="none",
            )
            if ok:
                passed += 1
        total = len(self._pairs)
        rate = float(passed / total) if total else 0.0
        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            metrics={
                "pass_rate": rate,
                "n_pairs": total,
                "passed": passed,
                "status": "ok",
            },
        )


__all__ = ["ExecBenchmark"]
