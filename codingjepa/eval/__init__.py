"""Eval harness surface (RFC-0010, spec/02 §codingjepa.eval). Placeholder stubs."""

from __future__ import annotations

import pathlib
from typing import Any, Protocol


class BenchmarkContext(Protocol):
    """Runtime context passed into Benchmark.prepare/run/score. See RFC-0010."""

    checkpoint_path: pathlib.Path
    manifest_path: pathlib.Path
    out_dir: pathlib.Path


class Benchmark(Protocol):
    name: str

    def prepare(self, ctx: BenchmarkContext) -> None: ...
    def run(self, ctx: BenchmarkContext) -> dict[str, Any]: ...
    def score(self, raw: dict[str, Any]) -> dict[str, Any]: ...


def run_all(checkpoint: pathlib.Path, manifest: pathlib.Path, out: pathlib.Path) -> None:
    """Run every Benchmark and emit results/results.json. Placeholder; #107 implements."""
    raise NotImplementedError


__all__ = ["Benchmark", "BenchmarkContext", "run_all"]
