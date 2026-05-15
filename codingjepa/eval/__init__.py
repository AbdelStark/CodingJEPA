"""Eval harness surface (RFC-0010, spec/02 §codingjepa.eval).

This module re-exports the :class:`Benchmark` ABC, the
:class:`BenchmarkResult` dataclass, the :func:`run_suite` orchestrator and
every concrete benchmark so callers can write::

    from codingjepa.eval import run_suite, RetBenchmark100

A backwards-compatible :func:`run_all` thin wrapper (spec/02 names it as the
top-level eval entry point) drives every registered benchmark through
:func:`run_suite`.
"""

from __future__ import annotations

import pathlib
from typing import Any

from codingjepa.eval.benchmarks import (
    ExecBenchmark,
    HumanBenchmark,
    IntentBenchmark,
    OodBenchmark,
    ProbeCloneBenchmark,
    ProbeDefectBenchmark,
    ProbeNameBenchmark,
    RetBenchmark,
    RetBenchmark1k,
    RetBenchmark100,
    RobDocBenchmark,
    RobFmtBenchmark,
    RobRenameBenchmark,
    RobustnessBenchmark,
)
from codingjepa.eval.harness import Benchmark, BenchmarkResult, run_suite


def run_all(
    checkpoint: pathlib.Path,
    manifest: pathlib.Path,
    out: pathlib.Path,
    *,
    global_seed: int = 0,
) -> dict[str, Any]:
    """Run every Benchmark and emit ``out/results.json`` (RFC-0010 §D1).

    The ``checkpoint`` and ``manifest`` arguments are part of the public
    contract from ``spec/02-public-api.md``. They are consumed in production
    eval runs (to verify hashes); here we only need them so the signature
    stays stable for downstream callers. The benchmarks themselves draw on
    ``manifest.parent`` as the data directory.
    """

    del checkpoint  # consumed by the hash-check enforcer in `make eval`.
    data_dir = pathlib.Path(manifest).parent
    benchmarks: list[Benchmark] = [
        RetBenchmark100(),
        RetBenchmark1k(),
        IntentBenchmark(),
        RobFmtBenchmark(),
        RobRenameBenchmark(),
        RobDocBenchmark(),
        OodBenchmark(),
        ProbeNameBenchmark(),
        ProbeDefectBenchmark(),
        ProbeCloneBenchmark(),
        ExecBenchmark(),
        HumanBenchmark(),
    ]
    return run_suite(
        benchmarks,
        data_dir=data_dir,
        out_dir=pathlib.Path(out),
        global_seed=global_seed,
    )


__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "ExecBenchmark",
    "HumanBenchmark",
    "IntentBenchmark",
    "OodBenchmark",
    "ProbeCloneBenchmark",
    "ProbeDefectBenchmark",
    "ProbeNameBenchmark",
    "RetBenchmark",
    "RetBenchmark100",
    "RetBenchmark1k",
    "RobDocBenchmark",
    "RobFmtBenchmark",
    "RobRenameBenchmark",
    "RobustnessBenchmark",
    "run_all",
    "run_suite",
]
