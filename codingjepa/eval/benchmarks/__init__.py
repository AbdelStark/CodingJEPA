"""Public surface of :mod:`codingjepa.eval.benchmarks` — every Benchmark
subclass registered by RFC-0010 §D3 lives here so callers can import them
from a single namespace.
"""

from __future__ import annotations

from codingjepa.eval.benchmarks.exec import ExecBenchmark
from codingjepa.eval.benchmarks.human import HumanBenchmark
from codingjepa.eval.benchmarks.intent import IntentBenchmark
from codingjepa.eval.benchmarks.ood import OodBenchmark
from codingjepa.eval.benchmarks.probes import (
    ProbeCloneBenchmark,
    ProbeDefectBenchmark,
    ProbeNameBenchmark,
)
from codingjepa.eval.benchmarks.ret import (
    RetBenchmark,
    RetBenchmark1k,
    RetBenchmark100,
)
from codingjepa.eval.benchmarks.robustness import (
    RobDocBenchmark,
    RobFmtBenchmark,
    RobRenameBenchmark,
    RobustnessBenchmark,
)

__all__ = [
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
]
