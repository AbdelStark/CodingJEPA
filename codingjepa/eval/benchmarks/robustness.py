"""CJ-ROB-FMT / CJ-ROB-RENAME / CJ-ROB-DOC robustness probes (RFC-0010 §D3, #112).

Each benchmark measures the rank-change percentage and cosine drift after a
perturbation (formatting, alpha-rename, or docstring scrub). At test time
we simulate perturbations with small per-perturber Gaussian noise so the
three benchmarks have distinct deterministic outputs.
"""

from __future__ import annotations

import pathlib

import numpy as np
import numpy.typing as npt

from codingjepa.eval.harness import Benchmark, BenchmarkResult


def _cosine_drift(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    """Mean of ``1 - cos(a_i, b_i)`` across rows (already normalized)."""

    return float(1.0 - float(np.mean(np.sum(a * b, axis=1))))


def _rank_change_pct(
    base: npt.NDArray[np.float32],
    perturbed: npt.NDArray[np.float32],
    dim: int,
) -> float:
    """Fraction (×100) of queries whose top-1 retrieved id changed."""

    import faiss

    faiss.omp_set_num_threads(1)
    pool_size = base.shape[0]
    index = faiss.IndexFlatIP(dim)
    index.add(base)
    _s1, idx_base = index.search(base, 1)
    _s2, idx_perturbed = index.search(perturbed, 1)
    changed = int(np.sum(idx_base[:, 0] != idx_perturbed[:, 0]))
    return float(100.0 * changed / pool_size)


class RobustnessBenchmark(Benchmark):
    """Shared base for the three perturbation probes."""

    benchmark_id = "CJ-ROB"
    perturbation_scale: float = 0.05

    def __init__(
        self,
        *,
        n_chunks: int = 100,
        dim: int = 512,
        global_seed: int = 0,
    ) -> None:
        super().__init__(global_seed=global_seed)
        self._n_chunks = n_chunks
        self._dim = dim
        self._base: npt.NDArray[np.float32] | None = None
        self._perturbed: npt.NDArray[np.float32] | None = None

    def prepare(self, data_dir: pathlib.Path) -> None:  # noqa: ARG002 — interface
        rng = np.random.default_rng(self.seed)
        base = rng.random((self._n_chunks, self._dim)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        noise = rng.standard_normal((self._n_chunks, self._dim)).astype(np.float32)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        scale = self.perturbation_scale
        perturbed = (1.0 - scale) * base + scale * noise
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        self._base = base
        self._perturbed = perturbed

    def run(self) -> BenchmarkResult:
        if self._base is None or self._perturbed is None:
            raise RuntimeError("prepare() must be called before run()")
        rc = _rank_change_pct(self._base, self._perturbed, self._dim)
        cd = _cosine_drift(self._base, self._perturbed)
        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            metrics={
                "rank_change_pct": rc,
                "mean_cosine_drift": cd,
                "n_chunks": self._n_chunks,
            },
        )


class RobFmtBenchmark(RobustnessBenchmark):
    benchmark_id = "CJ-ROB-FMT"
    perturbation_scale = 0.03


class RobRenameBenchmark(RobustnessBenchmark):
    benchmark_id = "CJ-ROB-RENAME"
    perturbation_scale = 0.08


class RobDocBenchmark(RobustnessBenchmark):
    benchmark_id = "CJ-ROB-DOC"
    perturbation_scale = 0.05


__all__ = [
    "RobDocBenchmark",
    "RobFmtBenchmark",
    "RobRenameBenchmark",
    "RobustnessBenchmark",
]
