"""CJ-OOD — out-of-distribution retrieval (RFC-0010 §D3, #113).

200 synthetic OOD pairs evaluated with R@10 over a 200-candidate pool.
Reuses the FAISS retrieval pattern from CJ-RET-* but with a different
``benchmark_id`` so it gets its own seed and lock file.
"""

from __future__ import annotations

import pathlib

import numpy as np
import numpy.typing as npt

from codingjepa.eval.harness import Benchmark, BenchmarkResult


class OodBenchmark(Benchmark):
    benchmark_id = "CJ-OOD"

    def __init__(
        self,
        *,
        pool_size: int = 200,
        n_queries: int = 200,
        dim: int = 512,
        global_seed: int = 0,
    ) -> None:
        super().__init__(global_seed=global_seed)
        self._pool_size = pool_size
        self._n_queries = n_queries
        self._dim = dim
        self._pool: npt.NDArray[np.float32] | None = None
        self._queries: npt.NDArray[np.float32] | None = None
        self._targets: list[int] = []

    def prepare(self, data_dir: pathlib.Path) -> None:  # noqa: ARG002 — interface
        rng = np.random.default_rng(self.seed)
        pool = rng.random((self._pool_size, self._dim)).astype(np.float32)
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        idxs = rng.integers(0, self._pool_size, size=self._n_queries)
        # Add light OOD noise so retrieval is non-trivial.
        noise = rng.standard_normal((self._n_queries, self._dim)).astype(np.float32)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        queries = 0.85 * pool[idxs] + 0.15 * noise
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        self._pool = pool
        self._queries = queries
        self._targets = idxs.tolist()

    def run(self) -> BenchmarkResult:
        if self._pool is None or self._queries is None:
            raise RuntimeError("prepare() must be called before run()")
        import faiss

        faiss.omp_set_num_threads(1)
        index = faiss.IndexFlatIP(self._dim)
        index.add(self._pool)
        topk = min(10, self._pool_size)
        _scores, indices = index.search(self._queries, topk)
        hits = 0
        for i, target in enumerate(self._targets):
            if target in indices[i].tolist():
                hits += 1
        r10 = float(hits / len(self._targets)) if self._targets else 0.0
        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            metrics={
                "R@10": r10,
                "n_queries": len(self._targets),
                "pool_size": self._pool_size,
            },
        )


__all__ = ["OodBenchmark"]
