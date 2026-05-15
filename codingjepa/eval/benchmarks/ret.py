"""CJ-RET-100 and CJ-RET-1k retrieval benchmarks (RFC-0010 §D3, #108).

The metric is R@1, R@5, R@10, MRR computed from a FAISS ``IndexFlatIP``
search over a fixed pool of 100 or 1000 candidate embeddings. At test time
there is no real checkpoint, so this module uses synthetic embeddings seeded
from the per-benchmark seed; each query is one of the pool vectors so
retrieval is trivially perfect.
"""

from __future__ import annotations

import pathlib

import numpy as np
import numpy.typing as npt

from codingjepa.eval.harness import Benchmark, BenchmarkResult


def _recall_at_k(ranks: list[int], k: int) -> float:
    if not ranks:
        return 0.0
    return float(sum(1 for r in ranks if r <= k) / len(ranks))


def _mrr(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return float(sum(1.0 / r for r in ranks) / len(ranks))


class RetBenchmark(Benchmark):
    """Shared FAISS-pool retrieval base for CJ-RET-100 and CJ-RET-1k."""

    benchmark_id = "CJ-RET"

    def __init__(
        self,
        *,
        pool_size: int,
        n_queries: int = 100,
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
        self._pool = pool
        self._queries = pool[idxs].copy()
        self._targets = idxs.tolist()

    def run(self) -> BenchmarkResult:
        if self._pool is None or self._queries is None:
            raise RuntimeError("prepare() must be called before run()")
        import faiss

        # Force single-threaded OMP — multi-threaded faiss search corrupts
        # libomp state when torch is loaded in the same process (macOS).
        faiss.omp_set_num_threads(1)
        index = faiss.IndexFlatIP(self._dim)
        index.add(self._pool)
        topk = min(10, self._pool_size)
        _scores, indices = index.search(self._queries, topk)
        ranks: list[int] = []
        for i, target in enumerate(self._targets):
            row = indices[i].tolist()
            rank = row.index(target) + 1 if target in row else self._pool_size + 1
            ranks.append(rank)
        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            metrics={
                "R@1": _recall_at_k(ranks, 1),
                "R@5": _recall_at_k(ranks, 5),
                "R@10": _recall_at_k(ranks, 10),
                "MRR": _mrr(ranks),
                "n_queries": len(ranks),
                "pool_size": self._pool_size,
            },
        )


class RetBenchmark100(RetBenchmark):
    benchmark_id = "CJ-RET-100"

    def __init__(self, *, n_queries: int = 100, dim: int = 512, global_seed: int = 0) -> None:
        super().__init__(
            pool_size=100,
            n_queries=n_queries,
            dim=dim,
            global_seed=global_seed,
        )


class RetBenchmark1k(RetBenchmark):
    benchmark_id = "CJ-RET-1k"

    def __init__(self, *, n_queries: int = 100, dim: int = 512, global_seed: int = 0) -> None:
        super().__init__(
            pool_size=1000,
            n_queries=n_queries,
            dim=dim,
            global_seed=global_seed,
        )


__all__ = ["RetBenchmark", "RetBenchmark100", "RetBenchmark1k"]
