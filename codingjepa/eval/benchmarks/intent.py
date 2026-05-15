"""CJ-INTENT — intent-conditioned R@5 vs. unconditional R@5 (RFC-0010 §D3, #109).

We compare retrieval performance when the query embedding has access to the
intent tag (``conditioned``) versus a stripped variant (``unconditioned``).
For testing we synthesize this by interpolating the conditioned query
closer to the target than the unconditioned query, so the conditioned
variant retrieves the target into the top-5 at least as often.
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


def _ranks(
    queries: npt.NDArray[np.float32],
    pool: npt.NDArray[np.float32],
    targets: list[int],
    dim: int,
) -> list[int]:
    import faiss

    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(dim)
    index.add(pool)
    pool_size = pool.shape[0]
    _scores, indices = index.search(queries, min(10, pool_size))
    out: list[int] = []
    for i, target in enumerate(targets):
        row = indices[i].tolist()
        rank = row.index(target) + 1 if target in row else pool_size + 1
        out.append(rank)
    return out


class IntentBenchmark(Benchmark):
    """Intent-conditioned R@5 vs. unconditional R@5."""

    benchmark_id = "CJ-INTENT"

    def __init__(
        self,
        *,
        pool_size: int = 100,
        n_queries: int = 100,
        dim: int = 512,
        global_seed: int = 0,
        condition_weight: float = 0.5,
        noise_weight: float = 0.6,
    ) -> None:
        super().__init__(global_seed=global_seed)
        self._pool_size = pool_size
        self._n_queries = n_queries
        self._dim = dim
        self._condition_weight = condition_weight
        self._noise_weight = noise_weight
        self._pool: npt.NDArray[np.float32] | None = None
        self._cond: npt.NDArray[np.float32] | None = None
        self._uncond: npt.NDArray[np.float32] | None = None
        self._targets: list[int] = []

    def prepare(self, data_dir: pathlib.Path) -> None:  # noqa: ARG002 — interface
        rng = np.random.default_rng(self.seed)
        pool = rng.random((self._pool_size, self._dim)).astype(np.float32)
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        idxs = rng.integers(0, self._pool_size, size=self._n_queries)
        targets = pool[idxs]
        noise = rng.standard_normal((self._n_queries, self._dim)).astype(np.float32)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)

        # Conditioned: closer to target (small noise mix).
        cw = self._condition_weight
        cond = cw * targets + (1.0 - cw) * noise
        cond /= np.linalg.norm(cond, axis=1, keepdims=True)
        # Unconditioned: heavier noise mix.
        nw = self._noise_weight
        uncond = (1.0 - nw) * targets + nw * noise
        uncond /= np.linalg.norm(uncond, axis=1, keepdims=True)

        self._pool = pool
        self._cond = cond
        self._uncond = uncond
        self._targets = idxs.tolist()

    def run(self) -> BenchmarkResult:
        if self._pool is None or self._cond is None or self._uncond is None:
            raise RuntimeError("prepare() must be called before run()")
        cond_ranks = _ranks(self._cond, self._pool, self._targets, self._dim)
        uncond_ranks = _ranks(self._uncond, self._pool, self._targets, self._dim)
        r5_cond = _recall_at_k(cond_ranks, 5)
        r5_uncond = _recall_at_k(uncond_ranks, 5)
        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            metrics={
                "R@5_conditioned": r5_cond,
                "R@5_unconditioned": r5_uncond,
                "delta_R5": r5_cond - r5_uncond,
                "n_queries": len(cond_ranks),
                "pool_size": self._pool_size,
            },
        )


__all__ = ["IntentBenchmark"]
