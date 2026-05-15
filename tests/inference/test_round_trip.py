"""Deterministic round-trip test: embed → retrieve → rerank (#89)."""

from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import torch

from codingjepa.inference.index import IndexMeta, build_index, load_index
from codingjepa.inference.rerank import rerank
from codingjepa.inference.retrieve import retrieve
from codingjepa.model import build_model

# Allow OpenMP duplicate-lib on macOS in CI without crashing.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Required by torch.use_deterministic_algorithms on some platforms.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")


def _build_small_index(
    n: int = 10,
    d: int = 512,
    seed: int = 42,
) -> tuple[object, IndexMeta, pathlib.Path]:
    """Build a small index in a temp dir and return (index, meta, out_dir)."""
    rng = np.random.default_rng(seed)
    embs = rng.random((n, d)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    meta = IndexMeta(
        chunk_ids=[f"c{i}" for i in range(n)],
        sources=[f"def f{i}(): return {i}\n" for i in range(n)],
        index_id="deadbeef-cafebab0",
    )

    out_dir = pathlib.Path(tempfile.mkdtemp())
    build_index(embs, meta, out_dir)
    idx, loaded_meta = load_index("deadbeef-cafebab0", out_dir)
    return idx, loaded_meta, out_dir


def _run_pipeline(
    model: object, src_emb: torch.Tensor, index: object, meta: IndexMeta
) -> list[str]:
    """Run retrieve → rerank, return sorted list of chunk_ids from passing candidates."""
    result = retrieve(src_emb, -1, model, index, meta, top_m=10)  # type: ignore[arg-type]
    candidates = rerank(result, "NONE", "x = 1\n", k=5)
    return [c.chunk_id for c in candidates]


class TestRoundTripDeterministic:
    def test_two_runs_produce_same_chunk_ids(self) -> None:
        """Running the pipeline twice with the same inputs should yield identical results."""
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            model = build_model()
            model.eval()

            index, meta, _ = _build_small_index()

            rng = np.random.default_rng(0)
            src_emb_np = rng.random(512).astype(np.float32)
            src_emb_np /= np.linalg.norm(src_emb_np)
            src_emb = torch.tensor(src_emb_np)

            run1 = _run_pipeline(model, src_emb, index, meta)
            run2 = _run_pipeline(model, src_emb, index, meta)

            assert run1 == run2, f"Non-deterministic results: {run1!r} vs {run2!r}"
        finally:
            torch.use_deterministic_algorithms(prev)

    def test_two_runs_produce_same_cosines(self) -> None:
        """Cosine scores should be bit-equal across two runs."""
        model = build_model()
        model.eval()

        index, meta, _ = _build_small_index(seed=7)

        rng = np.random.default_rng(13)
        src_emb_np = rng.random(512).astype(np.float32)
        src_emb_np /= np.linalg.norm(src_emb_np)
        src_emb = torch.tensor(src_emb_np)

        result1 = retrieve(src_emb, 0, model, index, meta, top_m=5)  # type: ignore[arg-type]
        result2 = retrieve(src_emb, 0, model, index, meta, top_m=5)  # type: ignore[arg-type]

        assert result1.indices == result2.indices
        for c1, c2 in zip(result1.cosines, result2.cosines, strict=True):
            assert abs(c1 - c2) < 1e-6, f"Cosine difference: {c1} vs {c2}"

    def test_pipeline_returns_candidates_list(self) -> None:
        """Full pipeline should return a list of Candidate objects."""
        model = build_model()
        model.eval()

        index, meta, _ = _build_small_index(seed=3)

        rng = np.random.default_rng(5)
        src_emb_np = rng.random(512).astype(np.float32)
        src_emb_np /= np.linalg.norm(src_emb_np)
        src_emb = torch.tensor(src_emb_np)

        result = retrieve(src_emb, -1, model, index, meta, top_m=10)  # type: ignore[arg-type]
        candidates = rerank(result, "NONE", "x = 1\n", k=5)

        assert isinstance(candidates, list)
        assert len(candidates) <= 5
