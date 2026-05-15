"""Tests for codingjepa.inference.retrieve (#84)."""

from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import pytest
import torch

from codingjepa.inference.index import IndexMeta, build_index, load_index
from codingjepa.inference.retrieve import RetrievalResult, retrieve
from codingjepa.model import build_model

# Suppress OpenMP duplicate-lib warning on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _make_index(n: int = 10, d: int = 512, seed: int = 99) -> tuple[object, IndexMeta, np.ndarray]:
    """Return (faiss_index, meta, embeddings) with random L2-normalized vectors."""
    import faiss

    rng = np.random.default_rng(seed)
    embs = rng.random((n, d)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    meta = IndexMeta(
        chunk_ids=[f"c{i}" for i in range(n)],
        sources=[f"def f{i}(): return {i}\n" for i in range(n)],
        index_id="aabb1234-ccdd5678",
    )
    index = faiss.IndexFlatIP(d)
    index.add(embs)  # type: ignore[arg-type]
    return index, meta, embs


@pytest.fixture(scope="module")
def small_model() -> object:
    """Small CodingJEPA model for retrieval tests."""
    return build_model()


class TestRetrieveReturnsResult:
    def test_returns_retrieval_result(self, small_model: object) -> None:
        index, meta, embs = _make_index(10, 512)
        src_emb = torch.tensor(embs[0])
        result = retrieve(src_emb, -1, small_model, index, meta, top_m=5)  # type: ignore[arg-type]
        assert isinstance(result, RetrievalResult)

    def test_has_indices_and_cosines(self, small_model: object) -> None:
        index, meta, embs = _make_index(10, 512)
        src_emb = torch.tensor(embs[0])
        result = retrieve(src_emb, -1, small_model, index, meta, top_m=5)  # type: ignore[arg-type]
        assert isinstance(result.indices, list)
        assert isinstance(result.cosines, list)

    def test_indices_and_cosines_same_length(self, small_model: object) -> None:
        index, meta, embs = _make_index(10, 512)
        src_emb = torch.tensor(embs[2])
        result = retrieve(src_emb, 0, small_model, index, meta, top_m=5)  # type: ignore[arg-type]
        assert len(result.indices) == len(result.cosines)

    def test_meta_preserved(self, small_model: object) -> None:
        index, meta, embs = _make_index(10, 512)
        src_emb = torch.tensor(embs[0])
        result = retrieve(src_emb, -1, small_model, index, meta, top_m=3)  # type: ignore[arg-type]
        assert result.meta is meta

    def test_intent_none_works(self, small_model: object) -> None:
        index, meta, embs = _make_index(10, 512)
        src_emb = torch.tensor(embs[0])
        # intent_idx=-1 → NONE
        result = retrieve(src_emb, -1, small_model, index, meta, top_m=5)  # type: ignore[arg-type]
        assert len(result.indices) > 0

    def test_intent_real_works(self, small_model: object) -> None:
        index, meta, embs = _make_index(10, 512)
        src_emb = torch.tensor(embs[0])
        # intent_idx=0 → "extract-helper"
        result = retrieve(src_emb, 0, small_model, index, meta, top_m=5)  # type: ignore[arg-type]
        assert len(result.indices) > 0


class TestRetrieveTopM:
    def test_top_m_hard_cap(self, small_model: object) -> None:
        """Results should not exceed top_m."""
        n, top_m = 20, 7
        index, meta, embs = _make_index(n, 512, seed=77)
        src_emb = torch.tensor(embs[0])
        result = retrieve(src_emb, -1, small_model, index, meta, top_m=top_m)  # type: ignore[arg-type]
        assert len(result.indices) <= top_m

    def test_top_m_larger_than_index(self, small_model: object) -> None:
        """top_m > index size should still return all index entries."""
        n = 5
        index, meta, embs = _make_index(n, 512, seed=33)
        src_emb = torch.tensor(embs[0])
        result = retrieve(src_emb, -1, small_model, index, meta, top_m=100)  # type: ignore[arg-type]
        assert len(result.indices) == n

    def test_top_m_one(self, small_model: object) -> None:
        index, meta, embs = _make_index(10, 512, seed=11)
        src_emb = torch.tensor(embs[0])
        result = retrieve(src_emb, -1, small_model, index, meta, top_m=1)  # type: ignore[arg-type]
        assert len(result.indices) == 1

    def test_indices_valid_range(self, small_model: object) -> None:
        n = 10
        index, meta, embs = _make_index(n, 512, seed=55)
        src_emb = torch.tensor(embs[0])
        result = retrieve(src_emb, -1, small_model, index, meta, top_m=10)  # type: ignore[arg-type]
        for i in result.indices:
            assert 0 <= i < n

    def test_persist_through_build_load(self, small_model: object) -> None:
        """Retrieve works with an index built via build_index and loaded via load_index."""
        n, d = 8, 512
        rng = np.random.default_rng(7)
        embs = rng.random((n, d)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        meta = IndexMeta(
            chunk_ids=[f"x{i}" for i in range(n)],
            sources=[f"x = {i}\n" for i in range(n)],
            index_id="00112233-44556677",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            build_index(embs, meta, out_dir)
            idx, loaded_meta = load_index("00112233-44556677", out_dir)
            src_emb = torch.tensor(embs[0])
            result = retrieve(src_emb, -1, small_model, idx, loaded_meta, top_m=5)  # type: ignore[arg-type]
        assert len(result.indices) == 5
