"""Tests for codingjepa.inference.index (#83, #90)."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from codingjepa.errors import IndexHashMismatch
from codingjepa.inference.index import IndexMeta, build_index, load_index


def _make_embeddings(n: int = 10, d: int = 64, seed: int = 42) -> np.ndarray:
    """Create L2-normalized float32 embeddings."""
    rng = np.random.default_rng(seed)
    embs = rng.random((n, d)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


def _make_meta(n: int = 10, index_id: str = "abcd1234-ef567890") -> IndexMeta:
    return IndexMeta(
        chunk_ids=[f"chunk_{i}" for i in range(n)],
        sources=[f"def f{i}(): pass\n" for i in range(n)],
        index_id=index_id,
    )


class TestBuildAndLoadIndex:
    def test_build_creates_faiss_file(self) -> None:
        embs = _make_embeddings(10, 64)
        meta = _make_meta(10)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            faiss_path = build_index(embs, meta, out_dir)
            assert faiss_path.exists()
            assert faiss_path.suffix == ".faiss"

    def test_build_creates_meta_json(self) -> None:
        embs = _make_embeddings(10, 64)
        meta = _make_meta(10, index_id="aa11bb22-cc33dd44")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            build_index(embs, meta, out_dir)
            meta_file = out_dir / "aa11bb22-cc33dd44.meta.json"
            assert meta_file.exists()

    def test_load_ntotal_matches(self) -> None:
        n = 10
        embs = _make_embeddings(n, 64)
        meta = _make_meta(n)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            build_index(embs, meta, out_dir)
            idx, loaded_meta = load_index("abcd1234-ef567890", out_dir)
            assert idx.ntotal == n

    def test_load_meta_chunk_ids_preserved(self) -> None:
        embs = _make_embeddings(5, 32)
        meta = _make_meta(5, index_id="11223344-55667788")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            build_index(embs, meta, out_dir)
            _, loaded_meta = load_index("11223344-55667788", out_dir)
            assert loaded_meta.chunk_ids == meta.chunk_ids

    def test_load_meta_sources_preserved(self) -> None:
        embs = _make_embeddings(5, 32)
        meta = _make_meta(5, index_id="aabbccdd-11223344")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            build_index(embs, meta, out_dir)
            _, loaded_meta = load_index("aabbccdd-11223344", out_dir)
            assert loaded_meta.sources == meta.sources

    def test_round_trip_search(self) -> None:
        """A query against the loaded index returns a valid index."""
        n, d = 10, 64
        embs = _make_embeddings(n, d)
        meta = _make_meta(n, index_id="12345678-abcdef01")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            build_index(embs, meta, out_dir)
            idx, _ = load_index("12345678-abcdef01", out_dir)
            query = embs[0:1]  # (1, D)
            distances, indices = idx.search(query, 3)
            assert indices[0][0] == 0  # exact match at top

    def test_out_dir_created_if_missing(self) -> None:
        embs = _make_embeddings(3, 16)
        meta = _make_meta(3, index_id="deadbeef-cafebabe")
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = pathlib.Path(tmpdir) / "deep" / "nested"
            build_index(embs, meta, nested)
            assert (nested / "deadbeef-cafebabe.faiss").exists()


class TestIndexIdFormat:
    def test_index_id_in_filenames(self) -> None:
        index_id = "a1b2c3d4-e5f60718"
        embs = _make_embeddings(3, 16)
        meta = IndexMeta(
            chunk_ids=["c0", "c1", "c2"],
            sources=["s0", "s1", "s2"],
            index_id=index_id,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            faiss_path = build_index(embs, meta, out_dir)
            assert index_id in faiss_path.name
            meta_path = out_dir / f"{index_id}.meta.json"
            assert meta_path.exists()


class TestLoadIndexHashMismatch:
    def test_wrong_checkpoint_hash_raises(self) -> None:
        embs = _make_embeddings(5, 32)
        meta = _make_meta(5, index_id="aabbccdd-11223344")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            build_index(embs, meta, out_dir)
            with pytest.raises(IndexHashMismatch):
                load_index(
                    "aabbccdd-11223344",
                    out_dir,
                    expected_checkpoint_hash="xxxxxxxx",
                )

    def test_wrong_manifest_hash_raises(self) -> None:
        embs = _make_embeddings(5, 32)
        meta = _make_meta(5, index_id="aabbccdd-11223344")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            build_index(embs, meta, out_dir)
            with pytest.raises(IndexHashMismatch):
                load_index(
                    "aabbccdd-11223344",
                    out_dir,
                    expected_manifest_hash="yyyyyyyy",
                )

    def test_correct_hash_prefix_passes(self) -> None:
        embs = _make_embeddings(5, 32)
        index_id = "aabbccdd-11223344"
        meta = _make_meta(5, index_id=index_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            build_index(embs, meta, out_dir)
            # Provide the correct prefix (first 8 chars of first segment)
            idx, loaded_meta = load_index(
                index_id,
                out_dir,
                expected_checkpoint_hash="aabbccdd",
                expected_manifest_hash="11223344",
            )
            assert idx.ntotal == 5

    def test_hash_mismatch_error_code(self) -> None:
        embs = _make_embeddings(3, 16)
        meta = _make_meta(3, index_id="12345678-abcdef01")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            build_index(embs, meta, out_dir)
            with pytest.raises(IndexHashMismatch) as exc_info:
                load_index(
                    "12345678-abcdef01",
                    out_dir,
                    expected_checkpoint_hash="00000000",
                )
            assert exc_info.value.code == "E_INDEX_HASH_MISMATCH"
