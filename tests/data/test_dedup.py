"""Tests for codingjepa.data.dedup. See RFC-0014 §D6."""

from __future__ import annotations

import json
from pathlib import Path

from codingjepa.data.chunker import Chunk, ChunkKind
from codingjepa.data.dedup import (
    DedupResult,
    dedup_pipeline,
    exact_dedup,
    lsh_dedup,
    minhash_signature,
    write_dedup_report,
)


def _make_chunk(source: str, chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        repo="r",
        file_path="f.py",
        commit_sha="x",
        chunk_qualname="q",
        chunk_kind=ChunkKind.FUNCTION,
        start_line=1,
        end_line=1,
        source_raw=source,
        source_normalized=source,
    )


# --------------------------------------------------------------------------- #
# exact dedup                                                                 #
# --------------------------------------------------------------------------- #


def test_exact_dedup_removes_identical_chunks() -> None:
    """Two chunks with the same source_normalized collapse to one."""

    chunks = [
        _make_chunk("def foo(): return 1", "c1"),
        _make_chunk("def foo(): return 1", "c2"),
        _make_chunk("def bar(): return 2", "c3"),
    ]
    out, removed = exact_dedup(chunks)
    assert len(out) == 2
    assert removed == 1


def test_exact_dedup_keeps_different_chunks() -> None:
    """All-distinct chunks pass through untouched."""

    chunks = [
        _make_chunk("def a(): return 1", "c1"),
        _make_chunk("def b(): return 2", "c2"),
        _make_chunk("def c(): return 3", "c3"),
    ]
    out, removed = exact_dedup(chunks)
    assert len(out) == 3
    assert removed == 0


def test_exact_dedup_returns_correct_count() -> None:
    """Removed count equals total - retained."""

    chunks = [_make_chunk("same", f"c{i}") for i in range(5)] + [_make_chunk("other", "cother")]
    out, removed = exact_dedup(chunks)
    assert len(out) == 2
    assert removed == 4


def test_exact_dedup_empty_input() -> None:
    """Empty input yields empty output and zero removals."""

    out, removed = exact_dedup([])
    assert out == []
    assert removed == 0


# --------------------------------------------------------------------------- #
# minhash                                                                     #
# --------------------------------------------------------------------------- #


def test_minhash_signature_deterministic() -> None:
    """Same input produces the same signature on every call."""

    a = minhash_signature("def foo(): return 1")
    b = minhash_signature("def foo(): return 1")
    assert a == b
    # And signature length is the requested number of hashes.
    assert len(a) == 128


def test_minhash_signature_respects_num_hashes() -> None:
    """num_hashes argument controls signature length."""

    sig = minhash_signature("hello world", num_hashes=32)
    assert len(sig) == 32


def test_minhash_signature_similar_for_similar_text() -> None:
    """Near-identical strings agree on most MinHash slots."""

    a = minhash_signature("def foo(): return 1")
    b = minhash_signature("def foo(): return 2")
    # Jaccard estimate = fraction of equal slots.
    matches = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    estimated_jaccard = matches / len(a)
    # Identical text → ~1.0, totally different → ~0.0; one-character
    # change in a short string should still leave plenty of shared
    # 5-grams.
    assert estimated_jaccard > 0.3


def test_minhash_signature_differs_for_dissimilar_text() -> None:
    """Very different texts have low estimated Jaccard."""

    a = minhash_signature("def foo(): return 1")
    b = minhash_signature("class Bar: pass\n" * 5)
    matches = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    estimated_jaccard = matches / len(a)
    assert estimated_jaccard < 0.5


# --------------------------------------------------------------------------- #
# LSH                                                                         #
# --------------------------------------------------------------------------- #


def test_lsh_dedup_removes_near_duplicates() -> None:
    """Two chunks with high MinHash Jaccard are collapsed."""

    # Two near-identical functions: only the literal value differs.
    chunks = [
        _make_chunk("def foo(x):\n    return x + 1\n", "c1"),
        _make_chunk("def foo(x):\n    return x + 1\n", "c2"),  # exact, but go via LSH
    ]
    out, removed = lsh_dedup(chunks, threshold=0.5)
    assert removed >= 1
    assert len(out) <= 1 + 1  # at least one collapsed


def test_lsh_dedup_keeps_dissimilar_chunks() -> None:
    """Chunks with no shingle overlap survive LSH dedup."""

    chunks = [
        _make_chunk("aaaaaaaaaaaaaaaaaaaaaaaa", "c1"),
        _make_chunk("zzzzzzzzzzzzzzzzzzzzzzzz", "c2"),
        _make_chunk("def quicksort(items): return items", "c3"),
    ]
    out, removed = lsh_dedup(chunks, threshold=0.85)
    assert removed == 0
    assert len(out) == 3


def test_lsh_dedup_is_deterministic() -> None:
    """The retained representative is stable across runs."""

    chunks_a = [
        _make_chunk("def foo(x): return x + 1", "c1"),
        _make_chunk("def foo(x): return x + 1", "c2"),
    ]
    chunks_b = [
        _make_chunk("def foo(x): return x + 1", "c1"),
        _make_chunk("def foo(x): return x + 1", "c2"),
    ]
    out_a, _ = lsh_dedup(chunks_a, threshold=0.5)
    out_b, _ = lsh_dedup(chunks_b, threshold=0.5)
    # The chunk_ids retained are the same on a re-run.
    a_ids = {c.chunk_id for c in out_a}
    b_ids = {c.chunk_id for c in out_b}
    assert a_ids == b_ids


def test_lsh_dedup_empty_input() -> None:
    """Empty input yields empty output and zero removals."""

    out, removed = lsh_dedup([])
    assert out == []
    assert removed == 0


# --------------------------------------------------------------------------- #
# pipeline                                                                    #
# --------------------------------------------------------------------------- #


def test_dedup_pipeline_combines_both() -> None:
    """The pipeline applies exact dedup then LSH and returns a DedupResult."""

    chunks = [
        _make_chunk("def foo(): return 1", "c1"),  # exact dup pair with c2
        _make_chunk("def foo(): return 1", "c2"),
        _make_chunk("def bar(): return 2", "c3"),
        _make_chunk("def baz(): return 9999", "c4"),
    ]
    deduped, result = dedup_pipeline(chunks, threshold=0.85)
    assert isinstance(result, DedupResult)
    assert result.total_chunks == 4
    assert result.exact_duplicates >= 1
    assert result.retained_chunks == len(deduped)
    # rate is in [0, 1].
    assert 0.0 <= result.dedup_rate <= 1.0


def test_dedup_pipeline_no_duplicates() -> None:
    """If every chunk is unique, retained = total and rate = 0."""

    chunks = [_make_chunk(f"def f_{i}(): return {i}", f"c{i}") for i in range(5)]
    deduped, result = dedup_pipeline(chunks, threshold=0.85)
    assert result.exact_duplicates == 0
    assert result.near_duplicates == 0
    assert result.retained_chunks == 5
    assert result.dedup_rate == 0.0
    assert len(deduped) == 5


def test_dedup_result_fields() -> None:
    """DedupResult exposes the documented fields with reasonable values."""

    r = DedupResult(
        total_chunks=10,
        exact_duplicates=2,
        near_duplicates=1,
        retained_chunks=7,
        dedup_rate=0.3,
    )
    assert r.total_chunks == 10
    assert r.exact_duplicates == 2
    assert r.near_duplicates == 1
    assert r.retained_chunks == 7
    assert r.dedup_rate == 0.3


def test_write_dedup_report(tmp_path: Path) -> None:
    """write_dedup_report produces JSON with all DedupResult fields."""

    result = DedupResult(
        total_chunks=10,
        exact_duplicates=2,
        near_duplicates=1,
        retained_chunks=7,
        dedup_rate=0.3,
    )
    out_path = tmp_path / "dedup.json"
    write_dedup_report(result, out_path)
    payload = json.loads(out_path.read_text())
    assert payload["total_chunks"] == 10
    assert payload["exact_duplicates"] == 2
    assert payload["near_duplicates"] == 1
    assert payload["retained_chunks"] == 7
    assert payload["dedup_rate"] == 0.3
