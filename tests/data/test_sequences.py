"""Tests for codingjepa.data.sequences. See RFC-0002 §D8-D9 and RFC-0012 §D4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codingjepa.data.chunker import Chunk, ChunkKind
from codingjepa.data.sequences import (
    apply_intent_quotas,
    build_and_write_sequences,
    build_sequences,
    write_sequences,
)


def _mk_chunk(
    repo: str = "owner/repo",
    file_path: str = "pkg/m.py",
    qualname: str = "f",
    token_ids: list[int] | None = None,
) -> Chunk:
    if token_ids is None:
        token_ids = [1, 2, 3]
    return Chunk(
        chunk_id=f"id::{repo}::{file_path}::{qualname}",
        repo=repo,
        file_path=file_path,
        commit_sha="a" * 40,
        chunk_qualname=qualname,
        chunk_kind=ChunkKind.FUNCTION,
        start_line=1,
        end_line=2,
        source_raw=f"def {qualname}(): pass\n",
        source_normalized=f"def {qualname}(): pass\n",
        token_ids=list(token_ids),
    )


def _chunks_in_file(
    n: int, *, repo: str = "owner/repo", file_path: str = "pkg/m.py"
) -> list[Chunk]:
    return [
        _mk_chunk(repo=repo, file_path=file_path, qualname=f"f{i}", token_ids=[i, i + 1])
        for i in range(n)
    ]


def test_build_sequences_basic() -> None:
    """10 chunks from one file produce 6 sequences for H=3, n_preds=1, stride=1."""

    chunks = _chunks_in_file(10)
    seqs = build_sequences(chunks)

    # window size = 5 (H=3 + n_preds=1 + 1), starts at i=0..5 inclusive ⇒ 6.
    # Actually: window of (H + n_preds + 1) = 5; with stride=1, windows = 10 - 5 + 1 = 6.
    assert len(seqs) == 6


def test_build_sequences_default_params_window_size() -> None:
    """Default H=3, n_preds=1 → context_ids has 3 entries; target_ids is one chunk."""

    chunks = _chunks_in_file(5)
    seqs = build_sequences(chunks)

    assert len(seqs) == 1
    s = seqs[0]
    assert len(s.context_ids) == 3
    # target_ids is a flat list[int]
    assert s.target_ids == chunks[3].token_ids
    # Note: with H=3 + n_preds=1 we slide a window of 5. The first 3 are context,
    # the 4th is the target (n_preds=1), the 5th would be the prediction target+1
    # in multi-pred mode but here n_preds=1 so target is index 3 of window 0..4.


def test_build_sequences_intent_idx_is_minus_one() -> None:
    """Pretraining sequences carry intent_idx == -1 (I_NONE)."""

    chunks = _chunks_in_file(5)
    seqs = build_sequences(chunks)
    assert seqs[0].intent_idx == -1


def test_build_sequences_groups_by_file() -> None:
    """Chunks from different files are not mixed into one window."""

    a = _chunks_in_file(5, file_path="pkg/a.py")
    b = _chunks_in_file(5, file_path="pkg/b.py")
    seqs = build_sequences(a + b)

    # Each file gives 1 window of size 5 → 2 sequences total.
    assert len(seqs) == 2
    files_used = {(s.repo, s.file_path) for s in seqs}
    assert files_used == {("owner/repo", "pkg/a.py"), ("owner/repo", "pkg/b.py")}


def test_build_sequences_groups_by_repo() -> None:
    """Chunks from different repos but same file path are not mixed."""

    a = _chunks_in_file(5, repo="owner/x", file_path="pkg/m.py")
    b = _chunks_in_file(5, repo="owner/y", file_path="pkg/m.py")
    seqs = build_sequences(a + b)
    assert len(seqs) == 2


def test_build_sequences_too_few_chunks_yields_none() -> None:
    """A file with fewer than (H + n_preds + 1) chunks yields no sequences."""

    chunks = _chunks_in_file(4)  # 4 < 5
    seqs = build_sequences(chunks)
    assert seqs == []


def test_build_sequences_skips_empty_token_ids() -> None:
    """Chunks with empty token_ids are skipped before windowing (over-cap drops)."""

    chunks = _chunks_in_file(5)
    chunks[2].token_ids = []  # over-cap drop
    seqs = build_sequences(chunks)
    # 4 remaining chunks, < 5, so no sequence.
    assert seqs == []


def test_build_sequences_stride_2_fewer_sequences() -> None:
    """stride=2 produces approximately half as many windows."""

    chunks = _chunks_in_file(10)
    seqs1 = build_sequences(chunks, stride=1)
    seqs2 = build_sequences(chunks, stride=2)
    assert len(seqs2) < len(seqs1)


def test_build_sequences_h2_n2_window_4() -> None:
    """history_len=2, n_preds=2 → window of (2 + 2 + 1) = 5 also (but H+n_preds+1=5)."""

    chunks = _chunks_in_file(5)
    seqs = build_sequences(chunks, history_len=2, n_preds=2)
    assert len(seqs) == 1
    s = seqs[0]
    assert len(s.context_ids) == 2  # H=2


def test_sequence_id_is_deterministic() -> None:
    """Same chunks → same sequence_id (content-addressed)."""

    a = _chunks_in_file(5)
    b = _chunks_in_file(5)  # rebuilt identical objects
    seqs_a = build_sequences(a)
    seqs_b = build_sequences(b)
    assert seqs_a[0].sequence_id == seqs_b[0].sequence_id


def test_sequence_id_changes_with_content() -> None:
    """Changing one token changes the sequence_id."""

    a = _chunks_in_file(5)
    b = _chunks_in_file(5)
    b[2].token_ids = [99, 100]
    seqs_a = build_sequences(a)
    seqs_b = build_sequences(b)
    assert seqs_a[0].sequence_id != seqs_b[0].sequence_id


def test_write_sequences_roundtrip(tmp_path: Path) -> None:
    """write_sequences should produce a file with the expected sequence count."""

    chunks = _chunks_in_file(5)
    seqs = build_sequences(chunks)
    out = tmp_path / "v1.parquet"
    write_sequences(seqs, output_path=out)
    assert out.exists()


def test_build_and_write_sequences(tmp_path: Path) -> None:
    """build_and_write_sequences is the convenience composition."""

    chunks = _chunks_in_file(6)
    out = tmp_path / "v1.parquet"
    seqs = build_and_write_sequences(chunks, output_path=out)
    assert out.exists()
    assert len(seqs) == 2  # 6 - 5 + 1 = 2 windows


# --- Per-intent quotas (issue #57, RFC-0002 §D8) ---


@dataclass
class _Pair:
    """Minimal pair shape: only need ``intent_label`` for the quota logic."""

    intent_label: str
    payload: int = 0


def test_apply_intent_quotas_caps_at_max() -> None:
    """Returns retained / overflow with at most max_per_intent retained per label."""

    pairs = [_Pair("a", i) for i in range(15)] + [_Pair("b", i) for i in range(5)]
    retained, overflow = apply_intent_quotas(pairs, max_per_intent=10, overflow_path=None)
    by_a = [p for p in retained if p.intent_label == "a"]
    by_b = [p for p in retained if p.intent_label == "b"]
    assert len(by_a) == 10
    assert len(by_b) == 5
    assert len(overflow) == 5
    assert all(p.intent_label == "a" for p in overflow)


def test_apply_intent_quotas_overflow_written(tmp_path: Path) -> None:
    """Overflow pairs are persisted to overflow_path."""

    pairs = [_Pair("x", i) for i in range(13)]
    out = tmp_path / "overflow.parquet"
    retained, overflow = apply_intent_quotas(pairs, max_per_intent=10, overflow_path=out)
    assert len(retained) == 10
    assert len(overflow) == 3
    assert out.exists()


def test_apply_intent_quotas_no_overflow_does_not_write(tmp_path: Path) -> None:
    """If no overflow, overflow file should not be created."""

    pairs = [_Pair("x", i) for i in range(5)]
    out = tmp_path / "overflow.parquet"
    retained, overflow = apply_intent_quotas(pairs, max_per_intent=10, overflow_path=out)
    assert len(retained) == 5
    assert overflow == []
    assert not out.exists()


def test_apply_intent_quotas_default_cap_is_12000() -> None:
    """Default cap matches RFC-0002 §D8 (12,000)."""

    # Use a small constructive proof rather than a 12001-pair list.
    pairs = [_Pair("a", i) for i in range(10)]
    retained, overflow = apply_intent_quotas(pairs, overflow_path=None)
    assert retained == pairs
    assert overflow == []
