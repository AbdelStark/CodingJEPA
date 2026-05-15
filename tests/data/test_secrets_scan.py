"""Tests for codingjepa.data.secrets_scan. See RFC-0014 §D5."""

from __future__ import annotations

from dataclasses import dataclass

from codingjepa.data.secrets_scan import (
    SecretHit,
    scan_chunk,
    scan_chunks,
    scan_report,
)


@dataclass
class _FakeChunk:
    chunk_id: str
    source_normalized: str


# ---------------------------------------------------------------------------
# scan_chunk
# ---------------------------------------------------------------------------


def test_scan_detects_aws_key() -> None:
    """An AWS access-key id in the source produces a hit."""

    src = 'AWS = "AKIAIOSFODNN7EXAMPLE"\n'
    hits = scan_chunk(src, chunk_id="c0")
    assert len(hits) >= 1
    assert any("aws" in h.pattern_name.lower() for h in hits)
    for h in hits:
        assert h.chunk_id == "c0"


def test_scan_detects_github_pat() -> None:
    """A GitHub Personal Access Token (ghp_) is flagged."""

    src = "TOKEN = 'ghp_" + "a" * 36 + "'\n"
    hits = scan_chunk(src, chunk_id="c1")
    assert any("github" in h.pattern_name.lower() for h in hits)


def test_scan_detects_private_key() -> None:
    """A PEM-style private-key header is flagged."""

    src = (
        "KEY = '''-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "data\n"
        "-----END OPENSSH PRIVATE KEY-----'''\n"
    )
    hits = scan_chunk(src, chunk_id="c2")
    assert any("ssh" in h.pattern_name.lower() or "private" in h.pattern_name.lower() for h in hits)


def test_scan_clean_chunk() -> None:
    """Normal Python code with no embedded secrets produces no hits."""

    src = "def foo(x):\n    return x + 1\n"
    hits = scan_chunk(src, chunk_id="c-clean")
    assert hits == []


def test_scan_hit_preview_is_redacted() -> None:
    """The match preview never returns the entire matched secret verbatim."""

    src = "TOKEN = 'ghp_" + "a" * 36 + "'\n"
    hits = scan_chunk(src, chunk_id="c3")
    assert hits
    for h in hits:
        # Bound to 20 chars per spec, so the full 40-char token is never present.
        assert len(h.match_preview) <= 20


# ---------------------------------------------------------------------------
# scan_chunks
# ---------------------------------------------------------------------------


def test_scan_chunks_removes_hit_chunks() -> None:
    """Chunks containing any secret hit are removed from the returned clean list."""

    clean = _FakeChunk("ok", "def foo(): return 1\n")
    dirty = _FakeChunk("dirty", "k = 'AKIAIOSFODNN7EXAMPLE'\n")
    kept, hits = scan_chunks([clean, dirty])
    ids = {c.chunk_id for c in kept}
    assert "ok" in ids
    assert "dirty" not in ids
    assert any(h.chunk_id == "dirty" for h in hits)


def test_scan_chunks_no_hits() -> None:
    """All-clean input passes through and returns no hits."""

    chunks = [
        _FakeChunk("a", "def foo(): return 1\n"),
        _FakeChunk("b", "class B:\n    pass\n"),
    ]
    kept, hits = scan_chunks(chunks)
    assert {c.chunk_id for c in kept} == {"a", "b"}
    assert hits == []


# ---------------------------------------------------------------------------
# scan_report
# ---------------------------------------------------------------------------


def test_scan_report_counts() -> None:
    """The report aggregates total hits, per-pattern counts, and dropped chunk count."""

    hits = [
        SecretHit(chunk_id="a", pattern_name="aws_access_key_id", match_preview="AKIA"),
        SecretHit(chunk_id="a", pattern_name="github_pat", match_preview="ghp_"),
        SecretHit(chunk_id="b", pattern_name="aws_access_key_id", match_preview="AKIA"),
    ]
    report = scan_report(hits)
    assert report["total_hits"] == 3
    assert report["by_pattern"]["aws_access_key_id"] == 2
    assert report["by_pattern"]["github_pat"] == 1
    # Distinct chunk ids in the hits set.
    assert report["chunks_dropped"] == 2


def test_scan_report_empty() -> None:
    """Zero hits → zero counts, empty by_pattern."""

    report = scan_report([])
    assert report["total_hits"] == 0
    assert report["by_pattern"] == {}
    assert report["chunks_dropped"] == 0
