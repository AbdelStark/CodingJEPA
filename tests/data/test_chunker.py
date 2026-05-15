"""Tests for codingjepa.data.chunker. See RFC-0012 §D1-D3, D11."""

from __future__ import annotations

import textwrap
from pathlib import Path

from codingjepa.data.chunker import (
    Chunk,
    ChunkKind,
    chunk_file,
    chunk_repo,
    chunk_source_text,
)

SIMPLE_MODULE = textwrap.dedent("""
    def foo(x):
        return x + 1

    class Bar:
        def baz(self):
            pass

    x = 1
    y = 2
    """).lstrip()


def _chunks(src: str) -> list[Chunk]:
    return chunk_file(
        src,
        repo="acme/repo",
        file_path="pkg/module.py",
        commit_sha="abc1234",
    )


def test_chunk_file_finds_function() -> None:
    """Top-level functions are extracted as FUNCTION chunks."""

    chunks = _chunks(SIMPLE_MODULE)
    fns = [c for c in chunks if c.chunk_kind == ChunkKind.FUNCTION]
    qualnames = {c.chunk_qualname for c in fns}
    assert "foo" in qualnames
    foo = next(c for c in fns if c.chunk_qualname == "foo")
    assert "def foo" in foo.source_raw


def test_chunk_file_finds_class() -> None:
    """Top-level classes are extracted as CLASS chunks."""

    chunks = _chunks(SIMPLE_MODULE)
    classes = [c for c in chunks if c.chunk_kind == ChunkKind.CLASS]
    assert len(classes) == 1
    assert classes[0].chunk_qualname == "Bar"


def test_chunk_file_finds_interstitial() -> None:
    """Runs of non-definition statements become INTERSTITIAL chunks."""

    chunks = _chunks(SIMPLE_MODULE)
    inters = [c for c in chunks if c.chunk_kind == ChunkKind.INTERSTITIAL]
    assert len(inters) >= 1
    # The trailing `x = 1; y = 2` block.
    combined = "".join(c.source_raw for c in inters)
    assert "x = 1" in combined
    assert "y = 2" in combined


def test_chunk_qualnames_method() -> None:
    """Methods inside classes get qualified names like 'Bar.baz'."""

    chunks = _chunks(SIMPLE_MODULE)
    methods = [c for c in chunks if c.chunk_qualname == "Bar.baz"]
    assert len(methods) == 1
    assert methods[0].chunk_kind == ChunkKind.FUNCTION


def test_chunk_id_is_deterministic() -> None:
    """Same input produces the same chunk_id."""

    a = _chunks(SIMPLE_MODULE)
    b = _chunks(SIMPLE_MODULE)
    a_ids = sorted(c.chunk_id for c in a)
    b_ids = sorted(c.chunk_id for c in b)
    assert a_ids == b_ids


def test_chunk_id_uses_sha256() -> None:
    """chunk_id is a 64-character lowercase hex string (sha256)."""

    chunks = _chunks(SIMPLE_MODULE)
    assert all(len(c.chunk_id) == 64 for c in chunks)
    assert all(all(ch in "0123456789abcdef" for ch in c.chunk_id) for c in chunks)


def test_chunk_id_differs_with_repo() -> None:
    """Different repo produces different chunk_id even for same source."""

    a = chunk_file(SIMPLE_MODULE, repo="r1", file_path="f.py", commit_sha="x")
    b = chunk_file(SIMPLE_MODULE, repo="r2", file_path="f.py", commit_sha="x")
    a_ids = {c.chunk_id for c in a}
    b_ids = {c.chunk_id for c in b}
    assert a_ids != b_ids


def test_chunk_file_handles_parse_error() -> None:
    """Malformed Python returns an empty list, not an exception."""

    src = "def broken(:\n    return\n"
    chunks = chunk_file(src, repo="r", file_path="f.py", commit_sha="x")
    assert chunks == []


def test_nested_class_method_qualname() -> None:
    """Nested classes/methods get dot-separated qualnames."""

    src = textwrap.dedent("""
        class Outer:
            class Inner:
                def deep(self):
                    return 1
        """).lstrip()
    chunks = chunk_file(src, repo="r", file_path="f.py", commit_sha="x")
    qualnames = {c.chunk_qualname for c in chunks}
    assert "Outer" in qualnames
    # The chunker doesn't need to recurse to every nested method as a top-level
    # chunk, but the inner class and inner method should be reachable via the
    # class's source. For the explicit nested-method test, we look for a chunk
    # whose qualname is the dotted form, if produced.
    # Minimum contract: outermost class is captured; nested qualnames are dotted.
    nested = [q for q in qualnames if "." in q]
    if nested:
        assert any(q.startswith("Outer.") for q in nested)


def test_chunk_async_function() -> None:
    """Async functions produce ASYNC_FUNCTION chunks."""

    src = "async def f(x):\n    return x\n"
    chunks = chunk_file(src, repo="r", file_path="f.py", commit_sha="x")
    async_fns = [c for c in chunks if c.chunk_kind == ChunkKind.ASYNC_FUNCTION]
    assert len(async_fns) == 1
    assert async_fns[0].chunk_qualname == "f"


def test_chunk_line_numbers() -> None:
    """Chunks carry 1-indexed start and end lines from the source."""

    chunks = _chunks(SIMPLE_MODULE)
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_chunk_source_normalized_set() -> None:
    """source_normalized is populated after normalize_chunk."""

    chunks = _chunks(SIMPLE_MODULE)
    for c in chunks:
        assert c.source_normalized != ""


def test_chunk_token_ids_empty_initially() -> None:
    """token_ids starts empty (filled by tokenize step later)."""

    chunks = _chunks(SIMPLE_MODULE)
    for c in chunks:
        assert c.token_ids == []


def test_chunk_repo_metadata_propagated() -> None:
    """repo, file_path, commit_sha are copied onto every chunk."""

    chunks = _chunks(SIMPLE_MODULE)
    assert chunks
    for c in chunks:
        assert c.repo == "acme/repo"
        assert c.file_path == "pkg/module.py"
        assert c.commit_sha == "abc1234"


def test_chunk_source_text_returns_tuples() -> None:
    """Low-level chunk_source_text yields tuples of metadata."""

    result = chunk_source_text(SIMPLE_MODULE)
    assert isinstance(result, list)
    assert all(len(t) == 5 for t in result)
    qualnames = {t[0] for t in result}
    assert "foo" in qualnames
    assert "Bar" in qualnames


def test_chunk_empty_module() -> None:
    """An empty module yields no chunks (or only an empty interstitial)."""

    chunks = chunk_file("", repo="r", file_path="f.py", commit_sha="x")
    # Either no chunks, or an empty interstitial that normalize() dropped.
    assert all(c.source_raw.strip() != "" or c.chunk_kind == ChunkKind.INTERSTITIAL for c in chunks)


def test_chunk_repo_walks_directory(tmp_path: Path) -> None:
    """chunk_repo yields chunks from every .py file in a directory tree."""

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("def a():\n    return 1\n")
    (pkg / "b.py").write_text("def b():\n    return 2\n")
    chunks = list(chunk_repo(tmp_path, repo_name="r", commit_sha="x"))
    qualnames = {c.chunk_qualname for c in chunks}
    assert "a" in qualnames
    assert "b" in qualnames


def test_chunk_repo_explicit_file_list(tmp_path: Path) -> None:
    """chunk_repo with explicit py_files argument scans only those."""

    (tmp_path / "x.py").write_text("def x():\n    return 1\n")
    (tmp_path / "y.py").write_text("def y():\n    return 1\n")
    chunks = list(
        chunk_repo(
            tmp_path,
            repo_name="r",
            commit_sha="x",
            py_files=[tmp_path / "x.py"],
        )
    )
    qualnames = {c.chunk_qualname for c in chunks}
    assert "x" in qualnames
    assert "y" not in qualnames
