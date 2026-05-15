"""Chunker for Python source files (RFC-0012 §D1-D3, D11).

The chunker parses a Python module with libcst, walks the top-level
statements, and emits one :class:`Chunk` per

* ``FunctionDef`` -> :data:`ChunkKind.FUNCTION`
* ``AsyncFunctionDef`` -> :data:`ChunkKind.ASYNC_FUNCTION`
* ``ClassDef`` -> :data:`ChunkKind.CLASS`
* runs of other top-level statements -> :data:`ChunkKind.INTERSTITIAL`

Each chunk carries a deterministic ``chunk_id`` (sha256 over
``repo:file_path:qualname:source_normalized``) so the same chunk in the
same revision of the same file always hashes the same. Chunks whose
:func:`codingjepa.data.normalize.normalize_chunk` output is ``None``
(i.e. the chunk did not compile after normalization) are dropped.

This module never raises on bad input: parse failures yield an empty
chunk list so callers can keep streaming files without a try/except
around every call.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import libcst as cst
import libcst.metadata as cst_metadata

from codingjepa.data.normalize import normalize_chunk


class ChunkKind(StrEnum):
    """The four chunk shapes produced by :func:`chunk_file`."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    INTERSTITIAL = "interstitial"


@dataclass
class Chunk:
    """A single normalized chunk extracted from a Python file.

    ``chunk_id`` is ``sha256(repo + ":" + file_path + ":" + qualname + ":" + source_normalized)``
    and is the canonical join key downstream (RFC-0012 §D11).
    """

    chunk_id: str
    repo: str
    file_path: str
    commit_sha: str
    chunk_qualname: str
    chunk_kind: ChunkKind
    start_line: int
    end_line: int
    source_raw: str
    source_normalized: str
    token_ids: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# low-level structural walk                                                   #
# --------------------------------------------------------------------------- #


_DEF_NODES = (cst.FunctionDef, cst.ClassDef)


def _kind_for(node: cst.CSTNode) -> ChunkKind:
    if isinstance(node, cst.FunctionDef):
        return ChunkKind.ASYNC_FUNCTION if node.asynchronous is not None else ChunkKind.FUNCTION
    if isinstance(node, cst.ClassDef):
        return ChunkKind.CLASS
    raise TypeError(f"not a definition node: {type(node).__name__}")


def _qualname_for(node: cst.FunctionDef | cst.ClassDef, parents: list[str]) -> str:
    """Build a dotted qualname from the enclosing class chain plus this name."""

    return ".".join(parents + [node.name.value])


def _collect_nested_definitions(
    node: cst.ClassDef, parents: list[str]
) -> list[tuple[str, str, ChunkKind, int, int, cst.CSTNode]]:
    """Recursively enumerate methods and inner classes inside a ClassDef.

    Returns ``(qualname, source_text, kind, start_line, end_line, node)`` tuples
    for every nested ``FunctionDef``, ``AsyncFunctionDef``, and ``ClassDef``
    found in the class body. The class itself is *not* included; the caller
    adds it separately.
    """

    out: list[tuple[str, str, ChunkKind, int, int, cst.CSTNode]] = []
    new_parents = parents + [node.name.value]
    if not isinstance(node.body, cst.IndentedBlock):
        return out
    for inner in node.body.body:
        if isinstance(inner, _DEF_NODES):
            kind = _kind_for(inner)
            qualname = _qualname_for(inner, new_parents)
            out.append((qualname, "", kind, 0, 0, inner))
            if isinstance(inner, cst.ClassDef):
                out.extend(_collect_nested_definitions(inner, new_parents))
    return out


def chunk_source_text(
    source: str,
) -> list[tuple[str, str, ChunkKind, int, int]]:
    """Low-level chunker: return ``(qualname, source_text, kind, start, end)`` tuples.

    Top-level definitions (functions, async functions, classes) become one
    tuple each; runs of consecutive non-definition statements become a single
    ``INTERSTITIAL`` tuple. Methods and inner classes are emitted in addition
    to their enclosing class, with dotted qualnames.

    Returns ``[]`` if the source cannot be parsed.
    """

    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return []

    wrapper = cst_metadata.MetadataWrapper(module, unsafe_skip_copy=True)
    positions = wrapper.resolve(cst_metadata.PositionProvider)
    # Use the wrapped module so position metadata keys line up.
    module = wrapper.module

    out: list[tuple[str, str, ChunkKind, int, int]] = []
    interstitial_buffer: list[cst.BaseStatement] = []
    interstitial_start: int | None = None
    interstitial_end: int = 0

    def flush_interstitial() -> None:
        nonlocal interstitial_buffer, interstitial_start, interstitial_end
        if not interstitial_buffer:
            return
        # Reconstruct the source slice for this run of statements.
        text = "".join(module.code_for_node(s) for s in interstitial_buffer)
        out.append(
            (
                "<interstitial>",
                text,
                ChunkKind.INTERSTITIAL,
                interstitial_start or 1,
                interstitial_end,
            )
        )
        interstitial_buffer = []
        interstitial_start = None
        interstitial_end = 0

    for stmt in module.body:
        pos = positions[stmt]
        start_line = pos.start.line
        end_line = pos.end.line

        if isinstance(stmt, _DEF_NODES):
            flush_interstitial()
            kind = _kind_for(stmt)
            qualname = _qualname_for(stmt, [])
            text = module.code_for_node(stmt)
            out.append((qualname, text, kind, start_line, end_line))
            # Recurse into class bodies for nested defs.
            if isinstance(stmt, cst.ClassDef):
                for q, _src, k, _s, _e, sub_node in _collect_nested_definitions(stmt, []):
                    sub_pos = positions[sub_node]
                    sub_text = module.code_for_node(sub_node)
                    out.append(
                        (
                            q,
                            sub_text,
                            k,
                            sub_pos.start.line,
                            sub_pos.end.line,
                        )
                    )
        else:
            interstitial_buffer.append(stmt)
            if interstitial_start is None:
                interstitial_start = start_line
            interstitial_end = end_line

    flush_interstitial()
    return out


# --------------------------------------------------------------------------- #
# high-level chunk_file / chunk_repo                                          #
# --------------------------------------------------------------------------- #


def _make_chunk_id(repo: str, file_path: str, qualname: str, source_normalized: str) -> str:
    """sha256 of the canonical join key (RFC-0012 §D11)."""

    payload = f"{repo}:{file_path}:{qualname}:{source_normalized}".encode()
    return hashlib.sha256(payload).hexdigest()


def chunk_file(
    source: str,
    *,
    repo: str,
    file_path: str,
    commit_sha: str,
) -> list[Chunk]:
    """Parse a Python source file and extract all chunks.

    Returns :class:`Chunk` objects for each FunctionDef, AsyncFunctionDef,
    ClassDef, and interstitial block. Chunks where
    :func:`codingjepa.data.normalize.normalize_chunk` returns ``None`` (i.e.
    the chunk does not compile after normalization) are dropped.

    Returns ``[]`` if the source cannot be parsed (RFC-0012 §D1: the chunker
    must be robust to malformed input from upstream mirrors).
    """

    raw_chunks = chunk_source_text(source)
    out: list[Chunk] = []
    for qualname, source_text, kind, start_line, end_line in raw_chunks:
        normalized = normalize_chunk(source_text)
        if normalized is None:
            continue
        if normalized.strip() == "":
            # An interstitial block that contained only whitespace / pragmas.
            continue
        chunk_id = _make_chunk_id(repo, file_path, qualname, normalized)
        out.append(
            Chunk(
                chunk_id=chunk_id,
                repo=repo,
                file_path=file_path,
                commit_sha=commit_sha,
                chunk_qualname=qualname,
                chunk_kind=kind,
                start_line=start_line,
                end_line=end_line,
                source_raw=source_text,
                source_normalized=normalized,
                token_ids=[],
            )
        )
    return out


def chunk_repo(
    repo_dir: Path,
    repo_name: str,
    commit_sha: str,
    *,
    py_files: list[Path] | None = None,
) -> Iterator[Chunk]:
    """Chunk all Python files in a repo directory.

    Yields chunks lazily; callers can stream into a writer without buffering
    the whole repo in memory. ``py_files`` overrides the directory walk and
    restricts processing to the given paths.

    Paths in the emitted :class:`Chunk` objects are recorded relative to
    ``repo_dir`` so the same chunk_id is reproducible from a different
    checkout location.
    """

    repo_dir = Path(repo_dir)
    if py_files is None:
        py_files = sorted(repo_dir.rglob("*.py"))

    for path in py_files:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            rel = path.resolve().relative_to(repo_dir.resolve())
            rel_str = str(rel)
        except ValueError:
            rel_str = str(path)
        yield from chunk_file(
            source,
            repo=repo_name,
            file_path=rel_str,
            commit_sha=commit_sha,
        )


__all__ = [
    "Chunk",
    "ChunkKind",
    "chunk_file",
    "chunk_repo",
    "chunk_source_text",
]
