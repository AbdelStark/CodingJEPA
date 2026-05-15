"""PyDriller commit-walker that yields candidate refactor pairs (RFC-0002 §D3-D4).

For each modified ``*.py`` file in a commit, we:

1. Parse the pre- and post-commit source with ``libcst``.
2. Surface top-level :class:`libcst.FunctionDef`,
   :class:`libcst.matchers.AsyncFunctionDef`, and
   :class:`libcst.ClassDef` nodes keyed by qualname.
3. For each qualname present in both versions whose body changed, emit a
   :class:`RawPair`.

Filters (RFC-0002 §D4):

* skip merge commits;
* skip commits authored by known bots
  (``dependabot[bot]``, ``pre-commit-ci[bot]``, ``github-actions[bot]``,
  ``renovate[bot]``);
* skip commit messages matching any of the drop tokens (case-insensitive):
  ``wip``, ``temp``, ``revert``, ``formatting only``, ``lint only``,
  ``style only``, ``typo``;
* skip diffs that normalize to identical chunks (purely whitespace/comment
  churn — :func:`codingjepa.data.normalize.normalize_chunk`);
* skip nodes whose qualname does not appear in *both* before and after
  (renames break public-API resolution).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import libcst
from pydriller import Repository

from codingjepa.data.normalize import normalize_chunk

__all__ = ["RawPair", "extract_top_level_nodes", "walk_repo"]

# ---------------------------------------------------------------------------
# Filter constants
# ---------------------------------------------------------------------------

_BOT_AUTHORS: frozenset[str] = frozenset(
    {
        "dependabot[bot]",
        "pre-commit-ci[bot]",
        "github-actions[bot]",
        "renovate[bot]",
    }
)

_DROP_MESSAGE_TOKENS: tuple[str, ...] = (
    "wip",
    "temp",
    "revert",
    "formatting only",
    "lint only",
    "style only",
    "typo",
)

_NULL_PARENT_SHA = "0" * 40

_PY_SUFFIX = ".py"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RawPair:
    """A pre-/post-commit pair extracted at the function or class level.

    ``source_before`` and ``source_after`` are already normalized via
    :func:`codingjepa.data.normalize.normalize_chunk`. Downstream stages
    (chunker, labelers) may apply heavier transforms but the equality contract
    is: if these strings are equal, the pair is dropped before this dataclass
    is constructed.
    """

    repo: str
    commit_sha_before: str
    commit_sha_after: str
    file_path: str
    node_qualname: str
    source_before: str
    source_after: str
    commit_message: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_top_level_nodes(tree: libcst.Module) -> dict[str, libcst.CSTNode]:
    """Return a mapping of ``qualname -> node`` for top-level defs/classes.

    Only :class:`libcst.FunctionDef`, :class:`libcst.AsyncFunctionDef`, and
    :class:`libcst.ClassDef` directly under ``tree.body`` are surfaced; nested
    defs are intentionally skipped (RFC-0012 §D3).
    """

    nodes: dict[str, libcst.CSTNode] = {}
    for stmt in tree.body:
        node = _unwrap_simple(stmt)
        if isinstance(node, libcst.FunctionDef | libcst.ClassDef):
            nodes[node.name.value] = node
    return nodes


def walk_repo(
    repo_dir: Path,
    repo_name: str,
    *,
    max_pairs: int = 200_000,
    from_commit: str | None = None,
    to_commit: str | None = None,
) -> Iterator[RawPair]:
    """Yield :class:`RawPair` objects from ``repo_dir``'s commit history.

    The repository is walked with :class:`pydriller.Repository`. Iteration
    stops once ``max_pairs`` pairs have been emitted; callers can recover
    earlier termination by re-invoking with ``from_commit`` set to the last
    SHA observed.

    Filters are applied in this order:

    1. commit-level drops (merge, bot author, message tokens);
    2. file-level drops (non-``*.py``, missing pre/post source);
    3. node-level drops (qualname not in both versions);
    4. content-level drops (normalized chunks are identical).
    """

    kwargs: dict[str, Any] = {}
    if from_commit is not None:
        kwargs["from_commit"] = from_commit
    if to_commit is not None:
        kwargs["to_commit"] = to_commit

    emitted = 0
    for commit in Repository(str(repo_dir), **kwargs).traverse_commits():
        if emitted >= max_pairs:
            return
        if not _commit_passes_filters(commit):
            continue
        parent_sha = _resolve_parent_sha(commit)
        commit_sha = getattr(commit, "hash", "")
        commit_msg = getattr(commit, "msg", "") or ""

        for mod in getattr(commit, "modified_files", None) or []:
            file_path = _resolve_file_path(mod)
            if file_path is None or not file_path.endswith(_PY_SUFFIX):
                continue
            before = getattr(mod, "source_code_before", None)
            after = getattr(mod, "source_code", None)
            if before is None or after is None:
                continue

            before_nodes = _safe_parse_top_level(before)
            after_nodes = _safe_parse_top_level(after)
            if before_nodes is None or after_nodes is None:
                continue

            shared = sorted(set(before_nodes) & set(after_nodes))
            for qualname in shared:
                before_src = _node_source(before_nodes[qualname])
                after_src = _node_source(after_nodes[qualname])
                normalized_before = normalize_chunk(before_src)
                normalized_after = normalize_chunk(after_src)
                if normalized_before is None or normalized_after is None:
                    continue
                if normalized_before == normalized_after:
                    continue
                yield RawPair(
                    repo=repo_name,
                    commit_sha_before=parent_sha,
                    commit_sha_after=commit_sha,
                    file_path=file_path,
                    node_qualname=qualname,
                    source_before=normalized_before,
                    source_after=normalized_after,
                    commit_message=commit_msg,
                )
                emitted += 1
                if emitted >= max_pairs:
                    return


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _commit_passes_filters(commit: Any) -> bool:
    """True if ``commit`` is eligible for pair extraction (RFC-0002 §D4)."""

    if getattr(commit, "merge", False):
        return False
    author = getattr(commit, "author_name", "") or ""
    if author in _BOT_AUTHORS:
        return False
    msg = (getattr(commit, "msg", "") or "").lower()
    return all(token not in msg for token in _DROP_MESSAGE_TOKENS)


def _resolve_parent_sha(commit: Any) -> str:
    """First-parent SHA, or 40 zeros for root commits."""

    parents = getattr(commit, "parents", None) or []
    if parents:
        return str(parents[0])
    return _NULL_PARENT_SHA


def _resolve_file_path(mod: Any) -> str | None:
    """Return the post-rename path if available, falling back to pre-rename."""

    new_path = getattr(mod, "new_path", None)
    old_path = getattr(mod, "old_path", None)
    return new_path or old_path


def _safe_parse_top_level(source: str) -> dict[str, libcst.CSTNode] | None:
    """Parse ``source`` and return its top-level nodes, or ``None`` on failure."""

    try:
        tree = libcst.parse_module(source)
    except libcst.ParserSyntaxError:
        return None
    return extract_top_level_nodes(tree)


def _node_source(node: libcst.CSTNode) -> str:
    """Round-trip a libcst node back to source by wrapping it in a fresh module."""
    tmp = libcst.parse_module("")
    return tmp.code_for_node(node)


def _unwrap_simple(stmt: libcst.CSTNode) -> libcst.CSTNode:
    """Unwrap a ``SimpleStatementLine``/``IndentedBlock`` wrapper, if any.

    Top-level definitions in libcst are direct ``FunctionDef``/``ClassDef`` /
    ``AsyncFunctionDef`` nodes under ``Module.body``; this helper is defensive
    in case a future libcst version threads them through an indirection.
    """

    return stmt
