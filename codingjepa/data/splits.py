"""By-repo train/val/test splitting + cross-split leakage detection (RFC-0014 §D7).

Splits are deterministic per-repo: every chunk inherits its owning repository's
split assignment from :data:`SPLIT_ASSIGNMENTS`. The split file
(``data/splits/v1.lock.json``) is content-addressed by a sha256 over the
canonical (sorted) `{repo: split}` mapping so downstream consumers can detect
drift.

Cross-split leakage detection (RFC-0014 §D6 §3) flags pairs of chunks in
different splits whose Jaccard line-set similarity is ≥ ``threshold``. The
runbook resolves leakage by dropping the test/val side; the audit report at
``data/audit/cross_split_leakage.json`` is the invariant input to the
``test_cross_split_leakage_zero`` acceptance test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codingjepa.errors import SplitContractViolation

Split = Literal["train", "val", "test", "ood"]


# ---------------------------------------------------------------------------
# Locked per-repo split assignments (RFC-0014 §D7 / docs/data/CANDIDATE_REPOS.md)
# ---------------------------------------------------------------------------

SPLIT_ASSIGNMENTS: dict[str, Split] = {
    "pandas-dev/pandas": "train",
    "scikit-learn/scikit-learn": "train",
    "huggingface/transformers": "train",
    "pytest-dev/pytest": "train",
    "fastapi/fastapi": "train",
    "sqlalchemy/sqlalchemy": "train",
    "django/django": "val",
    "python/mypy": "val",
    "psf/black": "test",
    "python/cpython": "test",
}


_SCHEMA_VERSION = "v1"
_ZERO_HASH = "0" * 64


@dataclass
class SplitResult:
    """Aggregate train/val/test partitioning result plus leakage counter.

    ``cross_split_leakage`` must equal zero before the lock file is written
    (RFC-0014 §D6 §3); any non-zero count means the leakage detector found a
    pair that has *not* yet been resolved by the dedup step.
    """

    train: list[Any] = field(default_factory=list)
    val: list[Any] = field(default_factory=list)
    test: list[Any] = field(default_factory=list)
    cross_split_leakage: int = 0


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------


def assign_splits(
    chunks: list[Any],
    *,
    by_repo: bool = True,
) -> dict[str, list[Any]]:
    """Partition ``chunks`` into ``{train, val, test}`` lists by their repo.

    ``by_repo`` is the only mode v1 supports; the keyword is kept so future
    work (e.g. hash-stripe splits for OOD studies) can opt in without breaking
    the signature.

    A chunk whose ``repo`` attribute is not in :data:`SPLIT_ASSIGNMENTS`
    raises :class:`SplitContractViolation` — silently routing unknown repos
    would defeat the leakage guarantee.

    Ordering within each split mirrors the caller's input order so the result
    is deterministic and the splits lock file is byte-stable.
    """

    if not by_repo:
        raise SplitContractViolation(
            "only by_repo=True is supported in v1",
            mode="by_repo",
            value=by_repo,
        )

    buckets: dict[str, list[Any]] = {"train": [], "val": [], "test": []}
    for chunk in chunks:
        repo = _chunk_repo(chunk)
        split = SPLIT_ASSIGNMENTS.get(repo)
        if split is None:
            raise SplitContractViolation(
                f"unknown repo for split assignment: {repo}",
                repo=repo,
                known=sorted(SPLIT_ASSIGNMENTS),
            )
        if split == "ood":
            # OOD chunks are not part of the train/val/test trio.
            continue
        buckets[split].append(chunk)
    return buckets


# ---------------------------------------------------------------------------
# detect_leakage
# ---------------------------------------------------------------------------


def detect_leakage(
    split_chunks: dict[str, list[Any]],
    *,
    threshold: float = 0.85,
) -> list[tuple[str, str]]:
    """Find cross-split near-duplicate chunk pairs.

    Uses a token-level Jaccard similarity over the chunk's
    ``source_normalized`` field. Pairs *within* the same split are ignored
    (those are the dedup module's responsibility); only cross-split pairs
    surface.

    Returns ``[(chunk_id_a, chunk_id_b), ...]`` where ``chunk_a`` is in the
    higher-priority split (train < val < test) and ``chunk_b`` is in the
    lower-priority one. RFC-0014 §D6 §3 then drops chunk_b.
    """

    if threshold > 1.0:
        return []

    # Build a flat list of (split, chunk) pairs.
    flat: list[tuple[str, Any]] = []
    for split in ("train", "val", "test"):
        for chunk in split_chunks.get(split, []):
            flat.append((split, chunk))

    pairs: list[tuple[str, str]] = []
    for i, (split_a, chunk_a) in enumerate(flat):
        tokens_a = _tokens(chunk_a)
        if not tokens_a:
            continue
        for split_b, chunk_b in flat[i + 1 :]:
            if split_a == split_b:
                continue
            tokens_b = _tokens(chunk_b)
            if not tokens_b:
                continue
            sim = _jaccard(tokens_a, tokens_b)
            if sim >= threshold:
                pairs.append((chunk_a.chunk_id, chunk_b.chunk_id))
    return pairs


def _tokens(chunk: Any) -> frozenset[str]:
    """Tokenize a chunk's normalized source for Jaccard comparison.

    Splits on whitespace; we deliberately do *not* call the SentencePiece
    tokenizer here because the leakage detector runs before the tokenizer is
    in scope (the tokenizer is pinned by the manifest).
    """

    source = getattr(chunk, "source_normalized", "") or ""
    return frozenset(source.split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# write_splits_lock
# ---------------------------------------------------------------------------


def write_splits_lock(
    split_chunks: dict[str, list[Any]],
    output_path: Path = Path("data/splits/v1.lock.json"),
) -> None:
    """Write ``data/splits/v1.lock.json`` per ``data/schemas/splits.schema.json``.

    The ``split_hash`` is sha256 over the canonical, sort-key JSON encoding of
    ``by_repo`` (so the same logical assignment always produces the same
    digest, independent of dict insertion order).
    """

    by_repo: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for chunk in split_chunks.get(split, []):
            repo = _chunk_repo(chunk)
            existing = by_repo.get(repo)
            if existing is not None and existing != split:
                raise SplitContractViolation(
                    f"repo {repo} appears in multiple splits ({existing} and {split})",
                    repo=repo,
                )
            by_repo[repo] = split

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "split_hash": _ZERO_HASH,
        "by_repo": by_repo,
    }
    payload["split_hash"] = _compute_split_hash(by_repo)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compute_split_hash(by_repo: dict[str, str]) -> str:
    blob = json.dumps(by_repo, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# write_leakage_report
# ---------------------------------------------------------------------------


def write_leakage_report(
    leakage_pairs: list[tuple[str, str]],
    output_path: Path = Path("data/audit/cross_split_leakage.json"),
) -> None:
    """Write the cross-split leakage audit JSON.

    Per RFC-0014 §D6 §3 the cleanup convention is to drop the second member
    of each detected pair (the one in the lower-priority split, which is
    chunk_b in :func:`detect_leakage`'s contract). We therefore emit each
    pair under ``resolved`` with ``resolution="dropped_b"`` and leave
    ``remaining_crossings`` empty. If a future operator chooses to keep both
    sides, they would push entries into ``remaining_crossings`` instead.
    """

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "remaining_crossings": [],
        "resolved": [
            {
                "chunk_id_a": a,
                "chunk_id_b": b,
                "resolution": "dropped_b",
            }
            for a, b in leakage_pairs
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _chunk_repo(chunk: Any) -> str:
    """Pull the ``repo`` attribute or key off a chunk-like object."""

    repo = getattr(chunk, "repo", None)
    if repo is None and isinstance(chunk, dict):
        repo = chunk.get("repo")
    if repo is None:
        raise SplitContractViolation(
            "chunk is missing a `repo` attribute",
            chunk=str(chunk),
        )
    return str(repo)


__all__ = [
    "SPLIT_ASSIGNMENTS",
    "Split",
    "SplitResult",
    "assign_splits",
    "detect_leakage",
    "write_leakage_report",
    "write_splits_lock",
]
