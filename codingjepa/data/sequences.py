"""Sliding-window sequence builder for Stage A pretraining (RFC-0002 §D9, RFC-0012 §D4).

A :class:`ChunkSequence` is a fixed-size window of consecutive chunks taken
from one file: the last chunk is the prediction target, the preceding ``H``
chunks are the context. Per RFC-0012 §D4 we slide windows with stride 1 within
each file in file order; cross-file sequences are *not* built in v1.

Per-intent quota logic for Stage B refactor pairs (RFC-0002 §D8) also lives
here — :func:`apply_intent_quotas` caps training pairs at ``max_per_intent``
per label and persists the overflow for v2.

Output format
-------------

Each sequence is one Parquet row with columns:

* ``repo`` — string
* ``file_path`` — string
* ``context_ids`` — ``list<list<int32>>``: the H context chunks' ``token_ids``
* ``target_ids`` — ``list<int32>``: the prediction target's ``token_ids``
* ``intent_idx`` — int8 (always ``-1`` / ``I_NONE`` for pretraining)
* ``sequence_id`` — content-addressed sha256 of context + target tokens

If ``pyarrow`` is not installed, the writer falls back to JSON Lines so unit
tests can still verify round-tripping without a heavy optional dep.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    _HAVE_PYARROW = True
except ImportError:  # pragma: no cover — exercised only when pyarrow is absent
    _HAVE_PYARROW = False


__all__ = [
    "ChunkSequence",
    "apply_intent_quotas",
    "build_and_write_sequences",
    "build_sequences",
    "write_sequences",
]


_INTENT_NONE_IDX = -1


@dataclass
class ChunkSequence:
    """A sliding-window sequence of chunks for Stage A pretraining.

    The last chunk in the window is the prediction target; the preceding
    ``len(context_ids)`` chunks are the context. ``intent_idx`` is fixed to
    ``-1`` (``I_NONE``) for pretraining sequences per RFC-0002 §D9.

    ``sequence_id`` is ``sha256(repo:file_path:context_token_blob:target_token_blob)``
    so identical content produces identical IDs across machines and runs.
    """

    repo: str
    file_path: str
    context_ids: list[list[int]]
    target_ids: list[int]
    intent_idx: int
    sequence_id: str


def _sequence_id(
    repo: str,
    file_path: str,
    context_ids: list[list[int]],
    target_ids: list[int],
) -> str:
    """Deterministic content hash for a sequence."""

    payload = json.dumps(
        {
            "repo": repo,
            "file_path": file_path,
            "context_ids": context_ids,
            "target_ids": target_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_sequences(
    chunks: list[Any],
    *,
    history_len: int = 3,
    n_preds: int = 1,
    stride: int = 1,
) -> list[ChunkSequence]:
    """Build sliding-window sequences from a list of :class:`Chunk` objects.

    Groups chunks by ``(repo, file_path)``, preserving file order within each
    group, then slides a window of size ``history_len + n_preds + 1`` with the
    given ``stride``. The first ``history_len`` chunks of the window are the
    context; the chunk at index ``history_len`` is the prediction target. Per
    RFC-0012 §D4 cross-file sequences are not built in v1. Chunks whose
    ``token_ids`` is empty (over-cap drops, see RFC-0012 §D2) are filtered out
    *before* windowing so they neither contribute to context nor anchor a
    window.
    """

    # Stable group order: groups appear in the order their first chunk is seen.
    groups: dict[tuple[str, str], list[Any]] = {}
    for chunk in chunks:
        if not chunk.token_ids:
            continue
        key = (chunk.repo, chunk.file_path)
        groups.setdefault(key, []).append(chunk)

    window_size = history_len + n_preds + 1
    out: list[ChunkSequence] = []
    for (repo, file_path), file_chunks in groups.items():
        if len(file_chunks) < window_size:
            continue
        last_start = len(file_chunks) - window_size
        for start in range(0, last_start + 1, stride):
            window = file_chunks[start : start + window_size]
            ctx = [list(window[i].token_ids) for i in range(history_len)]
            target = list(window[history_len].token_ids)
            seq_id = _sequence_id(repo, file_path, ctx, target)
            out.append(
                ChunkSequence(
                    repo=repo,
                    file_path=file_path,
                    context_ids=ctx,
                    target_ids=target,
                    intent_idx=_INTENT_NONE_IDX,
                    sequence_id=seq_id,
                )
            )
    return out


def _write_sequences_parquet(seqs: list[ChunkSequence], path: Path) -> None:
    """Write sequences to a Parquet file using pyarrow."""

    table = pa.table(
        {
            "repo": pa.array([s.repo for s in seqs], type=pa.string()),
            "file_path": pa.array([s.file_path for s in seqs], type=pa.string()),
            "context_ids": pa.array(
                [s.context_ids for s in seqs],
                type=pa.list_(pa.list_(pa.int32())),
            ),
            "target_ids": pa.array(
                [s.target_ids for s in seqs],
                type=pa.list_(pa.int32()),
            ),
            "intent_idx": pa.array([s.intent_idx for s in seqs], type=pa.int8()),
            "sequence_id": pa.array([s.sequence_id for s in seqs], type=pa.string()),
        }
    )
    pq.write_table(table, path)


def _write_sequences_jsonl(seqs: list[ChunkSequence], path: Path) -> None:
    """Fallback writer when pyarrow is unavailable: one JSON object per line."""

    with path.open("w", encoding="utf-8") as fh:
        for s in seqs:
            fh.write(
                json.dumps(
                    {
                        "repo": s.repo,
                        "file_path": s.file_path,
                        "context_ids": s.context_ids,
                        "target_ids": s.target_ids,
                        "intent_idx": s.intent_idx,
                        "sequence_id": s.sequence_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_sequences(
    sequences: list[ChunkSequence],
    output_path: Path = Path("data/sequences/v1.parquet"),
) -> None:
    """Write sequences to ``output_path``.

    If ``pyarrow`` is available, the file is a real Parquet table with the
    schema documented in the module docstring; otherwise we fall back to JSON
    Lines under the same name (the columns are identical).
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _HAVE_PYARROW:
        _write_sequences_parquet(sequences, output_path)
    else:
        _write_sequences_jsonl(sequences, output_path)


def build_and_write_sequences(
    chunks: list[Any],
    output_path: Path = Path("data/sequences/v1.parquet"),
    *,
    history_len: int = 3,
    n_preds: int = 1,
) -> list[ChunkSequence]:
    """Convenience: :func:`build_sequences` + :func:`write_sequences`."""

    seqs = build_sequences(chunks, history_len=history_len, n_preds=n_preds)
    write_sequences(seqs, output_path=output_path)
    return seqs


# --------------------------------------------------------------------------- #
# Per-intent quotas (RFC-0002 §D8, issue #57)                                 #
# --------------------------------------------------------------------------- #


def _write_pairs_parquet(pairs: list[Any], path: Path) -> None:
    """Best-effort writer for overflow pairs.

    The pair shape is intentionally loose — anything with an ``intent_label``
    is acceptable — so we serialize via the dataclass-or-dict pattern and rely
    on the pyarrow / JSON-fallback split used by :func:`write_sequences`.
    """

    rows: list[dict[str, Any]] = []
    for p in pairs:
        if hasattr(p, "__dict__"):
            rows.append(dict(p.__dict__))
        else:
            rows.append(dict(p))

    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAVE_PYARROW and rows:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
    else:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def apply_intent_quotas(
    pairs: list[Any],
    *,
    max_per_intent: int = 12_000,
    overflow_path: Path | None = Path("data/pairs/v1.overflow.parquet"),
) -> tuple[list[Any], list[Any]]:
    """Cap training pairs at ``max_per_intent`` per intent label.

    Returns ``(retained, overflow)``. Pairs are kept in their input order;
    once an intent's retained count reaches ``max_per_intent``, all subsequent
    pairs of that intent go to ``overflow``. If ``overflow`` is non-empty and
    ``overflow_path`` is provided, the overflow is written to disk for v2 (per
    RFC-0002 §D8). Setting ``overflow_path=None`` disables disk persistence
    (useful for tests).
    """

    counts: dict[str, int] = {}
    retained: list[Any] = []
    overflow: list[Any] = []
    for pair in pairs:
        intent = getattr(pair, "intent_label", None)
        if intent is None and isinstance(pair, dict):
            intent = pair.get("intent_label")
        if intent is None:
            # No intent label means the pair predates labeling; treat as overflow
            # rather than silently bloat the retained set.
            overflow.append(pair)
            continue
        if counts.get(intent, 0) >= max_per_intent:
            overflow.append(pair)
            continue
        counts[intent] = counts.get(intent, 0) + 1
        retained.append(pair)

    if overflow and overflow_path is not None:
        _write_pairs_parquet(overflow, overflow_path)

    return retained, overflow
