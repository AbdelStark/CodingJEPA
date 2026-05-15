"""DataLoaders for CodingJEPA (issue #68, RFC-0008 §D5, §D9).

Implements:

* :class:`ChunkSequenceDataset` — sequences of chunks used for unconditional
  pretraining (Stage A).
* :class:`RefactorPairDataset` — labeled refactor pairs used for
  intent-conditioned fine-tuning (Stage B).
* :func:`seed_worker` — deterministic worker seeding (RFC-0008 §D9).
* :func:`build_pretrain_dataloader` — DataLoader with uniform random sampling.
* :func:`build_finetune_dataloader` — DataLoader with intent-balanced
  ``WeightedRandomSampler`` (RFC-0008 §D5).

Both datasets accept ``.parquet`` files when ``pyarrow`` is installed and
``.jsonl`` files as a dependency-free fallback (used in the unit tests).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Parquet / JSONL loading
# ---------------------------------------------------------------------------


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Load rows from a ``.parquet`` (via pyarrow) or ``.jsonl`` file.

    JSONL is the dependency-free fallback used by the unit tests. Parquet
    loading is gated on ``pyarrow`` being installed.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only when pyarrow is missing
            raise RuntimeError(f"pyarrow is required to read {path}; install pyarrow") from exc
        table = pq.read_table(path)
        columns = table.column_names
        column_data = [table[c].to_pylist() for c in columns]
        return [dict(zip(columns, row, strict=False)) for row in zip(*column_data, strict=False)]
    raise ValueError(f"unsupported dataset suffix {suffix!r} for {path}")


def _pad_or_truncate(seq: list[int], max_length: int, pad_id: int = 0) -> list[int]:
    """Right-pad / truncate ``seq`` to exactly ``max_length`` tokens."""
    if len(seq) >= max_length:
        return list(seq[:max_length])
    return list(seq) + [pad_id] * (max_length - len(seq))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ChunkSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of pre-training sequences (RFC-0008 §D5 — Stage A).

    Each row of the underlying file is expected to contain:

    * ``context_ids``: list of ``H`` token lists, one per context chunk.
    * ``target_ids``: list of ``n_preds`` token lists (or a single list, which
      is wrapped in a length-1 outer list automatically).
    * ``intent_idx``: integer intent index; ``-1`` is mapped to ``[I_NONE]``.

    ``__getitem__`` returns a dict with ``token_ids`` shaped ``(S, L)`` and
    ``intent_idx`` shaped ``()``.

    Parameters
    ----------
    parquet_path:
        Path to a ``.parquet`` or ``.jsonl`` file.
    max_length:
        Maximum tokens per chunk; sequences are right-padded with ``0``.
    """

    # ``[I_NONE]`` lives at index 8 in the intent embedding (RFC-0003 §D5).
    _I_NONE: int = 8

    def __init__(self, parquet_path: Path, *, max_length: int = 512) -> None:
        self.path = Path(parquet_path)
        self.max_length = max_length
        self._rows: list[dict[str, Any]] = _load_rows(self.path)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._rows[idx]
        context_ids = row["context_ids"]
        target_ids = row["target_ids"]
        if not context_ids:
            raise ValueError(f"row {idx} has empty context_ids in {self.path}")
        # Accept both shapes: list-of-int (single target chunk) or list-of-lists.
        if target_ids and isinstance(target_ids[0], int):
            target_chunks: list[list[int]] = [list(target_ids)]
        else:
            target_chunks = [list(t) for t in target_ids]
        context_chunks: list[list[int]] = [list(c) for c in context_ids]
        chunks = context_chunks + target_chunks
        padded = [_pad_or_truncate(c, self.max_length) for c in chunks]
        token_ids = torch.tensor(padded, dtype=torch.long)

        intent_raw = int(row.get("intent_idx", -1))
        intent_idx = self._I_NONE if intent_raw < 0 else intent_raw

        return {
            "token_ids": token_ids,
            "intent_idx": torch.tensor(intent_idx, dtype=torch.long),
        }


class RefactorPairDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of intent-conditioned refactor pairs (RFC-0008 §D5 — Stage B).

    Each row contains:

    * ``before_ids``: token list for the ``chunk_before`` half of the pair.
    * ``after_ids``: token list for the ``chunk_after`` half of the pair.
    * ``intent_idx``: integer intent index ``0..7``.

    ``__getitem__`` returns ``token_ids`` shaped ``(2, L)`` and an integer
    ``intent_idx`` tensor.
    """

    def __init__(self, parquet_path: Path, *, max_length: int = 512) -> None:
        self.path = Path(parquet_path)
        self.max_length = max_length
        self._rows: list[dict[str, Any]] = _load_rows(self.path)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._rows[idx]
        before = _pad_or_truncate(list(row["before_ids"]), self.max_length)
        after = _pad_or_truncate(list(row["after_ids"]), self.max_length)
        token_ids = torch.tensor([before, after], dtype=torch.long)
        intent_idx = int(row.get("intent_idx", 0))
        return {
            "token_ids": token_ids,
            "intent_idx": torch.tensor(intent_idx, dtype=torch.long),
        }

    def intent_indices(self) -> list[int]:
        """Return the list of ``intent_idx`` values for every row.

        Used by :func:`build_finetune_dataloader` to construct the
        :class:`WeightedRandomSampler` weights.
        """
        return [int(row.get("intent_idx", 0)) for row in self._rows]


# ---------------------------------------------------------------------------
# Worker seeding (RFC-0008 §D9)
# ---------------------------------------------------------------------------


def seed_worker(worker_id: int) -> None:
    """Deterministic worker seeding per RFC-0008 §D9.

    Hashes ``torch.initial_seed()`` down to a 32-bit value so that NumPy /
    Python ``random`` (which require 32-bit seeds) inherit the same entropy
    as the per-worker torch generator.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # Suppress unused warning when worker_id is not referenced.
    _ = worker_id


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _persistent_workers(num_workers: int) -> bool:
    """``persistent_workers`` is only valid when there is at least one worker."""
    return num_workers > 0


def build_pretrain_dataloader(
    dataset: Dataset[dict[str, torch.Tensor]],
    *,
    batch_size: int = 64,
    num_workers: int = 8,
    seed: int = 42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a uniform-random DataLoader for pretraining per RFC-0008 §D5.

    Defaults:

    * ``pin_memory=True``
    * ``prefetch_factor=4`` (only when ``num_workers > 0``)
    * ``persistent_workers=True`` (only when ``num_workers > 0``)
    * ``worker_init_fn=seed_worker``
    """
    generator = _make_generator(seed)
    kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if _persistent_workers(num_workers):
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def build_finetune_dataloader(
    dataset: Dataset[dict[str, torch.Tensor]],
    *,
    batch_size: int = 64,
    num_workers: int = 8,
    seed: int = 42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build an intent-balanced DataLoader for fine-tuning per RFC-0008 §D5.

    Uses :class:`WeightedRandomSampler` to ensure each minibatch contains
    roughly equal proportions of the 8 intents. Falls back to uniform
    shuffling if the dataset cannot expose its intent labels.
    """
    generator = _make_generator(seed)

    # Try to extract per-row intent labels; fall back to uniform if unavailable.
    intent_indices: list[int] | None = None
    intent_indices_fn = getattr(dataset, "intent_indices", None)
    if callable(intent_indices_fn):
        try:
            intent_indices = list(intent_indices_fn())
        except Exception:  # noqa: BLE001 — defensive fallback
            intent_indices = None

    kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    if intent_indices is not None and intent_indices:
        # Per-class weight = 1 / class_count; per-sample weight = weight[class].
        class_counts: dict[int, int] = {}
        for c in intent_indices:
            class_counts[c] = class_counts.get(c, 0) + 1
        weights: list[float] = [1.0 / class_counts[c] for c in intent_indices]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(intent_indices),
            replacement=True,
            generator=generator,
        )
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = True

    if _persistent_workers(num_workers):
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True

    return DataLoader(**kwargs)


__all__ = [
    "ChunkSequenceDataset",
    "RefactorPairDataset",
    "build_finetune_dataloader",
    "build_pretrain_dataloader",
    "seed_worker",
]
