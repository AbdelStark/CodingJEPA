"""FAISS IndexFlatIP + sidecar meta JSON (RFC-0009 §D3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from codingjepa.errors import IndexHashMismatch


@dataclass
class IndexMeta:
    """Metadata sidecar for a FAISS index."""

    chunk_ids: list[str]
    sources: list[str]
    index_id: str


def build_index(
    embeddings: np.ndarray,
    meta: IndexMeta,
    out_dir: Path,
) -> Path:
    """Build FAISS IndexFlatIP, write to out_dir/<index_id>.faiss + .meta.json.

    Parameters
    ----------
    embeddings:
        ``(N, D)`` float32, already L2-normalized.
    meta:
        Metadata describing the chunk IDs, sources, and a unique index ID.
    out_dir:
        Directory to write artifacts into (created if missing).

    Returns
    -------
    Path
        Path to the written ``.faiss`` file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    N, D = embeddings.shape
    index = faiss.IndexFlatIP(D)
    index.add(embeddings.astype(np.float32))

    faiss_path = out_dir / f"{meta.index_id}.faiss"
    faiss.write_index(index, str(faiss_path))

    meta_path = out_dir / f"{meta.index_id}.meta.json"
    meta_dict: dict[str, object] = {
        "index_id": meta.index_id,
        "chunk_ids": meta.chunk_ids,
        "sources": meta.sources,
    }
    meta_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

    return faiss_path


def load_index(
    index_id: str,
    index_dir: Path,
    *,
    expected_checkpoint_hash: str | None = None,
    expected_manifest_hash: str | None = None,
) -> tuple[faiss.Index, IndexMeta]:
    """Load index + meta, raise IndexHashMismatch if hashes don't match.

    The ``index_id`` is expected to have the form
    ``{checkpoint_hash_8chars}-{manifest_hash_8chars}`` (or any string
    containing exactly one ``-`` separator for the hash parsing).

    Parameters
    ----------
    index_id:
        Identifier matching the filename stem of the stored artifacts.
    index_dir:
        Directory containing ``<index_id>.faiss`` and ``<index_id>.meta.json``.
    expected_checkpoint_hash:
        First 8-character hex prefix of the checkpoint hash. Verified against
        the first segment of ``index_id``.
    expected_manifest_hash:
        First 8-character hex prefix of the manifest hash. Verified against
        the second segment of ``index_id``.

    Returns
    -------
    tuple[faiss.Index, IndexMeta]
        Loaded FAISS index and its metadata.

    Raises
    ------
    IndexHashMismatch
        When any expected hash does not match the index_id segments.
    FileNotFoundError
        When the expected files are not present under index_dir.
    """
    meta_path = index_dir / f"{index_id}.meta.json"
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))

    stored_index_id: str = meta_dict["index_id"]

    # Parse index_id as "{checkpoint_hash_prefix}-{manifest_hash_prefix}"
    parts = stored_index_id.split("-", 1)
    if len(parts) == 2:
        stored_ckpt_hash, stored_manifest_hash = parts[0], parts[1]
    else:
        # Malformed index_id — skip hash validation
        stored_ckpt_hash = stored_index_id
        stored_manifest_hash = ""

    if expected_checkpoint_hash is not None:
        if not stored_ckpt_hash.startswith(expected_checkpoint_hash):
            raise IndexHashMismatch(
                f"Checkpoint hash mismatch: expected prefix "
                f"{expected_checkpoint_hash!r}, got {stored_ckpt_hash!r}",
                index_id=stored_index_id,
                expected_checkpoint_hash=expected_checkpoint_hash,
                stored_checkpoint_hash=stored_ckpt_hash,
            )

    if expected_manifest_hash is not None:
        if not stored_manifest_hash.startswith(expected_manifest_hash):
            raise IndexHashMismatch(
                f"Manifest hash mismatch: expected prefix "
                f"{expected_manifest_hash!r}, got {stored_manifest_hash!r}",
                index_id=stored_index_id,
                expected_manifest_hash=expected_manifest_hash,
                stored_manifest_hash=stored_manifest_hash,
            )

    faiss_path = index_dir / f"{index_id}.faiss"
    loaded_index: faiss.Index = faiss.read_index(str(faiss_path))

    index_meta = IndexMeta(
        chunk_ids=meta_dict["chunk_ids"],
        sources=meta_dict["sources"],
        index_id=stored_index_id,
    )

    return loaded_index, index_meta


__all__ = ["IndexMeta", "build_index", "load_index"]
