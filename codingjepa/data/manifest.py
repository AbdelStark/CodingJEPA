"""Content-addressed manifest writer (RFC-0012 §D11).

The frozen-dataset manifest is the single source of truth for what's in the v1
corpus. Every downstream stage (chunker, pairs, audit) reads ``commit_sha``,
``subset_paths``, ``split`` from this file and writes back per-repo counters
(``py_files_in_scope``, ``chunks_emitted``, ``chunks_dropped``).

The ``manifest_hash`` is computed over the JSON serialization of the manifest
with ``manifest_hash`` set to 64 zero hex chars. This makes the manifest
self-verifying: tampering with any other field changes the hash and
:func:`verify_manifest_hash` returns ``False``.

Schema: ``data/schemas/manifest.schema.json`` (validated on every write/load).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from codingjepa._jsonschema import load_schema, validate_record
from codingjepa.data.pairs import COMMIT_CUTOFF
from codingjepa.errors import UsageError

__all__ = [
    "load_manifest",
    "update_manifest_repo",
    "verify_manifest_hash",
    "write_manifest",
]


_SCHEMA_VERSION = "v1"
_ZERO_HASH = "0" * 64


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _commit_cutoff_iso() -> str:
    """ISO8601 string for the last UTC second included in the corpus (#175).

    :data:`codingjepa.data.pairs.COMMIT_CUTOFF` is the *exclusive* upper bound
    (e.g. ``2024-01-01T00:00:00Z``); we persist the last second that is still
    included (``2023-12-31T23:59:59Z``) so downstream consumers can read the
    manifest without re-deriving the off-by-one.
    """

    last_included = COMMIT_CUTOFF - timedelta(seconds=1)
    return last_included.strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialize_for_hash(manifest: dict[str, Any]) -> bytes:
    """Canonical JSON serialization used for the content hash."""

    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _compute_manifest_hash(manifest: dict[str, Any]) -> str:
    """SHA-256 over the manifest with ``manifest_hash`` set to ``0*64``."""

    zeroed = dict(manifest)
    zeroed["manifest_hash"] = _ZERO_HASH
    return hashlib.sha256(_serialize_for_hash(zeroed)).hexdigest()


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate ``manifest`` against ``data/schemas/manifest.schema.json``."""

    schema = load_schema("manifest")
    validate_record(manifest, schema)


def write_manifest(
    repos: list[dict[str, Any]],
    *,
    chunker_version: str = "v1",
    tokenizer_hash: str = _ZERO_HASH,
    secrets_scanner_version: str = "v1",
    splits_path: str = "data/splits/v1.lock.json",
    output_path: Path = Path("data/manifest.lock.json"),
) -> dict[str, Any]:
    """Write ``data/manifest.lock.json`` and return the manifest dict.

    The ``manifest_hash`` field is sha256 of the JSON with ``manifest_hash``
    set to zeros. Schema: ``data/schemas/manifest.schema.json``. The manifest
    is validated against the schema *before* the file is written, so a
    malformed input raises :class:`codingjepa.errors.UsageError` without
    touching the filesystem.
    """

    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "manifest_hash": _ZERO_HASH,
        "generated_at": _now_iso(),
        "chunker_version": chunker_version,
        "tokenizer_hash": tokenizer_hash,
        "secrets_scanner_version": secrets_scanner_version,
        "splits_path": splits_path,
        "commit_cutoff_utc": _commit_cutoff_iso(),
        "repos": [dict(r) for r in repos],
    }
    # Hash with the placeholder; then validate the *final* manifest.
    manifest["manifest_hash"] = _compute_manifest_hash(manifest)
    _validate_manifest(manifest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_manifest(path: Path = Path("data/manifest.lock.json")) -> dict[str, Any]:
    """Load and validate ``manifest.lock.json``.

    Raises :class:`codingjepa.errors.UsageError` if the file is missing or
    fails schema validation.
    """

    if not path.exists():
        raise UsageError(f"manifest not found: {path}", path=str(path))
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(payload)
    return payload


def verify_manifest_hash(manifest: dict[str, Any]) -> bool:
    """Return True iff ``manifest_hash`` matches the content."""

    current = manifest.get("manifest_hash")
    if not isinstance(current, str):
        return False
    return current == _compute_manifest_hash(manifest)


def update_manifest_repo(
    manifest: dict[str, Any],
    repo_name: str,
    *,
    py_files_in_scope: int | None = None,
    chunks_emitted: int | None = None,
    chunks_dropped: int | None = None,
    audit_path: str | None = None,
) -> dict[str, Any]:
    """Update a repo entry in the manifest and recompute ``manifest_hash``.

    Returns a *new* dict (the input is not mutated). Raises
    :class:`codingjepa.errors.UsageError` if ``repo_name`` does not match any
    entry's ``name``.
    """

    updates: dict[str, int | str] = {}
    if py_files_in_scope is not None:
        updates["py_files_in_scope"] = py_files_in_scope
    if chunks_emitted is not None:
        updates["chunks_emitted"] = chunks_emitted
    if chunks_dropped is not None:
        updates["chunks_dropped"] = chunks_dropped
    if audit_path is not None:
        updates["audit_path"] = audit_path

    new_repos: list[dict[str, Any]] = []
    matched = False
    for repo in manifest["repos"]:
        if repo["name"] == repo_name:
            merged = dict(repo)
            merged.update(updates)
            new_repos.append(merged)
            matched = True
        else:
            new_repos.append(dict(repo))

    if not matched:
        raise UsageError(
            f"repo not found in manifest: {repo_name}",
            repo_name=repo_name,
        )

    updated = dict(manifest)
    updated["repos"] = new_repos
    updated["manifest_hash"] = _compute_manifest_hash(updated)
    _validate_manifest(updated)
    return updated
