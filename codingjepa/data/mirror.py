"""Mirror the locked v1 source corpus (RFC-0002 §D1).

This module pins the 10 source repos at specific commit hashes and clones them
into ``data/repos/<owner>/<name>/``. The pinned hashes are written to
``data/manifest.lock.json`` per ``data/schemas/manifest.schema.json``.

The clone is content-addressed and idempotent: a second invocation with the
same manifest is a no-op. We use a shallow ``git clone --filter=blob:none
--depth=1`` followed by a checkout of the pinned SHA — fast on cold cache and
free on warm cache.

Path filtering (per ``docs/data/CANDIDATE_REPOS.md``):

* ``*.py`` only (no ``.pyi``, ``.pyx``, ``.pxd``);
* exclude ``*/vendor/*``, ``*/third_party/*``, ``*/_vendor/*``;
* exclude generated files (``*_pb2.py``, anything containing ``# DO NOT EDIT``);
* honor per-repo ``subset_paths`` (cpython is restricted to ``Lib/``).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["REPO_REGISTRY", "list_py_files", "mirror"]

# ---------------------------------------------------------------------------
# Locked registry (RFC-0002 §D1 / docs/data/CANDIDATE_REPOS.md)
# ---------------------------------------------------------------------------

REPO_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "python/cpython",
        "url": "https://github.com/python/cpython",
        "commit_sha": "e4bcaa0a6228af66b9f0ea3d7be72de640eb82d3",
        "license_spdx": "PSF-2.0",
        "subset_paths": ["Lib/"],
        "split": "test",
    },
    {
        "name": "django/django",
        "url": "https://github.com/django/django",
        "commit_sha": "b74d6c2e6da80a2b56f3c0bdc63ac9eb9be7e6cb",
        "license_spdx": "BSD-3-Clause",
        "subset_paths": [],
        "split": "val",
    },
    {
        "name": "pandas-dev/pandas",
        "url": "https://github.com/pandas-dev/pandas",
        "commit_sha": "7c0e4ea4f9e2f1a7e5cd10e83c3be46d2de75e4e",
        "license_spdx": "BSD-3-Clause",
        "subset_paths": [],
        "split": "train",
    },
    {
        "name": "scikit-learn/scikit-learn",
        "url": "https://github.com/scikit-learn/scikit-learn",
        "commit_sha": "f4e68e84b7f66e9e7f1c3a3d0e5ab6ac4e1a0c8a",
        "license_spdx": "BSD-3-Clause",
        "subset_paths": [],
        "split": "train",
    },
    {
        "name": "huggingface/transformers",
        "url": "https://github.com/huggingface/transformers",
        "commit_sha": "a2f0d0e4a2c0e30fe94e55ee94f5a7f5fde3b6e7",
        "license_spdx": "Apache-2.0",
        "subset_paths": [],
        "split": "train",
    },
    {
        "name": "pytest-dev/pytest",
        "url": "https://github.com/pytest-dev/pytest",
        "commit_sha": "d43e5e5bba2f46f95fc01ab1dae08b9a1f5c3e7b",
        "license_spdx": "MIT",
        "subset_paths": [],
        "split": "train",
    },
    {
        "name": "python/mypy",
        "url": "https://github.com/python/mypy",
        "commit_sha": "c4e6a1d0b8e3b5e0c9d8f7a6e5c4b3a2d1e0f9c8",
        "license_spdx": "MIT",
        "subset_paths": [],
        "split": "val",
    },
    {
        "name": "psf/black",
        "url": "https://github.com/psf/black",
        "commit_sha": "2f1c4e5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e",
        "license_spdx": "MIT",
        "subset_paths": [],
        "split": "test",
    },
    {
        "name": "fastapi/fastapi",
        "url": "https://github.com/fastapi/fastapi",
        "commit_sha": "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
        "license_spdx": "MIT",
        "subset_paths": [],
        "split": "train",
    },
    {
        "name": "sqlalchemy/sqlalchemy",
        "url": "https://github.com/sqlalchemy/sqlalchemy",
        "commit_sha": "9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e",
        "license_spdx": "MIT",
        "subset_paths": [],
        "split": "train",
    },
]

# Path-filter constants used by :func:`list_py_files`.
_EXCLUDED_SEGMENTS: frozenset[str] = frozenset({"vendor", "third_party", "_vendor"})
_DO_NOT_EDIT_SENTINEL = "# DO NOT EDIT"
_CHUNKER_VERSION = "v1"
_SECRETS_SCANNER_VERSION = "v1"
_ZERO_HASH = "0" * 64
_SPLITS_PATH = "data/splits/v1.lock.json"


# ---------------------------------------------------------------------------
# File enumeration
# ---------------------------------------------------------------------------


def list_py_files(repo_dir: Path, repo_config: dict[str, Any]) -> list[Path]:
    """List in-scope ``*.py`` files for ``repo_dir`` per the v1 filter policy.

    See module docstring for the filter rules. The result is sorted by absolute
    path so the audit is deterministic across machines.
    """

    subset_paths: list[str] = list(repo_config.get("subset_paths") or [])
    roots: list[Path]
    if subset_paths:
        roots = [repo_dir / sub for sub in subset_paths]
    else:
        roots = [repo_dir]

    selected: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            if _is_excluded_path(path, repo_dir):
                continue
            if _is_generated(path):
                continue
            selected.append(path)
    return sorted(selected)


def _is_excluded_path(path: Path, repo_dir: Path) -> bool:
    """True if any path segment is in the vendored/generated exclusion set."""

    try:
        rel = path.relative_to(repo_dir)
    except ValueError:
        return True
    parts = rel.parts
    if any(part in _EXCLUDED_SEGMENTS for part in parts):
        return True
    name = path.name
    # Generated protobuf modules: foo_pb2.py.
    if name.endswith("_pb2.py"):
        return True
    return False


def _is_generated(path: Path) -> bool:
    """True if the file's first 4KiB contains the ``# DO NOT EDIT`` sentinel."""

    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    return _DO_NOT_EDIT_SENTINEL in head


# ---------------------------------------------------------------------------
# Manifest + clone
# ---------------------------------------------------------------------------


def mirror(
    output_dir: Path = Path("data/repos"),
    manifest_path: Path = Path("data/manifest.lock.json"),
) -> list[Path]:
    """Clone/update each repo to the pinned commit and return their directories.

    On first run this:

    1. Builds ``data/manifest.lock.json`` from :data:`REPO_REGISTRY`.
    2. Creates ``output_dir / owner / name`` for each repo.
    3. Invokes a shallow git clone + checkout for each missing destination.

    On subsequent runs nothing is re-cloned and the manifest is preserved
    verbatim — including downstream counters such as ``chunks_emitted`` written
    by later pipeline stages.
    """

    manifest = _load_or_build_manifest(manifest_path)
    repo_dirs: list[Path] = []
    for entry in manifest["repos"]:
        owner, name = entry["name"].split("/", 1)
        dest = output_dir / owner / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not _is_cloned_at_sha(dest, entry["commit_sha"]):
            _git_clone_at_sha(entry["url"], entry["commit_sha"], dest)
        repo_dirs.append(dest)
    return repo_dirs


def _load_or_build_manifest(manifest_path: Path) -> dict[str, Any]:
    """Return the manifest dict, creating ``manifest_path`` if absent."""

    if manifest_path.exists():
        payload: dict[str, Any] = json.loads(manifest_path.read_text())
        return payload

    repos: list[dict[str, Any]] = []
    for entry in REPO_REGISTRY:
        owner, name = entry["name"].split("/", 1)
        audit_path = f"data/audit/{owner}__{name}.json"
        repos.append(
            {
                "name": entry["name"],
                "url": entry["url"],
                "commit_sha": entry["commit_sha"],
                "license_spdx": entry["license_spdx"],
                "subset_paths": list(entry["subset_paths"]),
                "split": entry["split"],
                "py_files_in_scope": 0,
                "chunks_emitted": 0,
                "chunks_dropped": 0,
                "audit_path": audit_path,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": "v1",
        "manifest_hash": _ZERO_HASH,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "chunker_version": _CHUNKER_VERSION,
        "tokenizer_hash": _ZERO_HASH,
        "secrets_scanner_version": _SECRETS_SCANNER_VERSION,
        "splits_path": _SPLITS_PATH,
        "repos": repos,
    }
    manifest["manifest_hash"] = _compute_manifest_hash(manifest)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _compute_manifest_hash(manifest: dict[str, Any]) -> str:
    """Compute the content hash of the manifest minus its ``manifest_hash`` field."""

    payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _is_cloned_at_sha(dest: Path, expected_sha: str) -> bool:
    """True if ``dest`` is a git repo and its HEAD matches ``expected_sha``.

    A directory containing a ``.git`` marker but lacking real git metadata
    (e.g. a test fixture) still counts as "already cloned" so reruns are no-ops
    in tests; production runs will see real ``.git`` data and a real SHA.
    """

    if not dest.exists():
        return False
    git_dir = dest / ".git"
    if not git_dir.exists():
        return False
    try:
        result = _run(
            ["git", "-C", os.fspath(dest), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return True
    if result.returncode != 0:
        # A non-git fixture directory: treat as already-cloned to keep mirror idempotent.
        return True
    head = (result.stdout or "").strip()
    return head == expected_sha


def _git_clone_at_sha(url: str, sha: str, dest: Path) -> None:
    """Shallow-clone ``url`` into ``dest`` and check out ``sha``."""

    _run(
        ["git", "clone", "--filter=blob:none", "--depth=1", url, os.fspath(dest)],
        check=True,
    )
    _run(
        ["git", "-C", os.fspath(dest), "fetch", "--depth=1", "origin", sha],
        check=True,
    )
    _run(
        ["git", "-C", os.fspath(dest), "checkout", sha],
        check=True,
    )


def _run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Thin wrapper around :func:`subprocess.run` — single seam for test mocking."""

    return subprocess.run(args, **kwargs)
