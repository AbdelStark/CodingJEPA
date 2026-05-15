"""Tests for codingjepa.data.manifest. See RFC-0012 §D11."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from codingjepa.data import manifest as manifest_mod
from codingjepa.errors import UsageError


def _make_repo(
    name: str = "owner/repo",
    *,
    commit_sha: str = "a" * 40,
    license_spdx: str = "MIT",
    subset_paths: list[str] | None = None,
    split: str = "train",
    py_files_in_scope: int = 0,
    chunks_emitted: int = 0,
    chunks_dropped: int = 0,
    audit_path: str = "data/audit/owner__repo.json",
) -> dict[str, Any]:
    return {
        "name": name,
        "url": f"https://github.com/{name}",
        "commit_sha": commit_sha,
        "license_spdx": license_spdx,
        "subset_paths": list(subset_paths or []),
        "split": split,
        "py_files_in_scope": py_files_in_scope,
        "chunks_emitted": chunks_emitted,
        "chunks_dropped": chunks_dropped,
        "audit_path": audit_path,
    }


def test_write_manifest_creates_file(tmp_path: Path) -> None:
    """write_manifest should create the output file on disk."""

    out = tmp_path / "manifest.lock.json"
    repos = [_make_repo()]
    manifest_mod.write_manifest(repos, output_path=out)
    assert out.exists()


def test_write_manifest_returns_dict_with_required_fields(tmp_path: Path) -> None:
    """The returned dict contains every required field from the schema."""

    out = tmp_path / "manifest.lock.json"
    result = manifest_mod.write_manifest([_make_repo()], output_path=out)

    required = {
        "schema_version",
        "manifest_hash",
        "generated_at",
        "chunker_version",
        "tokenizer_hash",
        "secrets_scanner_version",
        "splits_path",
        "repos",
    }
    assert required <= result.keys()


def test_write_manifest_hash_is_correct(tmp_path: Path) -> None:
    """manifest_hash is sha256 over the JSON with manifest_hash zeroed."""

    out = tmp_path / "manifest.lock.json"
    repos = [_make_repo(commit_sha="b" * 40)]
    manifest = manifest_mod.write_manifest(
        repos,
        chunker_version="v1",
        tokenizer_hash="c" * 64,
        output_path=out,
    )

    # Independently re-compute and compare.
    zeroed = dict(manifest)
    zeroed["manifest_hash"] = "0" * 64
    blob = json.dumps(zeroed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = hashlib.sha256(blob).hexdigest()
    assert manifest["manifest_hash"] == expected
    assert len(manifest["manifest_hash"]) == 64


def test_write_manifest_validates_schema(tmp_path: Path) -> None:
    """Invalid repos raise UsageError before any file is written."""

    out = tmp_path / "manifest.lock.json"
    bad = _make_repo(commit_sha="not-hex")  # fails sha pattern
    with pytest.raises(UsageError):
        manifest_mod.write_manifest([bad], output_path=out)
    assert not out.exists()


def test_load_manifest_roundtrip(tmp_path: Path) -> None:
    """write_manifest then load_manifest returns an equivalent dict."""

    out = tmp_path / "manifest.lock.json"
    repos = [_make_repo(), _make_repo(name="owner/another", commit_sha="d" * 40)]
    written = manifest_mod.write_manifest(repos, output_path=out)
    loaded = manifest_mod.load_manifest(out)

    assert loaded == written


def test_load_manifest_validates(tmp_path: Path) -> None:
    """load_manifest rejects a malformed file."""

    out = tmp_path / "manifest.lock.json"
    out.write_text(json.dumps({"schema_version": "v1"}))  # missing required fields
    with pytest.raises(UsageError):
        manifest_mod.load_manifest(out)


def test_verify_manifest_hash_true_for_valid(tmp_path: Path) -> None:
    """A manifest written by write_manifest verifies."""

    out = tmp_path / "manifest.lock.json"
    manifest = manifest_mod.write_manifest([_make_repo()], output_path=out)
    assert manifest_mod.verify_manifest_hash(manifest) is True


def test_verify_manifest_hash_false_when_tampered(tmp_path: Path) -> None:
    """If we mutate a field after the fact, verify_manifest_hash should be False."""

    out = tmp_path / "manifest.lock.json"
    manifest = manifest_mod.write_manifest([_make_repo()], output_path=out)
    tampered = dict(manifest)
    # Tamper a repo counter; recompute should now mismatch.
    tampered["repos"] = [dict(r) for r in manifest["repos"]]
    tampered["repos"][0]["chunks_emitted"] = 99
    assert manifest_mod.verify_manifest_hash(tampered) is False


def test_update_manifest_repo_changes_value_and_rehashes(tmp_path: Path) -> None:
    """update_manifest_repo mutates one repo and recomputes manifest_hash."""

    out = tmp_path / "manifest.lock.json"
    manifest = manifest_mod.write_manifest([_make_repo(name="owner/repo")], output_path=out)
    old_hash = manifest["manifest_hash"]

    updated = manifest_mod.update_manifest_repo(
        manifest, "owner/repo", chunks_emitted=123, py_files_in_scope=5
    )
    assert updated["repos"][0]["chunks_emitted"] == 123
    assert updated["repos"][0]["py_files_in_scope"] == 5
    assert updated["manifest_hash"] != old_hash
    assert manifest_mod.verify_manifest_hash(updated) is True


def test_update_manifest_repo_unknown_name_raises(tmp_path: Path) -> None:
    """Updating a non-existent repo entry is a UsageError."""

    out = tmp_path / "manifest.lock.json"
    manifest = manifest_mod.write_manifest([_make_repo(name="owner/a")], output_path=out)
    with pytest.raises(UsageError):
        manifest_mod.update_manifest_repo(manifest, "owner/missing", chunks_emitted=1)


def test_update_manifest_repo_partial_update(tmp_path: Path) -> None:
    """Only specified fields are updated; others preserved."""

    out = tmp_path / "manifest.lock.json"
    repos = [_make_repo(name="owner/x", py_files_in_scope=10, chunks_emitted=20)]
    manifest = manifest_mod.write_manifest(repos, output_path=out)

    updated = manifest_mod.update_manifest_repo(manifest, "owner/x", chunks_dropped=3)
    assert updated["repos"][0]["py_files_in_scope"] == 10
    assert updated["repos"][0]["chunks_emitted"] == 20
    assert updated["repos"][0]["chunks_dropped"] == 3
