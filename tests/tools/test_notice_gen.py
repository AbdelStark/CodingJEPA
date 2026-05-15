"""Tests for ``tools/notice_gen.py`` — NOTICE file generator (#30).

The script reads ``data/manifest.lock.json``, enumerates every repo entry, and
writes a ``NOTICE`` file with the SPDX identifier, commit SHA, and copyright
line for each upstream source. Behaviour:

* Missing manifest → exit 1 with an informative message on stderr.
* Repos with an explicit ``copyright`` field (forward-compatible) use that
  value; otherwise the NOTICE falls back to ``(see upstream repository)``.
* Output is deterministic given the same manifest.

The tests are filesystem-only and never touch the network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "notice_gen.py"


def _minimal_manifest(repos: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a manifest dict that satisfies ``data/schemas/manifest.schema.json``."""

    return {
        "schema_version": "v1",
        "manifest_hash": "0" * 64,
        "generated_at": "2024-01-01T00:00:00+00:00",
        "chunker_version": "v1",
        "tokenizer_hash": "0" * 64,
        "secrets_scanner_version": "v1",
        "splits_path": "data/splits/v1.lock.json",
        "commit_cutoff_utc": "2023-12-31T23:59:59Z",
        "repos": repos,
    }


def _repo_entry(
    *,
    name: str,
    commit_sha: str,
    license_spdx: str,
    split: str = "train",
    copyright_line: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "url": f"https://github.com/example/{name}",
        "commit_sha": commit_sha,
        "license_spdx": license_spdx,
        "subset_paths": ["src/"],
        "split": split,
        "py_files_in_scope": 1,
        "chunks_emitted": 1,
        "chunks_dropped": 0,
        "audit_path": f"data/audit/{name}.json",
    }
    if copyright_line is not None:
        # Forward-compatible optional field; the schema currently rejects unknown
        # properties, so this branch is only used by tests that point the script
        # at a hand-rolled manifest read with --no-validate (or that bypass
        # validation entirely).
        entry["copyright"] = copyright_line
    return entry


def _run(
    *args: str | Path,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *[str(a) for a in args]],
        cwd=str(cwd) if cwd is not None else str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_notice_gen_writes_notice_for_minimal_manifest(tmp_path: Path) -> None:
    """A 2-repo manifest produces a NOTICE listing both, with SPDX/SHA/copyright."""

    manifest = _minimal_manifest(
        [
            _repo_entry(
                name="alpha",
                commit_sha="a" * 40,
                license_spdx="Apache-2.0",
            ),
            _repo_entry(
                name="beta",
                commit_sha="b" * 40,
                license_spdx="MIT",
                split="val",
            ),
        ]
    )
    manifest_path = tmp_path / "manifest.lock.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    notice_path = tmp_path / "NOTICE"
    result = _run("--manifest", manifest_path, "--out", notice_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert notice_path.exists()

    text = notice_path.read_text(encoding="utf-8")
    assert "CodingJEPA" in text
    assert "## alpha" in text
    assert "## beta" in text
    assert "Apache-2.0" in text
    assert "MIT" in text
    assert "a" * 40 in text
    assert "b" * 40 in text


def test_notice_gen_missing_manifest_exits_1(tmp_path: Path) -> None:
    """Pointing at a non-existent manifest exits 1 with an informative message."""

    missing = tmp_path / "does-not-exist.json"
    result = _run("--manifest", missing, "--out", tmp_path / "NOTICE")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "manifest" in (result.stderr + result.stdout).lower()


def test_notice_gen_uses_copyright_when_present(tmp_path: Path) -> None:
    """If a repo entry carries an explicit ``copyright`` line, it is used verbatim.

    The script reads the JSON without enforcing the schema's
    ``additionalProperties: false`` constraint, so forward-compatible fields
    like ``copyright`` survive. (Adding ``copyright`` to the schema is a
    follow-up; this test pins the runtime behaviour today.)
    """

    manifest = _minimal_manifest(
        [
            _repo_entry(
                name="gamma",
                commit_sha="c" * 40,
                license_spdx="BSD-3-Clause",
                copyright_line="Copyright (c) 2020 Example Authors",
            )
        ]
    )
    manifest_path = tmp_path / "manifest.lock.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    notice_path = tmp_path / "NOTICE"
    result = _run("--manifest", manifest_path, "--out", notice_path)
    assert result.returncode == 0, result.stdout + result.stderr
    text = notice_path.read_text(encoding="utf-8")
    assert "Copyright (c) 2020 Example Authors" in text


def test_notice_gen_falls_back_when_copyright_missing(tmp_path: Path) -> None:
    """Without a copyright field, the NOTICE shows a stable fallback line."""

    manifest = _minimal_manifest(
        [
            _repo_entry(
                name="delta",
                commit_sha="d" * 40,
                license_spdx="PSF-2.0",
            )
        ]
    )
    manifest_path = tmp_path / "manifest.lock.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    notice_path = tmp_path / "NOTICE"
    result = _run("--manifest", manifest_path, "--out", notice_path)
    assert result.returncode == 0, result.stdout + result.stderr
    text = notice_path.read_text(encoding="utf-8")
    # Fallback should include some indicator that no explicit copyright was found.
    assert "see upstream repository" in text.lower() or "https://github.com/example/delta" in text


def test_notice_gen_is_deterministic(tmp_path: Path) -> None:
    """Running the generator twice on the same manifest produces byte-identical output."""

    manifest = _minimal_manifest(
        [
            _repo_entry(name="zeta", commit_sha="e" * 40, license_spdx="Apache-2.0"),
            _repo_entry(name="alpha", commit_sha="f" * 40, license_spdx="MIT"),
        ]
    )
    manifest_path = tmp_path / "manifest.lock.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out1 = tmp_path / "NOTICE.1"
    out2 = tmp_path / "NOTICE.2"
    r1 = _run("--manifest", manifest_path, "--out", out1)
    r2 = _run("--manifest", manifest_path, "--out", out2)
    assert r1.returncode == 0 and r2.returncode == 0
    assert out1.read_bytes() == out2.read_bytes()


def test_notice_gen_default_paths(tmp_path: Path) -> None:
    """When run with no args, the script reads ``data/manifest.lock.json`` and
    writes ``NOTICE`` in the current working directory.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = _minimal_manifest(
        [_repo_entry(name="omega", commit_sha="9" * 40, license_spdx="Apache-2.0")]
    )
    (data_dir / "manifest.lock.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = _run(cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    notice = (tmp_path / "NOTICE").read_text(encoding="utf-8")
    assert "## omega" in notice


def test_licenses_directory_contains_required_texts() -> None:
    """The ``LICENSES/`` directory ships verbatim text for every license SPDX
    we currently use across the corpus (#30 scope).
    """

    licenses_dir = REPO_ROOT / "LICENSES"
    assert licenses_dir.is_dir(), f"missing LICENSES/ directory at {licenses_dir}"

    required = ["Apache-2.0", "BSD-3-Clause", "MIT", "PSF-2.0"]
    for spdx in required:
        path = licenses_dir / f"{spdx}.txt"
        assert path.is_file(), f"missing license text: {path}"
        # Sanity check: real license texts are at least a few hundred bytes.
        assert path.stat().st_size > 200, f"license text suspiciously short: {path}"
