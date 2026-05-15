"""Tests for codingjepa.data.audit. See RFC-0002 §D2 and RFC-0014 §D10."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from codingjepa._jsonschema import load_schema, validate_record
from codingjepa.data.audit import (
    ALLOWED_LICENSES,
    check_all_gates,
    compute_audit,
    load_audit,
    write_audit,
)
from codingjepa.data.secrets_scan import SecretHit
from codingjepa.errors import AuditGateError


@dataclass
class _FakeChunk:
    chunk_id: str
    source_normalized: str = ""


def _make_clean_chunks(n: int = 100) -> list[_FakeChunk]:
    return [_FakeChunk(f"c{i}", f"def f_{i}(x):\n    return x + {i}\n") for i in range(n)]


# ---------------------------------------------------------------------------
# compute_audit — happy path
# ---------------------------------------------------------------------------


def test_compute_audit_passes_gates_for_clean_data() -> None:
    """Clean inputs satisfy every gate (compile/dedup/license/secrets)."""

    chunks = _make_clean_chunks(100)
    result = compute_audit(
        repo="pandas-dev/pandas",
        commit_sha="0" * 40,
        license_spdx="BSD-3-Clause",
        chunks=chunks,
        secret_hits=[],
    )
    assert result.passes_compile_gate
    assert result.passes_dedup_gate
    assert result.passes_license_gate
    assert result.passes_secrets_gate
    assert result.passes_all_gates


def test_compute_audit_records_repo_metadata() -> None:
    """The repo/commit/license fields round-trip from the inputs."""

    chunks = _make_clean_chunks(10)
    result = compute_audit(
        repo="psf/black",
        commit_sha="a" * 40,
        license_spdx="MIT",
        chunks=chunks,
    )
    assert result.repo == "psf/black"
    assert result.commit_sha == "a" * 40
    assert result.license_spdx == "MIT"
    assert result.chunk_count == 10


# ---------------------------------------------------------------------------
# Gate-failure cases
# ---------------------------------------------------------------------------


def test_compute_audit_fails_compile_gate() -> None:
    """compile_ok_rate < 0.95 → compile gate fails (RFC-0002 §D2)."""

    chunks = _make_clean_chunks(100)
    # Pretend 10 chunks failed compile (drop_rate_parse_fail = 0.10).
    result = compute_audit(
        repo="pandas-dev/pandas",
        commit_sha="0" * 40,
        license_spdx="BSD-3-Clause",
        chunks=chunks,
        dedup_result=_FakeDedup(dup_rate=0.0),
        secret_hits=[],
        parse_fail_count=10,
    )
    assert not result.passes_compile_gate
    assert not result.passes_all_gates


def test_compute_audit_fails_dedup_gate() -> None:
    """duplication_rate >= 0.30 → dedup gate fails."""

    chunks = _make_clean_chunks(100)
    result = compute_audit(
        repo="pandas-dev/pandas",
        commit_sha="0" * 40,
        license_spdx="BSD-3-Clause",
        chunks=chunks,
        dedup_result=_FakeDedup(dup_rate=0.35),
        secret_hits=[],
    )
    assert not result.passes_dedup_gate
    assert not result.passes_all_gates


def test_compute_audit_fails_secrets_gate() -> None:
    """Any secret hit → secrets gate fails."""

    chunks = _make_clean_chunks(50)
    result = compute_audit(
        repo="pandas-dev/pandas",
        commit_sha="0" * 40,
        license_spdx="BSD-3-Clause",
        chunks=chunks,
        secret_hits=[
            SecretHit(chunk_id="x", pattern_name="aws_access_key_id", match_preview="AKIA"),
        ],
    )
    assert not result.passes_secrets_gate
    assert not result.passes_all_gates


def test_compute_audit_fails_license_gate() -> None:
    """Licenses outside ALLOWED_LICENSES fail the license gate."""

    chunks = _make_clean_chunks(50)
    result = compute_audit(
        repo="bogus/repo",
        commit_sha="0" * 40,
        license_spdx="GPL-3.0",  # not in allowed set
        chunks=chunks,
    )
    assert not result.passes_license_gate
    assert not result.passes_all_gates


# ---------------------------------------------------------------------------
# write_audit / load_audit
# ---------------------------------------------------------------------------


def test_write_audit_creates_valid_json(tmp_path: Path) -> None:
    """The audit JSON file is written under data/audit/<owner>__<name>.json."""

    chunks = _make_clean_chunks(5)
    result = compute_audit(
        repo="psf/black",
        commit_sha="b" * 40,
        license_spdx="MIT",
        chunks=chunks,
    )
    out_path = write_audit(result, output_dir=tmp_path)
    assert out_path.exists()
    assert out_path.name == "psf__black.json"

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["repo"] == "psf/black"
    assert payload["commit_sha"] == "b" * 40
    assert payload["license_spdx"] == "MIT"


def test_write_audit_validates_against_schema(tmp_path: Path) -> None:
    """The written file passes data/schemas/audit.schema.json."""

    chunks = _make_clean_chunks(5)
    result = compute_audit(
        repo="psf/black",
        commit_sha="b" * 40,
        license_spdx="MIT",
        chunks=chunks,
    )
    out_path = write_audit(result, output_dir=tmp_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    validate_record(payload, load_schema("audit"))


def test_load_audit_round_trip(tmp_path: Path) -> None:
    """``load_audit`` reconstructs the AuditResult written by ``write_audit``."""

    chunks = _make_clean_chunks(3)
    original = compute_audit(
        repo="psf/black",
        commit_sha="c" * 40,
        license_spdx="MIT",
        chunks=chunks,
    )
    out_path = write_audit(original, output_dir=tmp_path)
    loaded = load_audit(out_path)
    assert loaded.repo == original.repo
    assert loaded.commit_sha == original.commit_sha
    assert loaded.license_spdx == original.license_spdx
    assert loaded.chunk_count == original.chunk_count


# ---------------------------------------------------------------------------
# check_all_gates
# ---------------------------------------------------------------------------


def test_check_all_gates_ok() -> None:
    """All gates passing → returns True."""

    chunks = _make_clean_chunks(5)
    result = compute_audit(
        repo="pandas-dev/pandas",
        commit_sha="0" * 40,
        license_spdx="BSD-3-Clause",
        chunks=chunks,
    )
    assert check_all_gates([result]) is True


def test_check_all_gates_raises_on_failure() -> None:
    """Any gate failing → raises AuditGateError."""

    chunks = _make_clean_chunks(5)
    result = compute_audit(
        repo="bogus/repo",
        commit_sha="0" * 40,
        license_spdx="GPL-3.0",  # license gate fails
        chunks=chunks,
    )
    with pytest.raises(AuditGateError):
        check_all_gates([result])


def test_allowed_licenses_set() -> None:
    """ALLOWED_LICENSES is the documented closed set."""

    assert ALLOWED_LICENSES == frozenset({"PSF-2.0", "BSD-3-Clause", "MIT", "Apache-2.0"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeDedup:
    """Stand-in for dedup.DedupResult exposing only ``duplication_rate``."""

    dup_rate: float

    @property
    def duplication_rate(self) -> float:
        return self.dup_rate
