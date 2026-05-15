"""Tests for codingjepa.data.splits. See RFC-0014 §D7 and SYSTEM-SPEC.md §3.2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from codingjepa._jsonschema import load_schema, validate_record
from codingjepa.data.splits import (
    SPLIT_ASSIGNMENTS,
    SplitResult,
    assign_splits,
    detect_leakage,
    write_leakage_report,
    write_splits_lock,
)
from codingjepa.errors import SplitContractViolation


@dataclass
class _FakeChunk:
    """Minimal stand-in for codingjepa.data.chunker.Chunk used by the splits tests."""

    chunk_id: str
    repo: str
    source_normalized: str = ""


# ---------------------------------------------------------------------------
# SPLIT_ASSIGNMENTS
# ---------------------------------------------------------------------------


def test_split_assignments_has_10_repos() -> None:
    """RFC-0002 §D1: the locked v1 corpus is exactly 10 repositories."""

    assert len(SPLIT_ASSIGNMENTS) == 10


def test_split_assignments_distribution() -> None:
    """6 train, 2 val, 2 test per RFC-0014 §D7."""

    splits = list(SPLIT_ASSIGNMENTS.values())
    assert splits.count("train") == 6
    assert splits.count("val") == 2
    assert splits.count("test") == 2


def test_split_assignments_known_repos() -> None:
    """The named repos are pinned to their documented splits."""

    assert SPLIT_ASSIGNMENTS["pandas-dev/pandas"] == "train"
    assert SPLIT_ASSIGNMENTS["scikit-learn/scikit-learn"] == "train"
    assert SPLIT_ASSIGNMENTS["huggingface/transformers"] == "train"
    assert SPLIT_ASSIGNMENTS["pytest-dev/pytest"] == "train"
    assert SPLIT_ASSIGNMENTS["fastapi/fastapi"] == "train"
    assert SPLIT_ASSIGNMENTS["sqlalchemy/sqlalchemy"] == "train"
    assert SPLIT_ASSIGNMENTS["django/django"] == "val"
    assert SPLIT_ASSIGNMENTS["python/mypy"] == "val"
    assert SPLIT_ASSIGNMENTS["psf/black"] == "test"
    assert SPLIT_ASSIGNMENTS["python/cpython"] == "test"


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------


def test_assign_splits_by_repo() -> None:
    """Chunks are routed to the split that owns their repo."""

    chunks = [
        _FakeChunk("a", "pandas-dev/pandas"),
        _FakeChunk("b", "django/django"),
        _FakeChunk("c", "psf/black"),
    ]
    result = assign_splits(chunks)
    assert {c.chunk_id for c in result["train"]} == {"a"}
    assert {c.chunk_id for c in result["val"]} == {"b"}
    assert {c.chunk_id for c in result["test"]} == {"c"}


def test_assign_splits_all_repos_covered() -> None:
    """One chunk per repo: every repo lands in exactly one of the three splits."""

    chunks = [_FakeChunk(f"id-{i}", repo) for i, repo in enumerate(SPLIT_ASSIGNMENTS)]
    result = assign_splits(chunks)
    total = len(result["train"]) + len(result["val"]) + len(result["test"])
    assert total == len(chunks)
    assert len(result["train"]) == 6
    assert len(result["val"]) == 2
    assert len(result["test"]) == 2


def test_assign_splits_deterministic() -> None:
    """Two invocations on the same input give the same per-split chunk ids."""

    chunks = [
        _FakeChunk("a", "pandas-dev/pandas"),
        _FakeChunk("b", "django/django"),
        _FakeChunk("c", "psf/black"),
    ]
    a = assign_splits(chunks)
    b = assign_splits(chunks)
    for split in ("train", "val", "test"):
        assert [c.chunk_id for c in a[split]] == [c.chunk_id for c in b[split]]


def test_assign_splits_unknown_repo_raises() -> None:
    """Unknown repos are not silently routed — would defeat the by-repo guarantee."""

    chunks = [_FakeChunk("a", "unknown/repo")]
    with pytest.raises(SplitContractViolation):
        assign_splits(chunks)


# ---------------------------------------------------------------------------
# detect_leakage
# ---------------------------------------------------------------------------


def test_detect_leakage_finds_identical_chunks() -> None:
    """Identical text across splits is flagged as cross-split leakage."""

    shared = "def foo():\n    return 1 + 2 + 3 + 4 + 5\n"
    train = _FakeChunk("t1", "pandas-dev/pandas", source_normalized=shared)
    test = _FakeChunk("x1", "psf/black", source_normalized=shared)
    pairs = detect_leakage({"train": [train], "val": [], "test": [test]})
    assert len(pairs) == 1
    assert set(pairs[0]) == {"t1", "x1"}


def test_detect_leakage_no_false_positives() -> None:
    """Completely different content does not trigger leakage."""

    train = _FakeChunk(
        "t1",
        "pandas-dev/pandas",
        source_normalized="def foo():\n    return 1\n",
    )
    test = _FakeChunk(
        "x1",
        "psf/black",
        source_normalized=(
            "class Quux:\n" "    def render(self, payload):\n" "        return payload.upper()\n"
        ),
    )
    pairs = detect_leakage({"train": [train], "val": [], "test": [test]})
    assert pairs == []


def test_detect_leakage_same_split_ignored() -> None:
    """Duplicates inside the same split are not 'cross-split' leakage."""

    same = "def foo():\n    return 42\n"
    a = _FakeChunk("a", "pandas-dev/pandas", source_normalized=same)
    b = _FakeChunk("b", "pandas-dev/pandas", source_normalized=same)
    pairs = detect_leakage({"train": [a, b], "val": [], "test": []})
    assert pairs == []


def test_detect_leakage_respects_threshold() -> None:
    """Pairs below ``threshold`` Jaccard similarity are not reported."""

    train = _FakeChunk(
        "t1",
        "pandas-dev/pandas",
        source_normalized="def foo():\n    return 1\n",
    )
    test = _FakeChunk(
        "x1",
        "psf/black",
        source_normalized="def foo():\n    return 1\n",
    )
    # Identical → max similarity, always above threshold.
    pairs = detect_leakage({"train": [train], "val": [], "test": [test]}, threshold=0.85)
    assert len(pairs) == 1
    # threshold=1.01 is unreachable → must produce no hits.
    pairs = detect_leakage({"train": [train], "val": [], "test": [test]}, threshold=1.01)
    assert pairs == []


# ---------------------------------------------------------------------------
# write_splits_lock
# ---------------------------------------------------------------------------


def test_write_splits_lock_creates_file(tmp_path: Path) -> None:
    """The lock file is written and is valid against splits.schema.json."""

    chunks = [
        _FakeChunk("a", "pandas-dev/pandas"),
        _FakeChunk("b", "django/django"),
        _FakeChunk("c", "psf/black"),
    ]
    split_chunks = assign_splits(chunks)
    out_path = tmp_path / "splits.lock.json"
    write_splits_lock(split_chunks, output_path=out_path)

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    validate_record(payload, load_schema("splits"))
    assert payload["schema_version"] == "v1"
    assert payload["by_repo"]["pandas-dev/pandas"] == "train"
    assert payload["by_repo"]["django/django"] == "val"
    assert payload["by_repo"]["psf/black"] == "test"


def test_write_splits_lock_hash_is_sha256(tmp_path: Path) -> None:
    """``split_hash`` is a 64-character lowercase hex digest."""

    chunks = [_FakeChunk("a", "pandas-dev/pandas")]
    split_chunks = assign_splits(chunks)
    out_path = tmp_path / "splits.lock.json"
    write_splits_lock(split_chunks, output_path=out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    h = payload["split_hash"]
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# write_leakage_report
# ---------------------------------------------------------------------------


def test_write_leakage_report_creates_file(tmp_path: Path) -> None:
    """The cross-split leakage report is written and schema-valid."""

    out_path = tmp_path / "cross_split_leakage.json"
    write_leakage_report([], output_path=out_path)

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    validate_record(payload, load_schema("cross_split_leakage"))
    assert payload["remaining_crossings"] == []
    assert payload["resolved"] == []


def test_write_leakage_report_records_pairs(tmp_path: Path) -> None:
    """Leakage pairs surface as resolved entries (test side dropped per RFC-0014 §D6)."""

    out_path = tmp_path / "cross_split_leakage.json"
    write_leakage_report([("train_chunk", "test_chunk")], output_path=out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    validate_record(payload, load_schema("cross_split_leakage"))
    assert payload["remaining_crossings"] == []
    assert len(payload["resolved"]) == 1
    entry = payload["resolved"][0]
    assert entry["chunk_id_a"] == "train_chunk"
    assert entry["chunk_id_b"] == "test_chunk"
    assert entry["resolution"] in {"dropped_a", "dropped_b", "moved_to_train"}


# ---------------------------------------------------------------------------
# SplitResult dataclass smoke
# ---------------------------------------------------------------------------


def test_split_result_dataclass_smoke() -> None:
    """The SplitResult dataclass is constructible with the documented fields."""

    sr = SplitResult(train=[], val=[], test=[], cross_split_leakage=0)
    assert sr.train == []
    assert sr.val == []
    assert sr.test == []
    assert sr.cross_split_leakage == 0
