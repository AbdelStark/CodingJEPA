"""build_pool is deterministic given (candidates, size, seed). RFC-0010 §D9."""

from __future__ import annotations

import pathlib

import pytest

from codingjepa.errors import UsageError
from codingjepa.eval.pools import build_pool, load_pool

CANDIDATES = [f"chk-{i:04d}" for i in range(50)]


def test_same_seed_produces_identical_pool(tmp_path: pathlib.Path) -> None:
    a = build_pool("CJ-RET-100", CANDIDATES, size=10, seed=42, lock_path=tmp_path / "a.lock.json")
    b = build_pool("CJ-RET-100", CANDIDATES, size=10, seed=42, lock_path=tmp_path / "b.lock.json")
    assert a.chunk_ids == b.chunk_ids
    assert a.pool_hash == b.pool_hash


def test_input_order_does_not_change_result(tmp_path: pathlib.Path) -> None:
    """Candidates are sorted internally, so the caller's order is irrelevant."""

    a = build_pool("CJ-RET-100", CANDIDATES, size=10, seed=42, lock_path=tmp_path / "a.lock.json")
    shuffled = list(CANDIDATES)
    shuffled.reverse()
    b = build_pool("CJ-RET-100", shuffled, size=10, seed=42, lock_path=tmp_path / "b.lock.json")
    assert a.chunk_ids == b.chunk_ids
    assert a.pool_hash == b.pool_hash


def test_duplicates_in_candidates_are_deduped(tmp_path: pathlib.Path) -> None:
    a = build_pool("CJ-RET-100", CANDIDATES, size=10, seed=42, lock_path=tmp_path / "a.lock.json")
    duplicated = CANDIDATES + CANDIDATES[:25]
    b = build_pool("CJ-RET-100", duplicated, size=10, seed=42, lock_path=tmp_path / "b.lock.json")
    assert a.chunk_ids == b.chunk_ids


def test_different_seed_produces_different_pool(tmp_path: pathlib.Path) -> None:
    a = build_pool("CJ-RET-100", CANDIDATES, size=10, seed=1, lock_path=tmp_path / "a.lock.json")
    b = build_pool("CJ-RET-100", CANDIDATES, size=10, seed=2, lock_path=tmp_path / "b.lock.json")
    assert a.pool_hash != b.pool_hash


def test_different_size_produces_different_pool(tmp_path: pathlib.Path) -> None:
    a = build_pool("CJ-RET-100", CANDIDATES, size=10, seed=42, lock_path=tmp_path / "a.lock.json")
    b = build_pool("CJ-RET-100", CANDIDATES, size=20, seed=42, lock_path=tmp_path / "b.lock.json")
    assert a.pool_hash != b.pool_hash
    assert len(b.chunk_ids) == 20


def test_chunk_ids_are_sorted(tmp_path: pathlib.Path) -> None:
    lock = build_pool(
        "CJ-RET-100", CANDIDATES, size=10, seed=42, lock_path=tmp_path / "out.lock.json"
    )
    assert list(lock.chunk_ids) == sorted(lock.chunk_ids)


def test_refuses_overwrite(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "existing.lock.json"
    build_pool("CJ-RET-100", CANDIDATES, size=10, seed=42, lock_path=path)
    with pytest.raises(UsageError) as exc:
        build_pool("CJ-RET-100", CANDIDATES, size=10, seed=43, lock_path=path)
    assert "refusing to overwrite" in str(exc.value)


def test_size_zero_rejected(tmp_path: pathlib.Path) -> None:
    with pytest.raises(UsageError):
        build_pool("CJ-RET-100", CANDIDATES, size=0, seed=42, lock_path=tmp_path / "x.lock.json")


def test_size_exceeds_candidates(tmp_path: pathlib.Path) -> None:
    with pytest.raises(UsageError) as exc:
        build_pool(
            "CJ-RET-100", CANDIDATES[:5], size=10, seed=42, lock_path=tmp_path / "x.lock.json"
        )
    assert exc.value.context["available"] == 5


def test_round_trip_through_load_pool(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "round.lock.json"
    written = build_pool("CJ-RET-100", CANDIDATES, size=10, seed=42, lock_path=path)
    read = load_pool(path)
    assert written == read


def test_load_pool_detects_tamper(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "round.lock.json"
    build_pool("CJ-RET-100", CANDIDATES, size=10, seed=42, lock_path=path)
    # Tamper: mutate one chunk_id without recomputing the hash.
    content = path.read_text(encoding="utf-8")
    tampered = content.replace("chk-00", "chk-XX", 1)
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(UsageError) as exc:
        load_pool(path)
    assert "tampered" in str(exc.value).lower()
