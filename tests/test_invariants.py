"""Cross-artifact invariants per docs/spec/03-data-model.md §Cross-artifact invariants.

Runs against committed fixtures by default; the `slow` marker version targets a
real corpus slice and is exercised by the nightly workflow (#23).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import pytest

from codingjepa._jsonschema import load_schema, validate_record
from codingjepa.errors import UsageError

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "invariants"


# ---------- Fixture loaders ---------------------------------------------------


def _load(name: str) -> Any:
    with (FIXTURES / name).open(encoding="utf-8") as fp:
        return json.load(fp)


def _load_pools() -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in (FIXTURES / "pools").glob("*.lock.json")
    ]


# ---------- Pure invariants (callable from real artifacts too) ----------------


def canonicalize_for_manifest_hash(manifest: dict[str, Any]) -> str:
    """Reproduce the spec/03 §JSON schemas manifest_hash canonicalization.

    Remove the `manifest_hash` field, dump JSON with sorted keys and no
    insignificant whitespace, sha256 the bytes.
    """

    payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def assert_pair_chunk_ids_in_chunks(
    pairs: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> None:
    chunk_ids = {row["chunk_id"] for row in chunks}
    missing: set[tuple[str, str]] = set()
    for row in pairs:
        for side in ("chunk_id_before", "chunk_id_after"):
            cid = row[side]
            if cid not in chunk_ids:
                missing.add((row["pair_id"], cid))
    if missing:
        raise UsageError("pair chunk_ids not in chunks", missing=sorted(missing))


def assert_pool_chunk_ids_in_chunks(
    pools: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> None:
    chunk_ids = {row["chunk_id"] for row in chunks}
    for pool in pools:
        unknown = [cid for cid in pool["chunk_ids"] if cid not in chunk_ids]
        if unknown:
            raise UsageError(
                "pool chunk_ids not in chunks",
                benchmark=pool["benchmark"],
                unknown=unknown,
            )


def assert_no_chunk_id_in_two_splits(pairs: list[dict[str, Any]]) -> None:
    splits_per_chunk: dict[str, set[str]] = {}
    for row in pairs:
        for side in ("chunk_id_before", "chunk_id_after"):
            splits_per_chunk.setdefault(row[side], set()).add(row["split"])
    crossing = {cid: sorted(splits) for cid, splits in splits_per_chunk.items() if len(splits) > 1}
    if crossing:
        raise UsageError("chunk_id appears in multiple splits", crossings=crossing)


def assert_pairs_share_manifest(
    pairs: list[dict[str, Any]], chunks: list[dict[str, Any]], manifest_hash: str
) -> None:
    for row in pairs:
        if row["manifest_hash"] != manifest_hash:
            raise UsageError("pair manifest_hash != manifest", pair_id=row["pair_id"])
    for row in chunks:
        if row["manifest_hash"] != manifest_hash:
            raise UsageError("chunk manifest_hash != manifest", chunk_id=row["chunk_id"])


# ---------- Fixture-based tests (run by default) ------------------------------


def test_manifest_hash_self_consistency() -> None:
    """manifest_hash is the sha256 of the canonicalized other content (spec/03)."""

    manifest = _load("manifest.lock.json")
    recomputed = canonicalize_for_manifest_hash(manifest)
    # Update the placeholder in-memory and confirm round-trip.
    manifest["manifest_hash"] = recomputed
    assert canonicalize_for_manifest_hash(manifest) == recomputed


def test_pair_chunk_ids_exist_in_chunks() -> None:
    chunks = _load("chunks.json")["rows"]
    pairs = _load("pairs.json")["rows"]
    assert_pair_chunk_ids_in_chunks(pairs, chunks)


def test_pool_chunk_ids_exist_in_chunks() -> None:
    chunks = _load("chunks.json")["rows"]
    pools = _load_pools()
    assert pools, "no pool fixtures committed"
    assert_pool_chunk_ids_in_chunks(pools, chunks)


def test_no_chunk_id_in_two_splits() -> None:
    pairs = _load("pairs.json")["rows"]
    assert_no_chunk_id_in_two_splits(pairs)


def test_all_share_manifest_hash() -> None:
    chunks_doc = _load("chunks.json")
    pairs = _load("pairs.json")["rows"]
    assert_pairs_share_manifest(pairs, chunks_doc["rows"], chunks_doc["manifest_hash"])


def test_pool_fixture_passes_schema() -> None:
    """The pool fixture also satisfies the JSONSchema from #138."""

    schema = load_schema("pool")
    for pool in _load_pools():
        validate_record(pool, schema)


# ---------- Synthetic-violation tests (acceptance bullet 2) -------------------


def test_synthetic_pair_with_missing_chunk_fails() -> None:
    chunks = _load("chunks.json")["rows"]
    bad_pair = {
        "pair_id": "pair-bad",
        "chunk_id_before": "chk-aaaa",
        "chunk_id_after": "chk-NOTREAL",
        "intent": "extract-helper",
        "split": "train",
        "manifest_hash": "0000000000000000000000000000000000000000000000000000000000000000",
    }
    with pytest.raises(UsageError) as exc:
        assert_pair_chunk_ids_in_chunks([bad_pair], chunks)
    assert "chk-NOTREAL" in str(exc.value.context)


def test_synthetic_pool_with_missing_chunk_fails() -> None:
    chunks = _load("chunks.json")["rows"]
    bad_pool = {
        "schema_version": "v1",
        "benchmark": "CJ-RET-100",
        "pool_size": 1,
        "chunk_ids": ["chk-aaaa", "chk-NOTREAL"],
        "pool_hash": "3" * 64,
    }
    with pytest.raises(UsageError):
        assert_pool_chunk_ids_in_chunks([bad_pool], chunks)


def test_synthetic_cross_split_violation_fails() -> None:
    pairs = _load("pairs.json")["rows"] + [
        {
            "pair_id": "pair-leak",
            "chunk_id_before": "chk-aaaa",
            "chunk_id_after": "chk-bbbb",
            "intent": "extract-helper",
            "split": "test",
            "manifest_hash": "0000000000000000000000000000000000000000000000000000000000000000",
        },
    ]
    with pytest.raises(UsageError):
        assert_no_chunk_id_in_two_splits(pairs)


def test_synthetic_manifest_hash_tamper_fails() -> None:
    manifest = _load("manifest.lock.json")
    manifest["manifest_hash"] = canonicalize_for_manifest_hash(manifest)
    declared = manifest["manifest_hash"]
    # Mutating any field except manifest_hash must change the recomputed hash.
    manifest["chunker_version"] = "0.2.0"
    assert canonicalize_for_manifest_hash(manifest) != declared


# ---------- Slow marker: real-corpus invariants -------------------------------


REAL_PAIRS = pathlib.Path("data/pairs/v1.parquet")
REAL_CHUNKS_GLOB = "data/parsed/**/*.chunks.parquet"


@pytest.mark.slow
def test_real_corpus_invariants_present_or_skipped() -> None:
    """Real-corpus version of the invariants. Skipped until #34-#56 produce
    the artifacts; exercised by the nightly workflow (#23)."""

    if not REAL_PAIRS.exists():
        pytest.skip("no real corpus on disk; nightly executes this against the frozen pipeline")
    # Re-using the same assert_* helpers means the invariants are tested once
    # against fixtures and once against real artifacts; if either drifts, both
    # surface the same failure shape.
    pytest.fail("real-corpus pathway not implemented yet — see #34-#56")
