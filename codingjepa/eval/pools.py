"""Deterministic eval-pool construction. RFC-0010 §D9.

Every retrieval benchmark records its exact pool composition in
`eval/pools/<benchmark>.lock.json`. The lock file is **content-addressed**:
the `pool_hash` is the sha256 over the sorted chunk_ids and is the join
key for downstream auditing.

The pool is **frozen** once written — `build_pool` refuses to overwrite an
existing lock file. Changing the seed, the candidate set, or the size
re-versions the benchmark (a new `<benchmark>.v2.lock.json` is created by
the caller; this module only emits a single lock file per call).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
from dataclasses import dataclass

from codingjepa._jsonschema import load_schema, validate_record
from codingjepa.errors import UsageError

SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class PoolLock:
    schema_version: str
    benchmark: str
    pool_size: int
    chunk_ids: tuple[str, ...]
    pool_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "benchmark": self.benchmark,
            "pool_size": self.pool_size,
            "chunk_ids": list(self.chunk_ids),
            "pool_hash": self.pool_hash,
        }


def build_pool(
    benchmark: str,
    candidate_chunk_ids: list[str],
    *,
    size: int,
    seed: int,
    lock_path: pathlib.Path | str,
) -> PoolLock:
    """Sample `size` chunk_ids deterministically and emit the lock file.

    Determinism contract:
      * candidates are deduplicated and sorted before any sampling, so the
        result is independent of caller-supplied ordering;
      * `random.Random(seed)` is the only entropy source;
      * the output `chunk_ids` are sorted (the sampled set is unordered);
      * `pool_hash = sha256(",".join(sorted_chunk_ids))`.

    Refuses to overwrite an existing `lock_path`; caller must remove the
    file or use a new path (a "v2" re-version per RFC-0010 §D9).
    """

    if size < 1:
        raise UsageError("pool size must be >= 1", size=size)

    deduped = sorted(set(candidate_chunk_ids))
    if size > len(deduped):
        raise UsageError(
            "pool size exceeds available candidates",
            size=size,
            available=len(deduped),
        )

    rng = random.Random(seed)
    sampled = rng.sample(deduped, k=size)
    sampled_sorted = tuple(sorted(sampled))
    pool_hash = _hash_pool(sampled_sorted)
    lock = PoolLock(
        schema_version=SCHEMA_VERSION,
        benchmark=benchmark,
        pool_size=size,
        chunk_ids=sampled_sorted,
        pool_hash=pool_hash,
    )

    path = pathlib.Path(lock_path)
    if path.exists():
        raise UsageError(
            "pool lock file already exists; refusing to overwrite",
            path=str(path),
            suggestion="re-version with a new path (e.g., <benchmark>.v2.lock.json)",
        )

    schema = load_schema("pool")
    validate_record(lock.to_dict(), schema)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def load_pool(lock_path: pathlib.Path | str) -> PoolLock:
    """Read and validate an existing pool lock file."""

    path = pathlib.Path(lock_path)
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)

    schema = load_schema("pool")
    validate_record(payload, schema)

    chunk_ids = tuple(payload["chunk_ids"])
    expected_hash = _hash_pool(chunk_ids)
    if expected_hash != payload["pool_hash"]:
        raise UsageError(
            "pool_hash mismatch — lock file has been tampered with",
            path=str(path),
            expected=expected_hash,
            actual=payload["pool_hash"],
        )

    return PoolLock(
        schema_version=payload["schema_version"],
        benchmark=payload["benchmark"],
        pool_size=payload["pool_size"],
        chunk_ids=chunk_ids,
        pool_hash=payload["pool_hash"],
    )


def _hash_pool(sorted_chunk_ids: tuple[str, ...]) -> str:
    body = ",".join(sorted_chunk_ids)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


__all__ = ["PoolLock", "SCHEMA_VERSION", "build_pool", "load_pool"]
