"""Exact and near deduplication for CodingJEPA chunks (RFC-0014 §D6).

This module is the dedup half of the data pipeline. It assumes the
caller has already produced a stream of :class:`~codingjepa.data.chunker.Chunk`
objects (each with a populated ``source_normalized`` field) and returns
the kept-survivor list plus a small report.

Pipeline:

1. **Exact dedup** by SHA-256 of ``source_normalized`` — the first
   occurrence wins, every subsequent identical chunk is dropped.
2. **Near dedup** via MinHash + LSH on character 5-gram shingles of
   ``source_normalized``. Chunks in the same LSH bucket are checked
   for actual Jaccard similarity against the supplied threshold; if
   they exceed it, only one representative survives (chosen by
   ``min(chunk_id)`` so the choice is deterministic).

We do **not** depend on ``datasketch`` (it is not in the project's
dependency manifest); the MinHash sketch is implemented with classic
universal hash functions ``(a * h(shingle) + b) % large_prime`` for a
fixed seeded pair list. This keeps the sketch reproducible across
machines and across Python versions (we never rely on ``hash()``'s
random salt — only on :func:`hashlib.sha1`).

The output of :func:`dedup_pipeline` is consumed by RFC-0014 §D6
manifest writers via :func:`write_dedup_report`, which dumps the
counters as JSON for ``data/audit/dedup.json``.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# We don't import codingjepa.data.chunker.Chunk here to keep this module
# free of pipeline-stage dependencies (dedup runs before tokenization
# and before any other consumer of Chunk). Each chunk is duck-typed:
# we only ever read ``source_normalized`` and ``chunk_id``.
Chunkish = Any

# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #

# A 64-bit Mersenne prime — large enough to make universal hash
# collisions astronomically rare for our shingle alphabet, and stays
# inside Python's fast int range for arithmetic. The MinHash sketch
# uses this prime as the universal-hash modulus; the per-slot
# "infinity" sentinel must therefore be >= prime so any real value
# beats it.
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = _MERSENNE_PRIME  # sentinel for "no shingle seen yet"
_SHINGLE_SIZE = 5

# Deterministic RNG seed for the hash function pair table. Changing
# this value invalidates every previously-computed MinHash signature,
# so it is kept fixed.
_HASH_SEED = 0xC0DE_CAFE


# --------------------------------------------------------------------------- #
# result                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class DedupResult:
    """Counters from the full dedup pipeline (RFC-0014 §D6 audit output)."""

    total_chunks: int
    exact_duplicates: int
    near_duplicates: int
    retained_chunks: int
    dedup_rate: float  # (exact + near) / total


# --------------------------------------------------------------------------- #
# exact dedup                                                                 #
# --------------------------------------------------------------------------- #


def exact_dedup(chunks: list[Chunkish]) -> tuple[list[Chunkish], int]:
    """Drop chunks whose ``source_normalized`` SHA-256 has already been seen.

    First occurrence wins (stable order). Returns ``(kept, removed)``.
    """

    seen: set[str] = set()
    kept: list[Chunkish] = []
    removed = 0
    for chunk in chunks:
        digest = hashlib.sha256(chunk.source_normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            removed += 1
            continue
        seen.add(digest)
        kept.append(chunk)
    return kept, removed


# --------------------------------------------------------------------------- #
# MinHash                                                                     #
# --------------------------------------------------------------------------- #


def _hash_pairs(num_hashes: int) -> list[tuple[int, int]]:
    """Generate ``num_hashes`` deterministic (a, b) pairs for universal hashing.

    Cached by ``num_hashes`` via :func:`functools.lru_cache` would be
    nicer but the caller may pass different lengths, so we just rebuild
    the list each time — cheap compared to the actual hashing work.
    """

    rng = random.Random(_HASH_SEED)
    pairs: list[tuple[int, int]] = []
    for _ in range(num_hashes):
        a = rng.randint(1, _MERSENNE_PRIME - 1)
        b = rng.randint(0, _MERSENNE_PRIME - 1)
        pairs.append((a, b))
    return pairs


def _shingles(text: str, size: int = _SHINGLE_SIZE) -> set[str]:
    """Return the set of character ``size``-grams of ``text``.

    Sets (not multisets) so the resulting Jaccard estimate matches
    what classic MinHash promises.
    """

    if len(text) < size:
        # For very short text, treat the whole thing as a single
        # shingle so downstream code doesn't see an empty signature.
        return {text} if text else set()
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def _hash_shingle(shingle: str) -> int:
    """Map a shingle to a 32-bit integer via SHA-1 (deterministic across runs)."""

    digest = hashlib.sha1(shingle.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def minhash_signature(text: str, num_hashes: int = 128) -> list[int]:
    """Compute a MinHash signature of length ``num_hashes`` over ``text``.

    Uses character 5-gram shingles and the universal hash family
    ``h_i(x) = (a_i * x + b_i) mod p``. Returns a list of integers, one
    per hash function, holding the minimum hashed-shingle value seen.
    """

    shingles = _shingles(text)
    pairs = _hash_pairs(num_hashes)
    if not shingles:
        # An empty signature still needs the right shape so callers
        # don't have to special-case zero-length text.
        return [_MAX_HASH] * num_hashes

    hashed = [_hash_shingle(s) for s in shingles]
    signature: list[int] = []
    for a, b in pairs:
        min_val = _MAX_HASH
        for h in hashed:
            val = (a * h + b) % _MERSENNE_PRIME
            if val < min_val:
                min_val = val
        signature.append(min_val)
    return signature


def _jaccard_from_signatures(a: list[int], b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures.

    The classical estimator: fraction of slots that are equal.
    """

    if not a or not b or len(a) != len(b):
        return 0.0
    matches = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    return matches / len(a)


# --------------------------------------------------------------------------- #
# LSH                                                                         #
# --------------------------------------------------------------------------- #


def lsh_dedup(
    chunks: list[Chunkish],
    *,
    threshold: float = 0.85,
    num_hashes: int = 128,
    num_bands: int = 32,
) -> tuple[list[Chunkish], int]:
    """Collapse chunks that are MinHash-similar above ``threshold`` into one.

    Uses LSH banding: the signature is split into ``num_bands`` bands of
    ``num_hashes / num_bands`` rows. Two signatures share a candidate
    pair if any band's tuple of rows is identical. Candidate pairs are
    then verified with the actual Jaccard estimate before any chunk is
    removed.

    Determinism: for each cluster of near-duplicates the survivor is
    ``min(chunk_id)`` (lexicographic). This makes the output
    reproducible regardless of input order.
    """

    if not chunks:
        return [], 0
    if num_hashes % num_bands != 0:
        raise ValueError(f"num_hashes ({num_hashes}) must be divisible by num_bands ({num_bands})")

    rows_per_band = num_hashes // num_bands

    # Compute signatures once.
    signatures: list[list[int]] = [
        minhash_signature(c.source_normalized, num_hashes) for c in chunks
    ]

    # Build per-band buckets: bucket key = (band_index, band_tuple).
    # Each bucket maps to the indices of chunks that hashed into it.
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for idx, sig in enumerate(signatures):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            band = tuple(sig[start : start + rows_per_band])
            buckets.setdefault((band_idx, band), []).append(idx)

    # Union-find over chunk indices: members of the same near-dup
    # cluster are joined whenever any band agrees AND their Jaccard
    # passes the threshold.
    parent = list(range(len(chunks)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Stable choice: lower index becomes the parent.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Iterate band buckets, doing pairwise Jaccard checks only inside
    # buckets (and only when they have at least 2 members).
    seen_pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i, ai in enumerate(members):
            for bj in members[i + 1 :]:
                pair = (ai, bj) if ai < bj else (bj, ai)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                jac = _jaccard_from_signatures(signatures[ai], signatures[bj])
                if jac >= threshold:
                    union(ai, bj)

    # Within each connected component pick the lexicographically
    # smallest chunk_id as the survivor.
    cluster_to_indices: dict[int, list[int]] = {}
    for i in range(len(chunks)):
        cluster_to_indices.setdefault(find(i), []).append(i)

    keep_mask = [False] * len(chunks)
    for indices in cluster_to_indices.values():
        survivor = min(indices, key=lambda i: chunks[i].chunk_id)
        keep_mask[survivor] = True

    kept = [c for c, m in zip(chunks, keep_mask, strict=True) if m]
    removed = len(chunks) - len(kept)
    return kept, removed


# --------------------------------------------------------------------------- #
# pipeline + report                                                           #
# --------------------------------------------------------------------------- #


def dedup_pipeline(
    chunks: list[Chunkish],
    *,
    threshold: float = 0.85,
) -> tuple[list[Chunkish], DedupResult]:
    """Run exact dedup → near dedup and return ``(kept, DedupResult)``.

    The :class:`DedupResult` records ``total_chunks`` (input length),
    ``exact_duplicates`` and ``near_duplicates`` (counts of removals
    by each stage), ``retained_chunks`` (output length), and
    ``dedup_rate`` ((exact + near) / total).
    """

    total = len(chunks)
    after_exact, exact_removed = exact_dedup(chunks)
    after_near, near_removed = lsh_dedup(after_exact, threshold=threshold)
    retained = len(after_near)
    dedup_rate = ((exact_removed + near_removed) / total) if total else 0.0
    result = DedupResult(
        total_chunks=total,
        exact_duplicates=exact_removed,
        near_duplicates=near_removed,
        retained_chunks=retained,
        dedup_rate=dedup_rate,
    )
    return after_near, result


def write_dedup_report(result: DedupResult, output_path: Path) -> None:
    """Serialise a :class:`DedupResult` to JSON at ``output_path``.

    The destination directory is created on demand so callers don't
    have to wire ``mkdir`` boilerplate alongside every call.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(result)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DedupResult",
    "dedup_pipeline",
    "exact_dedup",
    "lsh_dedup",
    "minhash_signature",
    "write_dedup_report",
]
