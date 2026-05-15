"""Secret/PII scanner for the data-pipeline (RFC-0014 §D5).

Every chunk's ``source_normalized`` is run through the closed set of regexes
in :data:`codingjepa.safety.secret_patterns.SECRET_PATTERNS`, plus a
conservative email and IPv4 pattern that the pipeline adds on top. Any hit
removes the entire chunk from the corpus and surfaces a :class:`SecretHit`
record. The audit ``secret_scanner_hits == 0`` gate (RFC-0002 §D2) is
computed off of the same hit list.

We import the base patterns from the safety package so the policy is single-
sourced — adding a pattern requires an RFC amendment to
``codingjepa/safety/secret_patterns.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from codingjepa.safety.secret_patterns import EMAIL_PATTERN
from codingjepa.safety.secret_patterns import SECRET_PATTERNS as _BASE_PATTERNS

__all__ = [
    "SECRET_PATTERNS",
    "SecretHit",
    "scan_chunk",
    "scan_chunks",
    "scan_report",
]


@dataclass
class SecretHit:
    """One regex hit on a single chunk.

    ``match_preview`` is a redacted slice of the matched substring (first 20
    characters, with non-alphanumeric runs collapsed). Storing only the
    preview keeps the audit log from leaking the actual secret back into
    artifacts.
    """

    chunk_id: str
    pattern_name: str
    match_preview: str


_PREVIEW_LEN = 20


# Conservative extras the data pipeline adds on top of the base safety set
# (RFC-0014 §D5). The base set in ``codingjepa.safety.secret_patterns`` is the
# authority for the redactor at inference time; this scanner additionally
# strips email/IPv4 to suppress training-time PII.
_PIPELINE_EXTRAS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email_address", EMAIL_PATTERN),
    (
        "ipv4_address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        ),
    ),
)


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = _BASE_PATTERNS + _PIPELINE_EXTRAS


# ---------------------------------------------------------------------------
# scan_chunk
# ---------------------------------------------------------------------------


def scan_chunk(chunk_source: str, chunk_id: str) -> list[SecretHit]:
    """Scan a single source string for secrets/PII. Returns one hit per regex match.

    The same chunk_id can appear multiple times if more than one pattern
    matches; :func:`scan_report` dedups by chunk for the "chunks dropped"
    figure.
    """

    if not chunk_source:
        return []

    hits: list[SecretHit] = []
    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(chunk_source):
            hits.append(
                SecretHit(
                    chunk_id=chunk_id,
                    pattern_name=name,
                    match_preview=_redact_preview(match.group(0)),
                )
            )
    return hits


# ---------------------------------------------------------------------------
# scan_chunks
# ---------------------------------------------------------------------------


def scan_chunks(
    chunks: list[Any],
    *,
    verbose: bool = False,
) -> tuple[list[Any], list[SecretHit]]:
    """Scan every chunk's ``source_normalized`` for secrets.

    Returns ``(clean_chunks, hits)``. Any chunk that produced at least one
    hit is removed from the returned list — the secrets gate is strict per
    RFC-0014 §D5 (no partial-mask fallback).
    """

    hits: list[SecretHit] = []
    clean: list[Any] = []
    for chunk in chunks:
        chunk_id = getattr(chunk, "chunk_id", None) or ""
        source = getattr(chunk, "source_normalized", "") or ""
        chunk_hits = scan_chunk(source, chunk_id=chunk_id)
        if chunk_hits:
            hits.extend(chunk_hits)
            if verbose:
                # Best-effort diagnostic — avoid print spam in tests by default.
                names = sorted({h.pattern_name for h in chunk_hits})
                print(f"[secrets_scan] drop {chunk_id}: {','.join(names)}")
            continue
        clean.append(chunk)
    return clean, hits


# ---------------------------------------------------------------------------
# scan_report
# ---------------------------------------------------------------------------


def scan_report(hits: list[SecretHit]) -> dict[str, Any]:
    """Aggregate hit list into a summary dict suitable for the audit JSON.

    Shape::

        {
            "total_hits": int,                       # total regex matches
            "by_pattern": {pattern_name: int, ...},  # matches per pattern
            "chunks_dropped": int                    # distinct chunk_ids in hits
        }
    """

    by_pattern: dict[str, int] = {}
    chunks_with_hits: set[str] = set()
    for h in hits:
        by_pattern[h.pattern_name] = by_pattern.get(h.pattern_name, 0) + 1
        chunks_with_hits.add(h.chunk_id)
    return {
        "total_hits": len(hits),
        "by_pattern": by_pattern,
        "chunks_dropped": len(chunks_with_hits),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redact_preview(matched: str) -> str:
    """Return a bounded, partially-masked preview of ``matched``.

    We keep the first 4 characters (enough to spot which pattern fired) and
    mask the rest with ``*`` up to :data:`_PREVIEW_LEN` characters. The match
    string is never stored verbatim in the audit log.
    """

    matched = matched.replace("\n", " ").replace("\r", " ")
    if len(matched) <= 4:
        return matched
    head = matched[:4]
    tail_len = min(len(matched), _PREVIEW_LEN) - len(head)
    return head + ("*" * max(tail_len, 0))
