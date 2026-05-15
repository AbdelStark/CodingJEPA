"""Redaction rules per spec/05 §Redaction. Run on every record before serialization."""

from __future__ import annotations

from typing import Any

from codingjepa.safety.secret_patterns import (
    EMAIL_PATTERN,
    KNOWN_HASH_FIELDS,
    SECRET_PATTERNS,
)

MAX_STRING_BYTES = 4 * 1024  # 4 KB cap; longer strings become "<redacted: too_long>".
DEBUG_SOURCE_TRUNCATION = 256


def _redact_string(value: str, *, field_name: str | None = None) -> str:
    """Apply the four redaction rules to one string value."""

    if len(value.encode("utf-8")) > MAX_STRING_BYTES:
        return "<redacted: too_long>"

    out = value
    if field_name not in KNOWN_HASH_FIELDS:
        for label, pattern in SECRET_PATTERNS:
            out = pattern.sub(f"<redacted: {label}>", out)
    out = EMAIL_PATTERN.sub("<redacted: email>", out)
    return out


def _truncate_source(value: str) -> str:
    if len(value) <= DEBUG_SOURCE_TRUNCATION:
        return value
    return value[:DEBUG_SOURCE_TRUNCATION] + "…"


def redact_payload(
    payload: dict[str, Any],
    *,
    level: str,
) -> dict[str, Any]:
    """Redact a payload dict in place-of-copy. `level` is the record's log level.

    Raw-source keys (`source`, `chunk_source`, `before_source`, `after_source`) are
    dropped at info-level-and-above and truncated to 256 chars at debug. Cycles
    are tolerated and surfaced as `"<redacted: cycle>"` so a circular payload
    never crashes the logger.
    """

    redacted = _redact_obj(payload, level=level, field_name=None, seen=set())
    assert isinstance(redacted, dict)
    return redacted


_RAW_SOURCE_KEYS = frozenset({"source", "chunk_source", "before_source", "after_source", "code"})


def _redact_obj(obj: Any, *, level: str, field_name: str | None, seen: set[int]) -> Any:
    if isinstance(obj, (dict, list)):
        oid = id(obj)
        if oid in seen:
            return "<redacted: cycle>"
        seen = seen | {oid}
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _RAW_SOURCE_KEYS and isinstance(v, str):
                if level == "debug":
                    out[k] = _truncate_source(v)
                else:
                    out[k] = "<redacted: source>"
                continue
            out[k] = _redact_obj(v, level=level, field_name=k, seen=seen)
        return out
    if isinstance(obj, list):
        return [_redact_obj(v, level=level, field_name=field_name, seen=seen) for v in obj]
    if isinstance(obj, str):
        return _redact_string(obj, field_name=field_name)
    return obj


__all__ = ["DEBUG_SOURCE_TRUNCATION", "MAX_STRING_BYTES", "redact_payload"]
