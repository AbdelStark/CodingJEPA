"""Spec/05 §Redaction: secrets, emails, oversized strings stripped from every payload."""

from __future__ import annotations

import io
import json

from codingjepa.observability.log import LogWriter
from codingjepa.observability.redact import (
    DEBUG_SOURCE_TRUNCATION,
    MAX_STRING_BYTES,
    redact_payload,
)


def _emit_and_decode(
    writer: LogWriter, event: str, level: str, payload: dict[str, object]
) -> dict[str, object]:
    """Emit a single record and decode the resulting JSONL line."""

    stream = writer._stream  # type: ignore[attr-defined]
    assert isinstance(stream, io.StringIO)
    stream.seek(0)
    stream.truncate()
    writer.emit(event, level=level, payload=payload)
    decoded: dict[str, object] = json.loads(stream.getvalue().strip())
    return decoded


def test_aws_key_redacted_in_payload() -> None:
    out = redact_payload({"trace": "creds AKIAIOSFODNN7EXAMPLE active"}, level="info")
    assert "AKIA" not in out["trace"]  # type: ignore[operator]
    assert "<redacted: aws_access_key_id>" in out["trace"]  # type: ignore[operator]


def test_github_pat_redacted() -> None:
    pat = "ghp_" + "a" * 36
    out = redact_payload({"msg": f"saw {pat} in env"}, level="info")
    assert pat not in out["msg"]  # type: ignore[operator]
    assert "<redacted: github_pat>" in out["msg"]  # type: ignore[operator]


def test_jwt_redacted() -> None:
    jwt = "eyJhbGciOi.eyJzdWIiOi.signature_part_here"
    out = redact_payload({"auth": f"token={jwt}"}, level="info")
    assert jwt not in out["auth"]  # type: ignore[operator]


def test_long_hex_redacted_unless_known_hash_field() -> None:
    long_hex = "deadbeef" * 8  # 64 chars
    out = redact_payload({"random_hex": long_hex}, level="info")
    assert long_hex not in out["random_hex"]  # type: ignore[operator]
    # Known hash field: preserved.
    out2 = redact_payload({"git_sha": long_hex}, level="info")
    assert out2["git_sha"] == long_hex


def test_email_redacted() -> None:
    out = redact_payload({"reporter": "alice@example.com filed it"}, level="info")
    assert "alice@example.com" not in out["reporter"]  # type: ignore[operator]


def test_oversized_string_replaced() -> None:
    big = "x" * (MAX_STRING_BYTES + 1)
    out = redact_payload({"blob": big}, level="info")
    assert out["blob"] == "<redacted: too_long>"


def test_raw_source_stripped_at_info_truncated_at_debug() -> None:
    payload = {"source": "def f():\n    return 1\n" * 100}
    info_out = redact_payload(payload, level="info")
    assert info_out["source"] == "<redacted: source>"
    debug_out = redact_payload(payload, level="debug")
    assert isinstance(debug_out["source"], str)
    assert len(debug_out["source"]) <= DEBUG_SOURCE_TRUNCATION + 1  # +1 for the ellipsis


def test_log_writer_redacts_before_emit() -> None:
    """End-to-end: the LogWriter pipes payloads through redact_payload."""

    writer = LogWriter(stream=io.StringIO())
    record = _emit_and_decode(
        writer,
        event="inference.success",
        level="info",
        payload={"top_1_chunk_id": "c-1", "leaked_token": "ghp_" + "z" * 36},
    )
    payload = record["payload"]
    assert isinstance(payload, dict)
    assert "ghp_" not in str(payload["leaked_token"])


def test_redaction_recurses_into_nested_structures() -> None:
    out = redact_payload(
        {"meta": {"keys": ["AKIA" + "A" * 16, "ok"]}, "reporter": "bob@x.io"},
        level="info",
    )
    keys = out["meta"]["keys"]  # type: ignore[index]
    assert keys[0] == "<redacted: aws_access_key_id>"
    assert keys[1] == "ok"
    assert "@" not in out["reporter"]  # type: ignore[operator]
