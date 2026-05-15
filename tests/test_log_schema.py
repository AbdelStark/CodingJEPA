"""data/schemas/log.schema.json validates every event in the closed taxonomy."""

from __future__ import annotations

import io
import json

import pytest

from codingjepa.errors import UsageError
from codingjepa.observability.log import (
    CLOSED_EVENT_TAXONOMY,
    LogWriter,
    load_log_schema,
    validate_log_record,
)


def _writer() -> tuple[LogWriter, io.StringIO]:
    stream = io.StringIO()
    return LogWriter(stream=stream, git_sha="abc1234", deterministic=True), stream


@pytest.mark.parametrize("event", sorted(CLOSED_EVENT_TAXONOMY - {"log.dropped"}))
def test_emitted_records_validate(event: str) -> None:
    """Each event in the taxonomy emits a record that passes the schema validator."""

    schema = load_log_schema()
    writer, stream = _writer()
    writer.emit(event, level="info", payload={"foo": "bar"})
    record = json.loads(stream.getvalue().strip())
    validate_log_record(record, schema)


def test_error_record_requires_error_code() -> None:
    writer, _ = _writer()
    with pytest.raises(UsageError):
        writer.emit("train.gate.failed", level="error", payload={})
    with pytest.raises(UsageError):
        writer.emit("train.gate.failed", level="info", payload={}, error_code="E_USAGE")


def test_unknown_event_rejected() -> None:
    writer, _ = _writer()
    with pytest.raises(UsageError) as excinfo:
        writer.emit("does.not.exist", level="info")
    assert "unknown event" in str(excinfo.value)


def test_unknown_level_rejected() -> None:
    writer, _ = _writer()
    with pytest.raises(UsageError):
        writer.emit("inference.success", level="critical")


def test_log_dropped_emitted_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """When serialization fails, the dropped counter rises and close() emits log.dropped."""

    writer, stream = _writer()

    real_dumps = json.dumps
    calls = {"n": 0}

    def flaky_dumps(*args: object, **kwargs: object) -> str:
        calls["n"] += 1
        # Fail the first call (the actual emit) but pass the log.dropped emit.
        if calls["n"] == 1:
            raise TypeError("synthetic serialization failure")
        return real_dumps(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("codingjepa.observability.log.json.dumps", flaky_dumps)
    writer.emit("inference.success", level="info", payload={"top_1_chunk_id": "c-1"})
    writer.close()

    lines = [json.loads(line) for line in stream.getvalue().strip().splitlines()]
    log_dropped = [r for r in lines if r["event"] == "log.dropped"]
    assert log_dropped, "log.dropped record missing"
    assert log_dropped[0]["payload"]["count"] == 1
    assert "synthetic serialization failure" in log_dropped[0]["payload"]["last_failure"]


def test_schema_root_shape() -> None:
    schema = load_log_schema()
    assert schema["type"] == "object"
    assert "ts" in schema["required"]
    assert "level" in schema["properties"]
    assert schema["properties"]["level"]["enum"] == ["debug", "info", "warn", "error"]


def test_record_with_unknown_top_level_field_fails() -> None:
    schema = load_log_schema()
    rec = {
        "ts": "2026-01-01T00:00:00.000+00:00",
        "level": "info",
        "event": "inference.success",
        "git_sha": "abc",
        "deterministic": False,
        "payload": {},
        "stowaway": "this key is not in the schema",
    }
    with pytest.raises(UsageError):
        validate_log_record(rec, schema)


def test_validator_catches_wrong_type() -> None:
    schema = load_log_schema()
    rec = {
        "ts": "2026-01-01T00:00:00.000+00:00",
        "level": "info",
        "event": "inference.success",
        "git_sha": 12345,  # should be string
        "deterministic": False,
        "payload": {},
    }
    with pytest.raises(UsageError):
        validate_log_record(rec, schema)
