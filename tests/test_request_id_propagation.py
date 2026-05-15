"""request_id flows from middleware → inference → log records. Spec/05 §Tracing."""

from __future__ import annotations

import io
import json
import re
import uuid

import pytest

from codingjepa.observability.log import LogWriter
from codingjepa.observability.request_id import (
    get_request_id,
    new_request_id,
    request_scope,
    uuid7,
)

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def test_uuid7_is_well_formed() -> None:
    """uuid7() returns a version-7 UUID with the variant bits in [89ab]."""

    for _ in range(64):
        u = uuid7()
        assert isinstance(u, uuid.UUID)
        assert UUID_RE.match(str(u)), f"uuid not v7: {u}"


def test_uuid7_is_time_ordered_across_ms_boundaries() -> None:
    """uuid7 carries a millisecond timestamp; samples spaced >1ms are lex-ordered.

    Within a single millisecond the lower bits are random, so we sample with a
    small sleep instead of consecutively. RFC 9562 §6.2 only requires monotonic
    counter ordering as an optional method; the spec/05 contract just needs the
    UUIDs to sort roughly by time.
    """

    import time

    first = uuid7()
    time.sleep(0.005)
    second = uuid7()
    time.sleep(0.005)
    third = uuid7()
    assert str(first) < str(second) < str(third), (first, second, third)


def test_request_id_default_is_none() -> None:
    """Outside a request_scope, get_request_id() is None."""

    assert get_request_id() is None


def test_request_scope_binds_and_restores() -> None:
    assert get_request_id() is None
    with request_scope() as rid:
        assert get_request_id() == rid
    assert get_request_id() is None


def test_request_scope_accepts_external_id() -> None:
    """When middleware passes an inbound X-Request-ID, it is used verbatim."""

    incoming = "01880000-0000-7000-8000-000000000000"
    with request_scope(incoming):
        assert get_request_id() == incoming


def test_log_record_carries_request_id_from_scope() -> None:
    stream = io.StringIO()
    writer = LogWriter(stream=stream, git_sha="abc1234")
    rid = new_request_id()
    with request_scope(rid):
        writer.emit("inference.success", level="info", payload={"top_1_chunk_id": "c-1"})
    record = json.loads(stream.getvalue().strip())
    assert record["request_id"] == rid


def test_log_record_request_id_null_outside_scope() -> None:
    stream = io.StringIO()
    writer = LogWriter(stream=stream)
    writer.emit("inference.success", level="info", payload={})
    record = json.loads(stream.getvalue().strip())
    assert record["request_id"] is None


@pytest.mark.parametrize("inbound", [None, "01890000-0000-7000-8000-fffffffffff0"])
def test_simulated_middleware_pipes_to_log(inbound: str | None) -> None:
    """Simulate an HTTP middleware: bind request_id, run a fake handler, capture log."""

    stream = io.StringIO()
    writer = LogWriter(stream=stream)

    def fake_handler() -> None:
        # An inference call inside this scope writes to the log.
        writer.emit(
            "inference.retrieve",
            level="info",
            payload={"intent": "extract-helper", "top_m": 100, "latency_ms": 12},
        )

    with request_scope(inbound) as rid:
        fake_handler()

    record = json.loads(stream.getvalue().strip())
    assert record["request_id"] == rid
    if inbound is not None:
        assert record["request_id"] == inbound
