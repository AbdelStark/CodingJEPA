"""UUIDv7 generator and contextvars-based request_id propagation. Spec/05 §Tracing."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_REQUEST_ID: ContextVar[str | None] = ContextVar("codingjepa_request_id", default=None)
_RUN_ID: ContextVar[str | None] = ContextVar("codingjepa_run_id", default=None)


def uuid7() -> uuid.UUID:
    """RFC 9562 UUIDv7. Inlined because `uuid.uuid7` is only in stdlib from 3.13.

    Layout: 48-bit unix_ts_ms | 4-bit version=7 | 12-bit rand_a | 2-bit variant=10
    | 62-bit rand_b.
    """

    unix_ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = int.from_bytes(os.urandom(2), "big") & 0xFFF
    rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)
    value = (unix_ts_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0x2 << 62) | rand_b
    return uuid.UUID(int=value)


def new_request_id() -> str:
    return str(uuid7())


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


def set_request_id(value: str) -> None:
    _REQUEST_ID.set(value)


def get_run_id() -> str | None:
    return _RUN_ID.get()


def set_run_id(value: str) -> None:
    _RUN_ID.set(value)


@contextmanager
def request_scope(request_id: str | None = None) -> Iterator[str]:
    """Bind a request_id for the duration of a `with` block. Used by middleware and CLI."""

    rid = request_id or new_request_id()
    token = _REQUEST_ID.set(rid)
    try:
        yield rid
    finally:
        _REQUEST_ID.reset(token)


@contextmanager
def run_scope(run_id: str | None = None) -> Iterator[str]:
    """Bind a run_id for the duration of a `with` block. Used by training/eval entry points."""

    rid = run_id or new_request_id()
    token = _RUN_ID.set(rid)
    try:
        yield rid
    finally:
        _RUN_ID.reset(token)


__all__ = [
    "get_request_id",
    "get_run_id",
    "new_request_id",
    "request_scope",
    "run_scope",
    "set_request_id",
    "set_run_id",
    "uuid7",
]
