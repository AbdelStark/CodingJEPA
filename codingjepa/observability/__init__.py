"""Observability package: structured logging, redaction, request_id propagation. Spec/05."""

from __future__ import annotations

from codingjepa.observability.log import (
    CLOSED_EVENT_TAXONOMY,
    LEVELS,
    LogWriter,
    load_log_schema,
    validate_log_record,
)
from codingjepa.observability.redact import redact_payload
from codingjepa.observability.request_id import (
    get_request_id,
    get_run_id,
    new_request_id,
    request_scope,
    run_scope,
    set_request_id,
    set_run_id,
    uuid7,
)

__all__ = [
    "CLOSED_EVENT_TAXONOMY",
    "LEVELS",
    "LogWriter",
    "get_request_id",
    "get_run_id",
    "load_log_schema",
    "new_request_id",
    "redact_payload",
    "request_scope",
    "run_scope",
    "set_request_id",
    "set_run_id",
    "uuid7",
    "validate_log_record",
]
