"""JSONLines structured log writer with the closed event taxonomy from spec/05."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import IO, Any

from codingjepa.errors import UsageError
from codingjepa.observability.redact import redact_payload
from codingjepa.observability.request_id import get_request_id, get_run_id

# ---------- Closed event taxonomy --------------------------------------------

CLOSED_EVENT_TAXONOMY: frozenset[str] = frozenset(
    {
        "data.mirror.repo",
        "data.chunk.file",
        "data.dedup.cluster",
        "data.audit.repo",
        "train.step",
        "train.eval_probe",
        "train.rank_diagnostic",
        "train.checkpoint.write",
        "train.gate.failed",
        "inference.embed",
        "inference.retrieve",
        "inference.rerank",
        "inference.refusal",
        "inference.success",
        "eval.benchmark.start",
        "eval.benchmark.done",
        "eval.harness.done",
        "eval.sandbox.run",
        "safety.checker.fired",
        "log.dropped",
    }
)

LEVELS: frozenset[str] = frozenset({"debug", "info", "warn", "error"})

# ---------- LogWriter --------------------------------------------------------


class LogWriter:
    """JSONLines log writer. Thread-unsafe by design (callers hold per-process state).

    Records are emitted to `stream` (default stdout) one line each. Records that
    fail JSON serialization are dropped and counted; a single `log.dropped`
    record is emitted at warn level on `close()`.
    """

    def __init__(
        self,
        stream: IO[str] | None = None,
        *,
        git_sha: str = "unknown",
        deterministic: bool = False,
    ) -> None:
        self._stream: IO[str] = stream if stream is not None else sys.stdout
        self._git_sha = git_sha
        self._deterministic = deterministic
        self._dropped = 0
        self._last_failure: str | None = None

    def emit(
        self,
        event: str,
        *,
        level: str = "info",
        payload: dict[str, Any] | None = None,
        manifest_hash: str | None = None,
        checkpoint_hash: str | None = None,
        index_id: str | None = None,
        seed: int | None = None,
        error_code: str | None = None,
    ) -> None:
        if event not in CLOSED_EVENT_TAXONOMY:
            raise UsageError(
                f"unknown event name: {event!r}",
                event=event,
                taxonomy_size=len(CLOSED_EVENT_TAXONOMY),
            )
        if level not in LEVELS:
            raise UsageError(f"unknown log level: {level!r}", level=level)
        if (level == "error") != (error_code is not None):
            raise UsageError(
                "error_code must be set iff level=='error'",
                level=level,
                error_code=error_code,
            )

        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": level,
            "event": event,
            "request_id": get_request_id(),
            "run_id": get_run_id(),
            "git_sha": self._git_sha,
            "deterministic": self._deterministic,
            "payload": redact_payload(payload or {}, level=level),
        }
        if manifest_hash is not None:
            record["manifest_hash"] = manifest_hash
        if checkpoint_hash is not None:
            record["checkpoint_hash"] = checkpoint_hash
        if index_id is not None:
            record["index_id"] = index_id
        if seed is not None:
            record["seed"] = seed
        if error_code is not None:
            record["error_code"] = error_code

        try:
            line = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            self._dropped += 1
            self._last_failure = repr(exc)
            return

        self._stream.write(line + "\n")
        self._stream.flush()

    def close(self) -> None:
        if self._dropped:
            self._stream.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                        "level": "warn",
                        "event": "log.dropped",
                        "request_id": get_request_id(),
                        "run_id": get_run_id(),
                        "git_sha": self._git_sha,
                        "deterministic": self._deterministic,
                        "payload": {"count": self._dropped, "last_failure": self._last_failure},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            self._stream.flush()
            self._dropped = 0
            self._last_failure = None


# ---------- JSON-Schema validation -------------------------------------------


def validate_log_record(record: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate a log record against the log JSONSchema. See codingjepa._jsonschema."""

    from codingjepa._jsonschema import validate_record

    validate_record(record, schema)


def load_log_schema() -> dict[str, Any]:
    """Load `data/schemas/log.schema.json`. Cached via codingjepa._jsonschema.load_schema."""

    from codingjepa._jsonschema import load_schema

    return load_schema("log")


__all__ = [
    "CLOSED_EVENT_TAXONOMY",
    "LEVELS",
    "LogWriter",
    "load_log_schema",
    "validate_log_record",
]
