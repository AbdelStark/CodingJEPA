"""JSONLines structured log writer with the closed event taxonomy from spec/05."""

from __future__ import annotations

import json
import os
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


# ---------- Minimal JSON-Schema validator (subset) ---------------------------


def validate_log_record(record: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate `record` against a JSONSchema subset (object/required/properties/type/enum).

    Raises UsageError on the first failure. The subset is the one used by
    `data/schemas/log.schema.json`; we deliberately do not depend on
    jsonschema-the-library at runtime (RFC-0013 §D3 keeps the dep stack lean).
    """

    if schema.get("type") != "object":
        raise UsageError("log schema root must declare type=object", actual=schema.get("type"))
    required = set(schema.get("required", ()))
    missing = required - record.keys()
    if missing:
        raise UsageError("missing required log fields", missing=sorted(missing))

    properties = schema.get("properties", {})
    for key, value in record.items():
        prop = properties.get(key)
        if prop is None:
            if not schema.get("additionalProperties", True):
                raise UsageError(f"unknown log field: {key!r}", field=key)
            continue
        _check_type(key, value, prop)


_JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def _check_type(key: str, value: Any, prop: dict[str, Any]) -> None:
    declared = prop.get("type")
    if declared is None:
        return
    types = declared if isinstance(declared, list) else [declared]
    accepted: tuple[type, ...] = tuple(t for name in types for t in _JSON_TYPE_MAP.get(name, ()))
    if not accepted:
        return
    # booleans are ints; reject the surprising overlap.
    if value is True or value is False:
        if not any(t is bool for t in accepted):
            raise UsageError(f"field {key!r}: bool not allowed", expected=types)
    elif not isinstance(value, accepted) or isinstance(value, bool):
        raise UsageError(
            f"field {key!r}: wrong type",
            expected=types,
            actual=type(value).__name__,
        )

    if "enum" in prop and value not in prop["enum"]:
        raise UsageError(
            f"field {key!r}: value not in enum",
            value=value,
            enum=prop["enum"],
        )


def load_log_schema() -> dict[str, Any]:
    """Load `data/schemas/log.schema.json`. Resolved relative to the package root."""

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(repo_root, "data", "schemas", "log.schema.json")
    with open(path, encoding="utf-8") as fp:
        result: dict[str, Any] = json.load(fp)
        return result


__all__ = [
    "CLOSED_EVENT_TAXONOMY",
    "LEVELS",
    "LogWriter",
    "load_log_schema",
    "validate_log_record",
]
