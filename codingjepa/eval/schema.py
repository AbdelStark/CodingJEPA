"""Eval results.json validator. Schema lives at data/schemas/results.schema.json.

The schema was added in #138; this module provides the pathlib-based public
entry point named in spec/02 and the schema-major-version gate from spec/03
§Schema versioning policy.
"""

from __future__ import annotations

import json
import pathlib

from codingjepa._jsonschema import load_schema, validate_record
from codingjepa.errors import SchemaVersionMismatch

SUPPORTED_MAJOR = 1


def validate(path: pathlib.Path | str) -> dict[str, object]:
    """Validate a `results.json` file. Returns the parsed payload on success.

    Raises `SchemaVersionMismatch` if `schema_version` is for a major version
    we do not support; raises `UsageError` (from the shared JSONSchema
    validator) on any other shape error.
    """

    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as fp:
        payload: dict[str, object] = json.load(fp)

    raw_version = payload.get("schema_version", "")
    version_str = str(raw_version)
    major = _parse_major(version_str)
    if major != SUPPORTED_MAJOR:
        raise SchemaVersionMismatch(
            f"results.json schema_version={version_str!r}; supported major is v{SUPPORTED_MAJOR}",
            path=str(path),
            schema_version=version_str,
            supported_major=SUPPORTED_MAJOR,
        )

    schema = load_schema("results")
    validate_record(payload, schema)
    return payload


def _parse_major(version: str) -> int:
    """`v1`, `v1.2`, `v12.0` → integer major. Returns -1 on unparseable input."""

    if not version.startswith("v"):
        return -1
    body = version[1:].split(".", 1)[0]
    try:
        return int(body)
    except ValueError:
        return -1


__all__ = ["SUPPORTED_MAJOR", "validate"]
