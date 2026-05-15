"""Minimal JSONSchema-2020-12 subset validator used by `data/schemas/`.

We do not depend on `jsonschema` at runtime (RFC-0013 §D3 dependency-creep rule).
The subset implemented covers exactly the keywords used by the schemas in
`data/schemas/`:

    type, enum, required, properties, additionalProperties, items, pattern,
    minimum, maximum, minLength, maxLength

Validation raises `codingjepa.errors.UsageError` on the first failure, with a
JSON-pointer-style `path` in the context.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from codingjepa.errors import UsageError

_JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def validate_record(record: Any, schema: dict[str, Any]) -> None:
    """Validate `record` against `schema`. Raises UsageError on failure."""

    _validate(record, schema, path="$")


def _validate(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if "type" in schema:
        _check_type(value, schema, path=path)

    if "enum" in schema and value not in schema["enum"]:
        raise UsageError(
            f"{path}: value not in enum",
            path=path,
            value=value,
            enum=schema["enum"],
        )

    if isinstance(value, str):
        _check_string(value, schema, path=path)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _check_number(value, schema, path=path)

    if isinstance(value, dict):
        _check_object(value, schema, path=path)

    if isinstance(value, list):
        _check_array(value, schema, path=path)


def _check_type(value: Any, schema: dict[str, Any], *, path: str) -> None:
    declared = schema["type"]
    types = declared if isinstance(declared, list) else [declared]
    accepted: tuple[type, ...] = tuple(t for name in types for t in _JSON_TYPE_MAP.get(name, ()))
    if not accepted:
        return

    if value is True or value is False:
        if not any(t is bool for t in accepted):
            raise UsageError(
                f"{path}: bool not allowed", path=path, expected=types, actual="boolean"
            )
        return

    if not isinstance(value, accepted):
        raise UsageError(
            f"{path}: wrong type",
            path=path,
            expected=types,
            actual=type(value).__name__,
        )


def _check_string(value: str, schema: dict[str, Any], *, path: str) -> None:
    if "pattern" in schema and not re.search(schema["pattern"], value):
        raise UsageError(
            f"{path}: pattern mismatch",
            path=path,
            pattern=schema["pattern"],
            value=value,
        )
    if "minLength" in schema and len(value) < schema["minLength"]:
        raise UsageError(f"{path}: shorter than minLength", path=path, value=value)
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        raise UsageError(f"{path}: longer than maxLength", path=path)


def _check_number(value: int | float, schema: dict[str, Any], *, path: str) -> None:
    if "minimum" in schema and value < schema["minimum"]:
        raise UsageError(f"{path}: below minimum", path=path, value=value)
    if "maximum" in schema and value > schema["maximum"]:
        raise UsageError(f"{path}: above maximum", path=path, value=value)


def _check_object(value: dict[str, Any], schema: dict[str, Any], *, path: str) -> None:
    required = set(schema.get("required", ()))
    missing = required - value.keys()
    if missing:
        raise UsageError(f"{path}: missing required fields", path=path, missing=sorted(missing))

    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)
    for key, val in value.items():
        prop = properties.get(key)
        if prop is None:
            if additional is False:
                raise UsageError(f"{path}.{key}: unknown property", path=f"{path}.{key}", field=key)
            if isinstance(additional, dict):
                _validate(val, additional, path=f"{path}.{key}")
            continue
        _validate(val, prop, path=f"{path}.{key}")


def _check_array(value: list[Any], schema: dict[str, Any], *, path: str) -> None:
    items_schema = schema.get("items")
    if isinstance(items_schema, dict):
        for i, v in enumerate(value):
            _validate(v, items_schema, path=f"{path}[{i}]")


@lru_cache(maxsize=32)
def load_schema(name: str) -> dict[str, Any]:
    """Load `data/schemas/<name>.schema.json` cached. `name` is the bare stem."""

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "data", "schemas", f"{name}.schema.json")
    with open(path, encoding="utf-8") as fp:
        result: dict[str, Any] = json.load(fp)
        return result


SCHEMA_NAMES: tuple[str, ...] = (
    "manifest",
    "splits",
    "audit",
    "dedup",
    "cross_split_leakage",
    "log",
    "results",
    "pool",
    "gold",
    "model_card",
)


__all__ = ["SCHEMA_NAMES", "load_schema", "validate_record"]
