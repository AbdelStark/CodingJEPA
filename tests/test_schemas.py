"""Validate every committed JSONSchema fixture against its schema. Spec/03."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from codingjepa._jsonschema import SCHEMA_NAMES, load_schema, validate_record
from codingjepa.errors import UsageError

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "schemas"
SCHEMAS_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "schemas"


def _load_fixture(name: str) -> Any:
    with (FIXTURES / f"{name}.example.json").open(encoding="utf-8") as fp:
        return json.load(fp)


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_is_well_formed(name: str) -> None:
    """Each schema file parses as JSON and has the required draft-2020-12 metadata."""

    schema = load_schema(name)
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    assert schema.get("$id", "").startswith("https://codingjepa.dev/schemas/")
    assert schema.get("title"), f"{name}.schema.json: missing title"
    assert schema.get("description"), f"{name}.schema.json: missing description"


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_fixture_validates(name: str) -> None:
    """Every committed example fixture passes its schema."""

    schema = load_schema(name)
    fixture = _load_fixture(name)
    validate_record(fixture, schema)


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_unknown_property_rejected(name: str) -> None:
    """Adding a stowaway top-level property to a fixture must fail validation.

    This proves the schemas are strict about additionalProperties at the root.
    """

    schema = load_schema(name)
    fixture = _load_fixture(name)
    fixture_with_stowaway = {**fixture, "stowaway": "unexpected"}
    with pytest.raises(UsageError) as excinfo:
        validate_record(fixture_with_stowaway, schema)
    assert "unknown property" in str(excinfo.value) or "stowaway" in str(excinfo.value)


def test_validator_handles_pattern() -> None:
    """The validator enforces pattern (sha256-shaped hashes)."""

    schema = load_schema("manifest")
    fixture = _load_fixture("manifest")
    bad = {**fixture, "manifest_hash": "not-a-sha256"}
    with pytest.raises(UsageError):
        validate_record(bad, schema)


def test_validator_handles_array_items() -> None:
    """Array items are recursed into."""

    schema = load_schema("pool")
    fixture = _load_fixture("pool")
    bad = {**fixture, "chunk_ids": ["ok", 42]}  # 42 is not a string
    with pytest.raises(UsageError):
        validate_record(bad, schema)


def test_validator_handles_enum() -> None:
    schema = load_schema("splits")
    fixture = _load_fixture("splits")
    bad = {**fixture, "by_repo": {"repo": "not-a-split"}}
    with pytest.raises(UsageError):
        validate_record(bad, schema)


def test_validator_handles_minimum_maximum() -> None:
    schema = load_schema("audit")
    fixture = _load_fixture("audit")
    bad = {**fixture, "compile_ok_rate": 1.5}
    with pytest.raises(UsageError):
        validate_record(bad, schema)


def test_all_schemas_committed() -> None:
    """The set of `.schema.json` files in `data/schemas/` equals SCHEMA_NAMES."""

    found = {p.name.removesuffix(".schema.json") for p in SCHEMAS_DIR.glob("*.schema.json")}
    assert found == set(
        SCHEMA_NAMES
    ), f"schema set drifted: extra={found - set(SCHEMA_NAMES)}, missing={set(SCHEMA_NAMES) - found}"
