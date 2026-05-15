"""codingjepa.eval.schema.validate — validates results.json against the JSONSchema."""

from __future__ import annotations

import json
import pathlib

import pytest

from codingjepa.errors import SchemaVersionMismatch, UsageError
from codingjepa.eval.schema import SUPPORTED_MAJOR, validate

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "schemas" / "results.example.json"
)


def test_valid_results_passes(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "results.json"
    target.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    payload = validate(target)
    assert payload["schema_version"] == "v1"


def test_supported_major_is_one() -> None:
    assert SUPPORTED_MAJOR == 1


def test_wrong_major_raises_version_mismatch(tmp_path: pathlib.Path) -> None:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record["schema_version"] = "v2"
    target = tmp_path / "results.json"
    target.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SchemaVersionMismatch) as excinfo:
        validate(target)
    assert excinfo.value.context["schema_version"] == "v2"


def test_unparseable_version_raises(tmp_path: pathlib.Path) -> None:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record["schema_version"] = "not-a-version"
    target = tmp_path / "results.json"
    target.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SchemaVersionMismatch):
        validate(target)


def test_unknown_property_fails(tmp_path: pathlib.Path) -> None:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record["stowaway"] = "unexpected"
    target = tmp_path / "results.json"
    target.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(UsageError):
        validate(target)


def test_required_field_missing_fails(tmp_path: pathlib.Path) -> None:
    """Removing a required top-level field surfaces UsageError.

    Adding a new required field bumps the schema major (spec/03 §Schema versioning).
    """

    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record.pop("metadata")
    target = tmp_path / "results.json"
    target.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(UsageError):
        validate(target)


def test_bad_metric_shape_fails(tmp_path: pathlib.Path) -> None:
    """A reported metric missing mean/std/n_seeds/ci95_* is rejected (RFC-0010 §D8)."""

    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record["benchmarks"]["CJ-RET-100"]["retrieval_at_10"].pop("ci95_high")
    target = tmp_path / "results.json"
    target.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(UsageError):
        validate(target)
