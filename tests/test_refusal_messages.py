"""Closure and shape checks for the refusal copy table. Spec/04 §Refusal taxonomy."""

from __future__ import annotations

import pathlib
import re

import pytest

from codingjepa.errors import UsageError
from codingjepa.safety.messages import (
    REFUSAL_CODES,
    REFUSAL_MESSAGES,
    get_refusal_message,
)

SPEC_PATH = pathlib.Path(__file__).resolve().parents[1] / "docs" / "spec" / "04-error-model.md"
SPEC_CODES_RE = re.compile(r"`(R\d{3}_[A-Z0-9_]+)`")

# Authoritative set per docs/spec/04-error-model.md §Refusal taxonomy.
SPEC_CODES = frozenset(
    {
        "R001_SOURCE_PARSE_FAILED",
        "R002_SOURCE_TOO_LONG",
        "R003_SOURCE_EMPTY",
        "R004_NO_CANDIDATE_PASSED_ACCEPTANCE",
        "R005_CONFIDENCE_BELOW_THRESHOLD",
        "R006_SAFETY_CHECKER_REJECTED_ALL",
        "R007_INTENT_UNSUPPORTED_ON_SOURCE",
    }
)


def test_table_matches_spec() -> None:
    """The runtime table is exactly the set declared in spec/04."""

    assert set(REFUSAL_CODES) == SPEC_CODES


def test_spec_doc_lists_every_code() -> None:
    """Every code in the runtime table appears verbatim in docs/spec/04-error-model.md."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    declared = set(SPEC_CODES_RE.findall(text))
    missing = set(REFUSAL_CODES) - declared
    assert not missing, f"codes missing from spec/04: {sorted(missing)}"


@pytest.mark.parametrize("code", sorted(REFUSAL_CODES))
def test_message_shape(code: str) -> None:
    """Every code maps to a non-empty single-paragraph user-facing string."""

    msg = REFUSAL_MESSAGES[code]
    assert isinstance(msg, str)
    assert msg.strip() == msg, "message must not start/end with whitespace"
    assert len(msg) >= 16, f"{code}: message looks too terse"
    assert "\n" not in msg, f"{code}: refusal messages are single-paragraph"


def test_code_pattern() -> None:
    """All codes follow the R0NN_<UPPER_SNAKE> shape from spec/04."""

    pattern = re.compile(r"^R\d{3}_[A-Z][A-Z0-9_]+$")
    for code in REFUSAL_CODES:
        assert pattern.match(code), f"malformed refusal code: {code!r}"


def test_get_message_round_trip() -> None:
    for code in REFUSAL_CODES:
        assert get_refusal_message(code) == REFUSAL_MESSAGES[code]


def test_unknown_code_raises_usage_error() -> None:
    with pytest.raises(UsageError) as excinfo:
        get_refusal_message("R999_NOT_REAL")
    assert "R999_NOT_REAL" in str(excinfo.value.context)


def test_codes_are_unique() -> None:
    assert len(REFUSAL_CODES) == len(set(REFUSAL_CODES))
