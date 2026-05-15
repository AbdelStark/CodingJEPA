"""Stable refusal copy table per RFC-0007 §D7 and docs/spec/04-error-model.md §Refusal taxonomy.

Each refusal carries:

* a stable machine-readable ``code`` of the form ``R0NN_*``;
* a human-readable ``message`` displayed verbatim in the CLI and demo UI.

The set is **closed**: adding or removing a code is an RFC amendment.
`tests/test_refusal_messages.py` enforces the closure against spec/04.
"""

from __future__ import annotations

REFUSAL_MESSAGES: dict[str, str] = {
    "R001_SOURCE_PARSE_FAILED": (
        "We can't refactor this snippet because it does not parse as valid Python 3.12. "
        "Fix the syntax error and try again."
    ),
    "R002_SOURCE_TOO_LONG": (
        "We can't refactor this snippet because it exceeds the 512-token chunk size we trained on. "
        "Try a smaller selection."
    ),
    "R003_SOURCE_EMPTY": (
        "We can't refactor this snippet because it is empty after normalization. "
        "Make sure the selection contains executable Python."
    ),
    "R004_NO_CANDIDATE_PASSED_ACCEPTANCE": (
        "We could not find a candidate that satisfies the selected intent's acceptance rule. "
        "Try a different intent or a different snippet."
    ),
    "R005_CONFIDENCE_BELOW_THRESHOLD": (
        "The top candidate's similarity was below the configured refusal threshold (τ=0.55). "
        "We are returning no recommendation rather than a low-confidence one."
    ),
    "R006_SAFETY_CHECKER_REJECTED_ALL": (
        "Every candidate was rejected by a safety checker (side-effect change, exception-contract "
        "change, public-API change, or async/sync boundary change)."
    ),
    "R007_INTENT_UNSUPPORTED_ON_SOURCE": (
        "The selected intent's acceptance rule cannot apply to this snippet "
        "(for example, comprehension-rewrite on a loop containing break or continue)."
    ),
}

REFUSAL_CODES: tuple[str, ...] = tuple(REFUSAL_MESSAGES.keys())


def get_refusal_message(code: str) -> str:
    """Look up the verbatim user-facing message for a refusal code.

    Raises codingjepa.errors.UsageError on an unknown code so the closed-set
    contract is machine-enforceable at every call site.
    """

    from codingjepa.errors import UsageError

    try:
        return REFUSAL_MESSAGES[code]
    except KeyError:
        raise UsageError(
            f"unknown refusal code: {code!r}", code=code, taxonomy_size=len(REFUSAL_MESSAGES)
        ) from None


__all__ = ["REFUSAL_CODES", "REFUSAL_MESSAGES", "get_refusal_message"]
