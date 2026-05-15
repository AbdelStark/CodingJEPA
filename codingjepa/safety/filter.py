"""Safety filter: run all checkers and short-circuit on first failure (RFC-0007 §D1, issue #96)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from codingjepa.safety.checkers import (
    CheckerResult,
    check_async_sync,
    check_exception_contract,
    check_public_api,
    check_side_effect_elim,
    check_side_effect_intro,
)

__all__ = ["SafetyResult", "run"]

_CHECKERS: list[Callable[[str, str], CheckerResult]] = [
    check_side_effect_intro,
    check_side_effect_elim,
    check_exception_contract,
    check_public_api,
    check_async_sync,
]


@dataclass
class SafetyResult:
    """Aggregate result from running all safety checkers.

    Attributes
    ----------
    passed:
        True when every checker passed.
    reason:
        Human-readable explanation from the first failing checker; empty when
        *passed* is True.
    refusal_code:
        Machine-readable refusal code from the first failing checker; empty
        when *passed* is True.
    """

    passed: bool
    reason: str = field(default="")
    refusal_code: str = field(default="")


def run(before_src: str, after_src: str) -> SafetyResult:
    """Run all safety checkers in order, short-circuiting on the first failure.

    Parameters
    ----------
    before_src:
        Normalised Python source before the refactor.
    after_src:
        Normalised Python source after the refactor.

    Returns
    -------
    SafetyResult
        ``passed=True`` when all checkers pass; otherwise the result of the
        first failing checker.
    """
    for checker in _CHECKERS:
        result = checker(before_src, after_src)
        if not result.passed:
            return SafetyResult(
                passed=False,
                reason=result.reason,
                refusal_code=result.refusal_code,
            )
    return SafetyResult(passed=True)
