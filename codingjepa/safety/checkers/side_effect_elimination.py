"""Checker: detect removed side effects (RFC-0007 §D1, issue #92)."""

from __future__ import annotations

import libcst as cst

from codingjepa.safety.checkers._result import CheckerResult
from codingjepa.safety.checkers.side_effect_introduction import _SideEffectCollector

__all__ = ["check"]

_REFUSAL_CODE = "R006_SAFETY_CHECKER_REJECTED_ALL"


def _collect(source: str) -> set[str] | None:
    """Parse *source* and collect side-effect call names.

    Returns None when parsing fails (caller treats as safe).
    """
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return None
    collector = _SideEffectCollector()
    tree.visit(collector)
    return collector.effects


def check(before_src: str, after_src: str) -> CheckerResult:
    """Return passed=False when *after_src* removes side effects present in *before_src*.

    Parameters
    ----------
    before_src:
        Normalised Python source before the refactor.
    after_src:
        Normalised Python source after the refactor.
    """
    before_effects = _collect(before_src)
    after_effects = _collect(after_src)

    # Parse failure → safe (we cannot determine unsafety).
    if before_effects is None or after_effects is None:
        return CheckerResult(passed=True)

    removed_effects = before_effects - after_effects
    if removed_effects:
        names = ", ".join(sorted(removed_effects))
        return CheckerResult(
            passed=False,
            reason=f"after removes side effects: {names}",
            refusal_code=_REFUSAL_CODE,
        )
    return CheckerResult(passed=True)
