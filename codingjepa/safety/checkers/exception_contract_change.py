"""Checker: detect changes to the set of raised exceptions (RFC-0007 §D1, issue #93)."""

from __future__ import annotations

import libcst as cst

from codingjepa.safety.checkers._result import CheckerResult

__all__ = ["check"]

_REFUSAL_CODE = "R006_SAFETY_CHECKER_REJECTED_ALL"


def _get_exception_name(node: cst.BaseExpression) -> str | None:
    """Extract a canonical name from a raise-target expression."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return node.attr.value
    if isinstance(node, cst.Call):
        return _get_exception_name(node.func)
    return None


class _ExceptionCollector(cst.CSTVisitor):
    """Collect exception class names from all ``raise`` statements."""

    def __init__(self) -> None:
        self.exceptions: set[str] = set()

    def visit_Raise(self, node: cst.Raise) -> None:  # noqa: N802
        if node.exc is not None:
            name = _get_exception_name(node.exc)
            if name is not None:
                self.exceptions.add(name)


def _collect(source: str) -> set[str] | None:
    """Parse *source* and collect raised exception names.

    Returns None on parse error (caller treats as safe).
    """
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return None
    collector = _ExceptionCollector()
    tree.visit(collector)
    return collector.exceptions


def check(before_src: str, after_src: str) -> CheckerResult:
    """Return passed=False when the set of raised exceptions changes.

    Parameters
    ----------
    before_src:
        Normalised Python source before the refactor.
    after_src:
        Normalised Python source after the refactor.
    """
    before_excs = _collect(before_src)
    after_excs = _collect(after_src)

    # Parse failure → safe.
    if before_excs is None or after_excs is None:
        return CheckerResult(passed=True)

    added = after_excs - before_excs
    removed = before_excs - after_excs
    if added or removed:
        return CheckerResult(
            passed=False,
            reason=f"exception contract changed: added={sorted(added)}, removed={sorted(removed)}",
            refusal_code=_REFUSAL_CODE,
        )
    return CheckerResult(passed=True)
