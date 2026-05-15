"""Checker: detect async/sync boundary changes for top-level functions (RFC-0007 §D1, issue #95)."""

from __future__ import annotations

import libcst as cst

from codingjepa.safety.checkers._result import CheckerResult

__all__ = ["check"]

_REFUSAL_CODE = "R006_SAFETY_CHECKER_REJECTED_ALL"


class _AsyncCollector(cst.CSTVisitor):
    """Collect ``{name: is_async}`` for all top-level function definitions."""

    def __init__(self) -> None:
        self.functions: dict[str, bool] = {}
        self._depth: int = 0

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:  # noqa: N802
        self._depth += 1
        return None

    def leave_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
        self._depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:  # noqa: N802
        if self._depth == 0:
            self.functions[node.name.value] = node.asynchronous is not None
        self._depth += 1
        return None

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
        self._depth -= 1


def _collect(source: str) -> dict[str, bool] | None:
    """Parse *source* and return ``{name: is_async}`` for top-level functions.

    Returns None on parse error.
    """
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return None
    collector = _AsyncCollector()
    tree.visit(collector)
    return collector.functions


def check(before_src: str, after_src: str) -> CheckerResult:
    """Return passed=False when any top-level function changes between sync and async.

    Parameters
    ----------
    before_src:
        Normalised Python source before the refactor.
    after_src:
        Normalised Python source after the refactor.
    """
    before_fns = _collect(before_src)
    after_fns = _collect(after_src)

    # Parse failure → safe.
    if before_fns is None or after_fns is None:
        return CheckerResult(passed=True)

    for name in set(before_fns) & set(after_fns):
        if before_fns[name] != after_fns[name]:
            direction = "sync → async" if after_fns[name] else "async → sync"
            return CheckerResult(
                passed=False,
                reason=f"async/sync boundary changed for '{name}' ({direction})",
                refusal_code=_REFUSAL_CODE,
            )
    return CheckerResult(passed=True)
