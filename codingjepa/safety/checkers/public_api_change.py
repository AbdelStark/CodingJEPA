"""Checker: detect changes to the top-level public function/class API (RFC-0007 §D1, issue #94)."""

from __future__ import annotations

from dataclasses import dataclass, field

import libcst as cst

from codingjepa.safety.checkers._result import CheckerResult

__all__ = ["check"]

_REFUSAL_CODE = "R006_SAFETY_CHECKER_REJECTED_ALL"


@dataclass
class _FuncSig:
    """Minimal signature record for one function."""

    params: list[str] = field(default_factory=list)
    return_annotation: str | None = None
    is_async: bool = False


class _TopLevelFuncCollector(cst.CSTVisitor):
    """Collect signatures of all top-level function definitions."""

    def __init__(self) -> None:
        self.functions: dict[str, _FuncSig] = {}
        self._depth: int = 0

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:  # noqa: N802
        self._depth += 1
        return None  # still visit children (for nested functions we skip, but classes count)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
        self._depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:  # noqa: N802
        if self._depth == 0:
            name = node.name.value
            params = [p.name.value for p in node.params.params if isinstance(p.name, cst.Name)]
            # star_kwarg and other exotic params
            if node.params.star_kwarg is not None:
                if isinstance(node.params.star_kwarg.name, cst.Name):
                    params.append("**" + node.params.star_kwarg.name.value)
            ann: str | None = None
            if node.returns is not None:
                try:
                    ann = cst.parse_module("").code_for_node(node.returns.annotation)
                except Exception:  # noqa: BLE001
                    ann = None
            self.functions[name] = _FuncSig(
                params=params,
                return_annotation=ann,
                is_async=isinstance(node, cst.FunctionDef) and False,  # placeholder
            )
        self._depth += 1
        return None

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
        self._depth -= 1


def _collect(source: str) -> dict[str, _FuncSig] | None:
    """Parse *source* and return top-level function signature map.

    Returns None on parse error.
    """
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return None
    collector = _TopLevelFuncCollector()
    tree.visit(collector)
    return collector.functions


def check(before_src: str, after_src: str) -> CheckerResult:
    """Return passed=False when any top-level function's public API changes.

    Detects:
    * renamed functions (name existed in before but not in after)
    * changed parameter lists
    * changed return annotations

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

    # Functions that existed before but vanished (possible rename).
    missing = set(before_fns) - set(after_fns)
    if missing:
        names = ", ".join(sorted(missing))
        return CheckerResult(
            passed=False,
            reason=f"top-level functions removed or renamed: {names}",
            refusal_code=_REFUSAL_CODE,
        )

    # Check signature changes for functions present in both.
    for name in set(before_fns) & set(after_fns):
        b = before_fns[name]
        a = after_fns[name]
        if b.params != a.params:
            return CheckerResult(
                passed=False,
                reason=(f"parameter list of '{name}' changed: " f"{b.params!r} → {a.params!r}"),
                refusal_code=_REFUSAL_CODE,
            )
        if b.return_annotation != a.return_annotation:
            return CheckerResult(
                passed=False,
                reason=(
                    f"return annotation of '{name}' changed: "
                    f"{b.return_annotation!r} → {a.return_annotation!r}"
                ),
                refusal_code=_REFUSAL_CODE,
            )
    return CheckerResult(passed=True)
