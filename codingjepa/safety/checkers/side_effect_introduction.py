"""Checker: detect newly introduced side effects (RFC-0007 §D1, issue #91)."""

from __future__ import annotations

import libcst as cst

from codingjepa.safety.checkers._result import CheckerResult

__all__ = ["check"]

# Call-target names (or prefixes) that we classify as side-effecting.
_SIDE_EFFECT_NAMES: frozenset[str] = frozenset(
    [
        "print",
        "logging",
        "logger",
        "log",
        "open",
        "write",
        "os",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
    ]
)

_REFUSAL_CODE = "R006_SAFETY_CHECKER_REJECTED_ALL"


def _call_name(node: cst.BaseExpression) -> str | None:
    """Return a canonical name for the call target, or None if unrecognised."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        # a.b.c  → leaf is "c", but we want the *root* for prefix matching.
        return _root_name(node)
    return None


def _root_name(node: cst.BaseExpression) -> str | None:
    """Walk Attribute chain and return the root Name value."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return _root_name(node.value)
    return None


def _is_side_effect(name: str) -> bool:
    return name in _SIDE_EFFECT_NAMES


class _SideEffectCollector(cst.CSTVisitor):
    """Collect names of side-effecting calls in a CST."""

    def __init__(self) -> None:
        self.effects: set[str] = set()

    def visit_Call(self, node: cst.Call) -> None:  # noqa: N802
        name = _call_name(node.func)
        if name is not None and _is_side_effect(name):
            self.effects.add(name)


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
    """Return passed=False when *after_src* introduces new side effects absent in *before_src*.

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

    new_effects = after_effects - before_effects
    if new_effects:
        names = ", ".join(sorted(new_effects))
        return CheckerResult(
            passed=False,
            reason=f"after adds new side effects: {names}",
            refusal_code=_REFUSAL_CODE,
        )
    return CheckerResult(passed=True)
