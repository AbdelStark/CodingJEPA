"""Single source of truth for the 8 acceptance rules from RFC-0004 §D2.

Consumed by labelers (#41–#48), the inference rerank filter (#85), eval
scoring (#110), and the safety property tests (#98). Importing this module
from outside `codingjepa.intents` is forbidden; consumers go through
`from codingjepa.intents import acceptance_check`. `tests/test_acceptance_singleton.py`
enforces this contract.

Each rule is a structural heuristic over libcst.Module pairs. The rules are
deliberately permissive: labelers are expected to be more conservative
(see RFC-0004 §D2 note "Labelers may be more conservative than acceptance").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import libcst as cst

from codingjepa.errors import UsageError


def acceptance_check(intent: str, before: cst.Module, after: cst.Module) -> bool:
    """Boolean acceptance gate for `intent` on a `(before, after)` CST pair.

    `INTENT_NONE` is always accepted (used as the pretraining sentinel and the
    refusal fallback). Unknown intents raise UsageError.
    """

    if intent == "NONE":
        return True
    rule = _RULES.get(intent)
    if rule is None:
        raise UsageError(f"unknown intent: {intent!r}", intent=intent)
    return rule(before, after)


# ---------- Helpers ----------------------------------------------------------


def _top_level_function_names(module: cst.Module) -> set[str]:
    names: set[str] = set()
    for stmt in module.body:
        if isinstance(stmt, cst.FunctionDef):
            names.add(stmt.name.value)
        elif isinstance(stmt, cst.SimpleStatementLine):
            continue
    return names


def _top_level_classes(module: cst.Module) -> dict[str, cst.ClassDef]:
    return {stmt.name.value: stmt for stmt in module.body if isinstance(stmt, cst.ClassDef)}


def _all_calls(module: cst.Module) -> list[cst.Call]:
    calls: list[cst.Call] = []

    class _V(cst.CSTVisitor):
        def visit_Call(self, node: cst.Call) -> None:
            calls.append(node)

    module.visit(_V())
    return calls


def _has_call_to(module: cst.Module, func_name: str) -> bool:
    """True iff `module` calls `func_name` as a free function (not a method).

    Method calls like `obj.func_name(...)` do NOT count; they refer to a
    different binding.
    """

    for call in _all_calls(module):
        if isinstance(call.func, cst.Name) and call.func.value == func_name:
            return True
    return False


def _all_for_loops(module: cst.Module) -> list[cst.For]:
    loops: list[cst.For] = []

    class _V(cst.CSTVisitor):
        def visit_For(self, node: cst.For) -> None:
            loops.append(node)

    module.visit(_V())
    return loops


def _all_comprehensions(module: cst.Module) -> int:
    count = 0

    class _V(cst.CSTVisitor):
        def visit_ListComp(self, node: cst.ListComp) -> None:
            nonlocal count
            count += 1

        def visit_SetComp(self, node: cst.SetComp) -> None:
            nonlocal count
            count += 1

        def visit_DictComp(self, node: cst.DictComp) -> None:
            nonlocal count
            count += 1

        def visit_GeneratorExp(self, node: cst.GeneratorExp) -> None:
            nonlocal count
            count += 1

    module.visit(_V())
    return count


def _module_source(module: cst.Module) -> str:
    return module.code


# ---------- Per-intent rules -------------------------------------------------


def _check_extract_helper(before: cst.Module, after: cst.Module) -> bool:
    """A new top-level function appears in `after` and is called from `after`."""

    new = _top_level_function_names(after) - _top_level_function_names(before)
    if not new:
        return False
    return any(_has_call_to(after, name) for name in new)


def _check_inline_helper(before: cst.Module, after: cst.Module) -> bool:
    """A top-level function present in `before` is gone in `after` and no longer called."""

    removed = _top_level_function_names(before) - _top_level_function_names(after)
    if not removed:
        return False
    return all(not _has_call_to(after, name) for name in removed)


def _check_comprehension_rewrite(before: cst.Module, after: cst.Module) -> bool:
    """Before has a for-loop+append/add/update; after has a comprehension and one fewer for-loop."""

    before_comps = _all_comprehensions(before)
    after_comps = _all_comprehensions(after)
    if after_comps <= before_comps:
        return False
    before_loops = len(_all_for_loops(before))
    after_loops = len(_all_for_loops(after))
    if after_loops >= before_loops:
        return False

    # Reject `break` / `continue` in the disappearing loop heuristically by
    # checking that the before-source does not contain those tokens in a loop
    # near the same line. Conservative: if the source mentions `break` or
    # `continue` anywhere outside a comprehension, refuse.
    before_src = _module_source(before)
    if " break\n" in before_src or " continue\n" in before_src:
        return False
    return True


def _check_dataclass_migration(before: cst.Module, after: cst.Module) -> bool:
    """Same-named class: before has __init__ self.x=x; after is @dataclass w/o __init__."""

    before_classes = _top_level_classes(before)
    after_classes = _top_level_classes(after)
    common = before_classes.keys() & after_classes.keys()
    for name in common:
        b = before_classes[name]
        a = after_classes[name]
        if not _has_init_with_self_assignments(b):
            continue
        if _has_init(a):
            continue
        if _has_dataclass_decorator(a):
            return True
    return False


def _has_dataclass_decorator(cls: cst.ClassDef) -> bool:
    for dec in cls.decorators:
        d = dec.decorator
        if isinstance(d, cst.Name) and d.value == "dataclass":
            return True
        if isinstance(d, cst.Attribute) and d.attr.value == "dataclass":
            return True
        if isinstance(d, cst.Call):
            fn = d.func
            if isinstance(fn, cst.Name) and fn.value == "dataclass":
                return True
            if isinstance(fn, cst.Attribute) and fn.attr.value == "dataclass":
                return True
    return False


def _has_init(cls: cst.ClassDef) -> bool:
    for stmt in cls.body.body:
        if isinstance(stmt, cst.FunctionDef) and stmt.name.value == "__init__":
            return True
    return False


def _has_init_with_self_assignments(cls: cst.ClassDef) -> bool:
    """True iff `__init__`'s body is a non-empty sequence of `self.x = x` plain
    assignments where the RHS is a `Name` matching the corresponding parameter.

    Rejects derived state (e.g., `self.x = x*2`), defaulting (`x=5`), and
    bodies with anything other than the canonical assignments.
    """

    for stmt in cls.body.body:
        if not (isinstance(stmt, cst.FunctionDef) and stmt.name.value == "__init__"):
            continue
        param_names = {
            p.name.value
            for params in (
                stmt.params.params,
                stmt.params.kwonly_params,
                stmt.params.posonly_params,
            )
            for p in params
        } - {"self"}
        any_self_assign = False
        for body_stmt in stmt.body.body:
            if not isinstance(body_stmt, cst.SimpleStatementLine):
                return False
            for small in body_stmt.body:
                if not isinstance(small, cst.Assign):
                    return False
                if len(small.targets) != 1:
                    return False
                target = small.targets[0].target
                if not (
                    isinstance(target, cst.Attribute)
                    and isinstance(target.value, cst.Name)
                    and target.value.value == "self"
                ):
                    return False
                attr_name = target.attr.value
                if not (isinstance(small.value, cst.Name) and small.value.value == attr_name):
                    return False
                if attr_name not in param_names:
                    return False
                any_self_assign = True
        return any_self_assign
    return False


def _check_exception_handling_cleanup(before: cst.Module, after: cst.Module) -> bool:
    """Before has bare-except or try/except/pass; after tightens it."""

    before_src = _module_source(before)
    after_src = _module_source(after)

    bare_before = "except:" in before_src
    bare_after = "except:" in after_src
    if bare_before and not bare_after:
        return True

    pass_before = "except" in before_src and "pass" in before_src and _has_try_except_pass(before)
    pass_after = _has_try_except_pass(after)
    if pass_before and not pass_after:
        return True

    return False


def _has_try_except_pass(module: cst.Module) -> bool:
    seen = {"hit": False}

    class _V(cst.CSTVisitor):
        def visit_Try(self, node: cst.Try) -> None:
            for handler in node.handlers:
                for body_stmt in handler.body.body:
                    if isinstance(body_stmt, cst.SimpleStatementLine):
                        for s in body_stmt.body:
                            if isinstance(s, cst.Pass):
                                seen["hit"] = True
                                return

    module.visit(_V())
    return seen["hit"]


def _check_loop_to_vectorized(before: cst.Module, after: cst.Module) -> bool:
    """A for-loop in `before` is gone in `after`; `after` has a vectorized call.

    Vectorized markers: .apply, .map, .where, np.where, np.vectorize. RFC-0004 §D2
    calls observational equivalence a "best-effort heuristic"; it is not verified
    at this layer.
    """

    if len(_all_for_loops(after)) >= len(_all_for_loops(before)):
        return False

    after_src = _module_source(after)
    vector_calls = (".apply(", ".map(", ".where(", "np.where(", "np.vectorize(")
    return any(token in after_src for token in vector_calls)


def _check_argument_defaulting(before: cst.Module, after: cst.Module) -> bool:
    """At least one parameter in a same-named function gained a default in `after`."""

    before_fns = {fn.name.value: fn for fn in before.body if isinstance(fn, cst.FunctionDef)}
    after_fns = {fn.name.value: fn for fn in after.body if isinstance(fn, cst.FunctionDef)}
    for name, after_fn in after_fns.items():
        if name not in before_fns:
            continue
        before_fn = before_fns[name]
        before_defaults = _param_defaults(before_fn)
        after_defaults = _param_defaults(after_fn)
        gained = after_defaults - before_defaults
        if gained:
            for param_name in gained:
                if _default_is_mutable_literal(after_fn, param_name):
                    return False
            return True
    return False


def _param_defaults(fn: cst.FunctionDef) -> set[str]:
    """Names of parameters in `fn` that carry a default value."""

    out: set[str] = set()
    for params in (fn.params.params, fn.params.kwonly_params, fn.params.posonly_params):
        for p in params:
            if p.default is not None:
                out.add(p.name.value)
    return out


def _default_is_mutable_literal(fn: cst.FunctionDef, name: str) -> bool:
    """def f(x=[]) and friends are rejected per RFC-0004 §D2 argument-defaulting."""

    for params in (fn.params.params, fn.params.kwonly_params, fn.params.posonly_params):
        for p in params:
            if p.name.value != name or p.default is None:
                continue
            d = p.default
            if isinstance(d, (cst.List, cst.Dict, cst.Set)):
                return True
    return False


def _check_none_typing_modernization(before: cst.Module, after: cst.Module) -> bool:
    """Source-level: `before` has Optional[/Union[/typing.List[; `after` uses PEP 604/585 forms."""

    before_src = _module_source(before)
    after_src = _module_source(after)

    legacy_markers = (
        "Optional[",
        "Union[",
        "typing.List[",
        "typing.Dict[",
        "typing.Set[",
        "typing.Tuple[",
    )
    legacy_in_before = any(m in before_src for m in legacy_markers)
    if not legacy_in_before:
        return False

    modern_markers = ("| None", " | ", "list[", "dict[", "set[", "tuple[")
    modern_in_after = any(m in after_src for m in modern_markers)
    if not modern_in_after:
        return False

    # No new legacy markers introduced.
    if any(m in after_src for m in legacy_markers):
        return False
    return True


# ---------- Dispatch ---------------------------------------------------------


_RULES: dict[str, Callable[[cst.Module, cst.Module], bool]] = {
    "extract-helper": _check_extract_helper,
    "inline-helper": _check_inline_helper,
    "comprehension-rewrite": _check_comprehension_rewrite,
    "dataclass-migration": _check_dataclass_migration,
    "exception-handling-cleanup": _check_exception_handling_cleanup,
    "loop-to-vectorized": _check_loop_to_vectorized,
    "argument-defaulting": _check_argument_defaulting,
    "none-typing-modernization": _check_none_typing_modernization,
}


def parse(source: str) -> cst.Module:
    """Test helper: parse a Python source string into a libcst.Module."""

    return cst.parse_module(source)


__all__ = ["acceptance_check", "parse"]


# Suppress unused-import noise from `Any` (kept for forward compat).
_ = Any
