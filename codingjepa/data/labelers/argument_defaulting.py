"""Labeler for `argument-defaulting` (RFC-0002 §D6 / issue #47).

Detects when a function parameter gains a default value between `before` and
`after`. Conservative:
- Same function name in both modules.
- Parameter name and position match.
- Default is not a mutable literal (`[]`, `{}`, `set()` literal).
- Optional bonus: at least one call site updated (drops the explicit arg).

Returns (matched, confidence):
- 0.9 when at least one call site updated to omit the now-defaulted arg.
- 0.7 when only the signature changed.
"""

from __future__ import annotations

import libcst as cst


def _safe_parse(source: str) -> cst.Module | None:
    if not source.strip():
        return None
    try:
        return cst.parse_module(source)
    except Exception:
        return None


def _top_level_functions(module: cst.Module) -> dict[str, cst.FunctionDef]:
    """Collect *all* top-level function defs (including nested under classes)
    keyed by their dotted name."""

    out: dict[str, cst.FunctionDef] = {}
    for stmt in module.body:
        if isinstance(stmt, cst.FunctionDef):
            out[stmt.name.value] = stmt
        elif isinstance(stmt, cst.ClassDef):
            for body_stmt in stmt.body.body:
                if isinstance(body_stmt, cst.FunctionDef):
                    out[f"{stmt.name.value}.{body_stmt.name.value}"] = body_stmt
    return out


def _params_with_defaults(fn: cst.FunctionDef) -> dict[str, cst.BaseExpression]:
    """Mapping of param name → default expression (only params that *have* a default)."""

    out: dict[str, cst.BaseExpression] = {}
    for params in (fn.params.params, fn.params.kwonly_params, fn.params.posonly_params):
        for p in params:
            if p.default is not None:
                out[p.name.value] = p.default
    return out


def _all_param_names(fn: cst.FunctionDef) -> list[str]:
    """Positional order of all params (incl. posonly + regular)."""

    out: list[str] = []
    for p in fn.params.posonly_params:
        out.append(p.name.value)
    for p in fn.params.params:
        out.append(p.name.value)
    return out


def _is_mutable_default(default: cst.BaseExpression) -> bool:
    if isinstance(default, (cst.List, cst.Dict, cst.Set)):
        return True
    # set() / list() / dict() with no args are also mutable.
    if isinstance(default, cst.Call) and isinstance(default.func, cst.Name):
        if default.func.value in {"list", "dict", "set"} and not default.args:
            return True
    return False


def _call_arg_count_matrix(module: cst.Module, fn_name: str) -> list[int]:
    """Return the positional-arg count for every call to `fn_name` in `module`.

    Method calls (`obj.fn_name(...)`) are excluded — they refer to different
    bindings.
    """

    counts: list[int] = []

    class _V(cst.CSTVisitor):
        def visit_Call(self, node: cst.Call) -> None:
            if isinstance(node.func, cst.Name) and node.func.value == fn_name:
                positional = sum(1 for arg in node.args if arg.keyword is None)
                counts.append(positional)

    module.visit(_V())
    return counts


def argument_defaulting_labeler(before: str, after: str) -> tuple[bool, float]:
    """Return (matched, confidence) for the argument-defaulting intent."""

    before_mod = _safe_parse(before)
    after_mod = _safe_parse(after)
    if before_mod is None or after_mod is None:
        return False, 0.0

    before_fns = _top_level_functions(before_mod)
    after_fns = _top_level_functions(after_mod)

    gained_default_in_some_fn = False
    call_site_updated = False

    for name, after_fn in after_fns.items():
        if name not in before_fns:
            continue
        before_fn = before_fns[name]

        # Param positions must align (same arity, same names in same positions)
        # for a true "defaulting" change.
        before_names = _all_param_names(before_fn)
        after_names = _all_param_names(after_fn)
        if before_names != after_names:
            # Position changed or parameter added — that's a different intent
            # (likely a signature change), not a defaulting change.
            continue

        before_defaults = _params_with_defaults(before_fn)
        after_defaults = _params_with_defaults(after_fn)
        gained = set(after_defaults) - set(before_defaults)
        if not gained:
            continue

        # Conservative: reject mutable-literal defaults.
        if any(_is_mutable_default(after_defaults[g]) for g in gained):
            return False, 0.0

        gained_default_in_some_fn = True

        # Check call-site updates: an explicit call in `before` that now omits
        # the defaulted arg.
        before_call_counts = _call_arg_count_matrix(before_mod, name)
        after_call_counts = _call_arg_count_matrix(after_mod, name)
        if before_call_counts and after_call_counts:
            max_before = max(before_call_counts)
            min_after = min(after_call_counts)
            if max_before > min_after:
                call_site_updated = True

    if not gained_default_in_some_fn:
        return False, 0.0
    if call_site_updated:
        return True, 0.9
    return True, 0.7
