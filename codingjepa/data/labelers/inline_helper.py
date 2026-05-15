"""Labeler for `inline-helper` (RFC-0002 §D6 / issue #42).

Detects the inverse of extract-helper: a function present in `before` is gone
in `after`, and its body has been inlined at the call site.

Strong heuristic:
- A top-level FunctionDef in `before` is missing in `after`.
- The removed function is *no longer called* anywhere in `after` (else the
  inline is broken and the diff is something else).
- The body expression of the removed function (when trivial, e.g. a single
  `return expr`) appears textually in `after`.

Confidence:
- 0.85 when the body expression of the removed helper appears textually in
  `after` (clear inlining).
- 0.7 when only the removal-plus-no-calls structure is detected.
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
    return {stmt.name.value: stmt for stmt in module.body if isinstance(stmt, cst.FunctionDef)}


def _has_call_to(module: cst.Module, func_name: str) -> bool:
    found = {"hit": False}

    class _V(cst.CSTVisitor):
        def visit_Call(self, node: cst.Call) -> None:
            if isinstance(node.func, cst.Name) and node.func.value == func_name:
                found["hit"] = True

    module.visit(_V())
    return found["hit"]


def _return_expression_source(fn: cst.FunctionDef) -> str | None:
    """If `fn` is a single `return <expr>`, return the textual form of
    `<expr>`. Otherwise None.
    """

    if not isinstance(fn.body, cst.IndentedBlock):
        return None
    body = fn.body.body
    if len(body) != 1:
        return None
    stmt = body[0]
    if not isinstance(stmt, cst.SimpleStatementLine):
        return None
    if len(stmt.body) != 1:
        return None
    ret = stmt.body[0]
    if not isinstance(ret, cst.Return) or ret.value is None:
        return None
    try:
        return cst.Module(body=[]).code_for_node(ret.value).strip()
    except Exception:
        return None


def inline_helper_labeler(before: str, after: str) -> tuple[bool, float]:
    """Return (matched, confidence) for the inline-helper intent."""

    before_mod = _safe_parse(before)
    after_mod = _safe_parse(after)
    if before_mod is None or after_mod is None:
        return False, 0.0

    before_fns = _top_level_functions(before_mod)
    after_fns = _top_level_functions(after_mod)

    removed_fn_names = set(before_fns) - set(after_fns)
    if not removed_fn_names:
        return False, 0.0

    # All removed helpers must be fully gone — no surviving call site, else
    # this isn't a clean inline.
    for name in removed_fn_names:
        if _has_call_to(after_mod, name):
            return False, 0.0

    # Conservative: the inline labeler only fires when the file still has
    # *some* remaining top-level function — otherwise it's likely a deletion,
    # not an inline.
    if not after_fns:
        return False, 0.0

    after_src = after_mod.code

    # Strong signal: at least one removed function's body expression now
    # appears textually in `after`.
    inlined_textually = False
    for name in removed_fn_names:
        body_src = _return_expression_source(before_fns[name])
        if body_src is None:
            continue
        if body_src in after_src:
            inlined_textually = True
            break

    if inlined_textually:
        return True, 0.85
    return True, 0.7
