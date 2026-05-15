"""Labeler for `comprehension-rewrite` (RFC-0002 §D6 / issue #43).

Detects:
- for-loop with single `xs.append(expr)` → ListComp
- for-loop with single `s.add(expr)`     → SetComp
- for-loop with single `d[k] = v`         → DictComp

Conservative:
- The number of `for` loops must strictly decrease.
- The number of comprehensions must strictly increase.
- `break`/`continue` in any disappearing loop → reject (non-equivalent rewrite).
- An already-existing comprehension in `before` does not count as new.
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


def _count_for_loops(module: cst.Module) -> int:
    count = 0

    class _V(cst.CSTVisitor):
        def visit_For(self, node: cst.For) -> None:
            nonlocal count
            count += 1

    module.visit(_V())
    return count


def _count_comprehensions(module: cst.Module) -> int:
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


def _has_break_or_continue(module: cst.Module) -> bool:
    """True iff any `break` or `continue` exists in `module` (outside of any
    comprehension — those don't have break/continue anyway)."""

    found = {"hit": False}

    class _V(cst.CSTVisitor):
        def visit_Break(self, node: cst.Break) -> None:
            found["hit"] = True

        def visit_Continue(self, node: cst.Continue) -> None:
            found["hit"] = True

    module.visit(_V())
    return found["hit"]


def _loop_appears_appendish(module: cst.Module) -> bool:
    """True iff *any* for-loop in `module` has a single-statement body that is
    `xs.append(...)`, `s.add(...)`, or `d[k] = v` (or a wrapping `if` with one
    of those inside)."""

    found = {"hit": False}

    class _V(cst.CSTVisitor):
        def visit_For(self, node: cst.For) -> None:
            if _is_appendish_body(node.body):
                found["hit"] = True

    module.visit(_V())
    return found["hit"]


def _is_appendish_body(suite: cst.BaseSuite) -> bool:
    if not isinstance(suite, cst.IndentedBlock):
        return False
    body = suite.body
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, cst.SimpleStatementLine):
        for small in stmt.body:
            if _is_append_call(small):
                return True
            if _is_subscript_assign(small):
                return True
    elif isinstance(stmt, cst.If):
        # `for x in xs: if cond: r.append(x)` — list comp with filter.
        if stmt.orelse is None:
            return _is_appendish_body(stmt.body)
    return False


def _is_append_call(small: cst.BaseSmallStatement) -> bool:
    if not isinstance(small, cst.Expr):
        return False
    if not isinstance(small.value, cst.Call):
        return False
    func = small.value.func
    if not isinstance(func, cst.Attribute):
        return False
    return func.attr.value in {"append", "add", "update"}


def _is_subscript_assign(small: cst.BaseSmallStatement) -> bool:
    """`d[k] = v` style assignment."""

    if not isinstance(small, cst.Assign):
        return False
    if len(small.targets) != 1:
        return False
    target = small.targets[0].target
    return isinstance(target, cst.Subscript)


def comprehension_rewrite_labeler(before: str, after: str) -> tuple[bool, float]:
    """Return (matched, confidence) for the comprehension-rewrite intent."""

    before_mod = _safe_parse(before)
    after_mod = _safe_parse(after)
    if before_mod is None or after_mod is None:
        return False, 0.0

    before_loops = _count_for_loops(before_mod)
    after_loops = _count_for_loops(after_mod)
    before_comps = _count_comprehensions(before_mod)
    after_comps = _count_comprehensions(after_mod)

    # Strict requirements: a loop disappeared *and* a comprehension appeared.
    if after_loops >= before_loops:
        return False, 0.0
    if after_comps <= before_comps:
        return False, 0.0

    # Loop semantics must be safe to comprehend: no break/continue in before.
    if _has_break_or_continue(before_mod):
        return False, 0.0

    # The disappearing loop should have looked appendish.
    if not _loop_appears_appendish(before_mod):
        return False, 0.0

    # Strong match: exactly one loop removed and one comprehension gained.
    if (before_loops - after_loops) == 1 and (after_comps - before_comps) == 1:
        return True, 0.9
    return True, 0.7
