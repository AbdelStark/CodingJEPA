"""Labeler for `exception-handling-cleanup` (RFC-0002 §D6 / issue #45).

Detects:
- bare `except:` → `except Exception:` (or narrower)
- `try/except/pass` → `try/except` that logs or re-raises
- broad `except Exception:` → narrower exception types
- `try/except/pass` → `contextlib.suppress(...)`

Conservative:
- Require strict improvement: bare-handler count decreases OR pass-only
  handler count decreases. The opposite (a regression) returns (False, 0.0).
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


def _count_bare_except(module: cst.Module) -> int:
    """Count `except:` handlers (no type)."""

    count = 0

    class _V(cst.CSTVisitor):
        def visit_ExceptHandler(self, node: cst.ExceptHandler) -> None:
            nonlocal count
            if node.type is None:
                count += 1

    module.visit(_V())
    return count


def _count_pass_only_handlers(module: cst.Module) -> int:
    """Count handlers whose body is exactly `pass`."""

    count = 0

    class _V(cst.CSTVisitor):
        def visit_ExceptHandler(self, node: cst.ExceptHandler) -> None:
            nonlocal count
            body = node.body.body
            if (
                len(body) == 1
                and isinstance(body[0], cst.SimpleStatementLine)
                and len(body[0].body) == 1
                and isinstance(body[0].body[0], cst.Pass)
            ):
                count += 1

    module.visit(_V())
    return count


def _count_broad_exception_handlers(module: cst.Module) -> int:
    """Count handlers explicitly catching `Exception` or `BaseException`."""

    count = 0

    class _V(cst.CSTVisitor):
        def visit_ExceptHandler(self, node: cst.ExceptHandler) -> None:
            nonlocal count
            t = node.type
            if isinstance(t, cst.Name) and t.value in {"Exception", "BaseException"}:
                count += 1

    module.visit(_V())
    return count


def _has_contextlib_suppress(module: cst.Module) -> bool:
    """Detect `with contextlib.suppress(...)` or `with suppress(...)`."""

    found = {"hit": False}

    class _V(cst.CSTVisitor):
        def visit_With(self, node: cst.With) -> None:
            for item in node.items:
                expr = item.item
                if isinstance(expr, cst.Call):
                    func = expr.func
                    if isinstance(func, cst.Attribute) and func.attr.value == "suppress":
                        found["hit"] = True
                    elif isinstance(func, cst.Name) and func.value == "suppress":
                        found["hit"] = True

    module.visit(_V())
    return found["hit"]


def exception_handling_cleanup_labeler(before: str, after: str) -> tuple[bool, float]:
    """Return (matched, confidence) for the exception-handling-cleanup intent."""

    before_mod = _safe_parse(before)
    after_mod = _safe_parse(after)
    if before_mod is None or after_mod is None:
        return False, 0.0

    bare_before = _count_bare_except(before_mod)
    bare_after = _count_bare_except(after_mod)

    pass_before = _count_pass_only_handlers(before_mod)
    pass_after = _count_pass_only_handlers(after_mod)

    broad_before = _count_broad_exception_handlers(before_mod)
    broad_after = _count_broad_exception_handlers(after_mod)

    suppress_after = _has_contextlib_suppress(after_mod)
    suppress_before = _has_contextlib_suppress(before_mod)

    # Regression checks: bare/pass handlers increasing → reject.
    if bare_after > bare_before:
        return False, 0.0
    if pass_after > pass_before and not (suppress_after and not suppress_before):
        # New pass-only handler appeared with no compensating suppress → reject.
        return False, 0.0

    # bare-except → typed except (direct fix).
    if bare_before > bare_after and bare_before > 0:
        return True, 0.9

    # try/except/pass → suppress(...) (idiomatic improvement).
    if pass_before > 0 and pass_after < pass_before and suppress_after and not suppress_before:
        return True, 0.9

    # try/except/pass → try/except with logging or raise.
    if pass_before > 0 and pass_after < pass_before:
        return True, 0.7

    # broad → narrower (e.g. Exception → KeyError).
    if broad_before > broad_after and broad_before > 0:
        return True, 0.7

    return False, 0.0
