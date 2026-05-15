"""Labeler for `extract-helper` (RFC-0002 §D6 / issue #41).

Detects a refactor where a block of code is hoisted into a new top-level
helper function, and that helper is called at the original site.

Strong heuristic:
- A new top-level FunctionDef appears in `after` (not in `before`).
- The helper body has ≥ 2 statements (a single-statement "helper" is too
  trivial to flag as an extract-helper refactor).
- A call to the new helper appears in `after` as a free function (not a
  method on an unrelated object).

Confidence:
- 0.9 when the helper body's statements (in textual form) match a subsequence
  of one of `before`'s function bodies → strong "moved code" signal.
- 0.7 when only the new-function-plus-call structure is detected.
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


def _function_body_statements(fn: cst.FunctionDef) -> list[cst.BaseStatement]:
    if isinstance(fn.body, cst.IndentedBlock):
        return list(fn.body.body)
    return []


def _stmt_signatures(stmts: list[cst.BaseStatement]) -> list[str]:
    """A coarse textual fingerprint for each statement — used to detect
    statement-level overlap between the helper and any before-function body.

    The fingerprint is just the libcst code() output with leading/trailing
    whitespace normalized. Imperfect but robust enough for the conservative
    "moved ≥ 2 statements" signal.
    """

    out: list[str] = []
    empty = cst.Module(body=[])
    for s in stmts:
        try:
            out.append(empty.code_for_node(s).strip())
        except Exception:
            out.append(repr(s))
    return out


def _has_subsequence_match(helper_sigs: list[str], host_sigs: list[str]) -> bool:
    """True iff at least two consecutive items of `helper_sigs` appear (in
    order) as a contiguous subsequence of `host_sigs`."""

    if len(helper_sigs) < 2 or len(host_sigs) < 2:
        return False
    n = len(helper_sigs)
    m = len(host_sigs)
    for start in range(m - 1):
        # Look for at least 2-element contiguous match.
        match_len = 0
        for k in range(min(n, m - start)):
            if helper_sigs[k] == host_sigs[start + k]:
                match_len += 1
            else:
                break
        if match_len >= 2:
            return True
    return False


def extract_helper_labeler(before: str, after: str) -> tuple[bool, float]:
    """Return (matched, confidence) for the extract-helper intent."""

    before_mod = _safe_parse(before)
    after_mod = _safe_parse(after)
    if before_mod is None or after_mod is None:
        return False, 0.0

    before_fns = _top_level_functions(before_mod)
    after_fns = _top_level_functions(after_mod)

    new_fn_names = set(after_fns) - set(before_fns)
    if not new_fn_names:
        return False, 0.0

    best_confidence = 0.0
    for helper_name in new_fn_names:
        helper_fn = after_fns[helper_name]
        helper_body = _function_body_statements(helper_fn)
        if len(helper_body) < 2:
            # Single-statement helpers are too trivial — conservative reject.
            continue
        if not _has_call_to(after_mod, helper_name):
            continue

        helper_sigs = _stmt_signatures(helper_body)

        # Strong signal: helper-body statements appear as a contiguous slice of
        # one of before's function bodies → ≥ 2 statements were moved.
        moved = False
        for before_fn in before_fns.values():
            host_sigs = _stmt_signatures(_function_body_statements(before_fn))
            if _has_subsequence_match(helper_sigs, host_sigs):
                moved = True
                break

        confidence = 0.9 if moved else 0.7
        if confidence > best_confidence:
            best_confidence = confidence

    if best_confidence == 0.0:
        return False, 0.0
    return True, best_confidence
