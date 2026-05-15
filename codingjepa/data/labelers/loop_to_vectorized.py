"""Labeler for `loop-to-vectorized` (RFC-0002 §D6 / issue #46).

Detects a Python for-loop over a numpy/pandas object being replaced by a
vectorized operation.

Conservative:
- The number of for-loops in `after` must be strictly less than in `before`.
- The disappearing loop iterates over a *numpy/pandas-looking* iterable
  (heuristic on variable name or attribute access).
- `after` contains a vectorized API token (`.apply(`, `.map(`, `.where(`,
  `np.where(`, `np.vectorize(`, `np.sum(`, `np.mean(`, etc.).

Confidence:
- 0.8 for clear numpy/pandas pattern (variable name match or .values/.iterrows).
- 0.6 for uncertain (loop disappeared, vectorized call appeared, but no
  strong numpy/pandas signal).
"""

from __future__ import annotations

import libcst as cst

# Variable-name heuristic: variables likely to hold a numpy/pandas object.
NUMPY_PANDAS_NAME_HINTS: frozenset[str] = frozenset(
    {
        "arr",
        "array",
        "arrays",
        "df",
        "series",
        "s",
        "data",
        "x",
        "y",
        "X",
        "Y",
        "col",
        "column",
        "col_",
        "vec",
        "vector",
        "values",
    }
)

# String tokens that indicate a vectorized API call appeared in `after`.
VECTORIZED_TOKENS: tuple[str, ...] = (
    ".apply(",
    ".map(",
    ".where(",
    ".filter(",
    ".sum(",
    ".mean(",
    ".max(",
    ".min(",
    "np.where(",
    "np.vectorize(",
    "np.sum(",
    "np.mean(",
    "np.array(",
    ".values",
    ".str.",
    ".dt.",
)


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


def _for_loop_iterables(module: cst.Module) -> list[cst.BaseExpression]:
    """Return the iterable expression for each for-loop."""

    out: list[cst.BaseExpression] = []

    class _V(cst.CSTVisitor):
        def visit_For(self, node: cst.For) -> None:
            out.append(node.iter)

    module.visit(_V())
    return out


def _looks_like_numpy_pandas(iter_expr: cst.BaseExpression) -> bool:
    """Heuristic check whether `iter_expr` looks like a numpy/pandas object."""

    # `df['c']`, `frame.values`, `arr.iterrows()`.
    if isinstance(iter_expr, cst.Subscript):
        # `df['col']` — subscript access on a name is pandas-ish.
        return True
    if isinstance(iter_expr, cst.Attribute):
        attr = iter_expr.attr.value
        if attr in {"values", "iterrows", "itertuples", "items"}:
            return True
        if attr.endswith("_") or attr in NUMPY_PANDAS_NAME_HINTS:
            return True
    if isinstance(iter_expr, cst.Call):
        # `df.iterrows()`, `np.arange(...)`.
        func = iter_expr.func
        if isinstance(func, cst.Attribute) and func.attr.value in {
            "iterrows",
            "itertuples",
            "items",
            "values",
        }:
            return True
        if isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name):
            if func.value.value == "np":
                return True
    if isinstance(iter_expr, cst.Name):
        return iter_expr.value in NUMPY_PANDAS_NAME_HINTS
    return False


def _has_vectorized_token(source: str) -> bool:
    return any(token in source for token in VECTORIZED_TOKENS)


def loop_to_vectorized_labeler(before: str, after: str) -> tuple[bool, float]:
    """Return (matched, confidence) for the loop-to-vectorized intent."""

    before_mod = _safe_parse(before)
    after_mod = _safe_parse(after)
    if before_mod is None or after_mod is None:
        return False, 0.0

    before_loops = _count_for_loops(before_mod)
    after_loops = _count_for_loops(after_mod)

    # Strict requirement: a loop disappeared.
    if after_loops >= before_loops or before_loops == 0:
        return False, 0.0

    # `after` must contain a vectorized-API token.
    if not _has_vectorized_token(after):
        return False, 0.0

    # Strong signal: at least one disappearing loop iterated over something
    # numpy/pandas-looking.
    before_iters = _for_loop_iterables(before_mod)
    if any(_looks_like_numpy_pandas(it) for it in before_iters):
        return True, 0.8
    # Uncertain — loop disappeared, vectorized API appeared, but no numpy/pandas hint.
    return False, 0.0
