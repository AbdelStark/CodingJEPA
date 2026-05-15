"""Labeler for `dataclass-migration` (RFC-0002 §D6 / issue #44).

Detects a same-named class transitioning from a hand-written `__init__` that
just does `self.x = x` for n ≥ 2 params to a `@dataclass` class with the
same n annotated attributes.

Conservative:
- `before` class has __init__ whose body is *only* `self.x = x` assignments.
- `after` class has @dataclass (or @dataclasses.dataclass / @dataclass(...)).
- `after` class has no remaining __init__ (handed off to the dataclass).
- Field names match between before and after (full match → 0.9; partial → 0.7).
- Requires ≥ 2 fields per RFC-0002 §D6 (no false positives on single-field).
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


def _top_level_classes(module: cst.Module) -> dict[str, cst.ClassDef]:
    return {stmt.name.value: stmt for stmt in module.body if isinstance(stmt, cst.ClassDef)}


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


def _init_with_plain_self_assignments(cls: cst.ClassDef) -> list[str] | None:
    """If the class has an __init__ that is *only* `self.<name> = <name>` for
    every param (besides self), return the ordered list of field names.

    Returns None if __init__ is missing or contains anything else (derived
    state, extra statements, etc.).
    """

    init_fn: cst.FunctionDef | None = None
    for stmt in cls.body.body:
        if isinstance(stmt, cst.FunctionDef) and stmt.name.value == "__init__":
            init_fn = stmt
            break
    if init_fn is None:
        return None

    param_names_in_order: list[str] = []
    for p in init_fn.params.posonly_params:
        if p.name.value != "self":
            param_names_in_order.append(p.name.value)
    for p in init_fn.params.params:
        if p.name.value != "self":
            param_names_in_order.append(p.name.value)
    param_set = set(param_names_in_order)
    if not param_set:
        return None

    found_assigns: list[str] = []
    for body_stmt in init_fn.body.body:
        if not isinstance(body_stmt, cst.SimpleStatementLine):
            return None
        for small in body_stmt.body:
            if not isinstance(small, cst.Assign):
                return None
            if len(small.targets) != 1:
                return None
            target = small.targets[0].target
            if not (
                isinstance(target, cst.Attribute)
                and isinstance(target.value, cst.Name)
                and target.value.value == "self"
            ):
                return None
            attr_name = target.attr.value
            if not (isinstance(small.value, cst.Name) and small.value.value == attr_name):
                return None
            if attr_name not in param_set:
                return None
            found_assigns.append(attr_name)

    if not found_assigns:
        return None
    return found_assigns


def _annotated_fields(cls: cst.ClassDef) -> list[str]:
    """Return the ordered list of annotated attribute names directly under cls."""

    out: list[str] = []
    for stmt in cls.body.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                if isinstance(small, cst.AnnAssign) and isinstance(small.target, cst.Name):
                    out.append(small.target.value)
    return out


def dataclass_migration_labeler(before: str, after: str) -> tuple[bool, float]:
    """Return (matched, confidence) for the dataclass-migration intent."""

    before_mod = _safe_parse(before)
    after_mod = _safe_parse(after)
    if before_mod is None or after_mod is None:
        return False, 0.0

    before_classes = _top_level_classes(before_mod)
    after_classes = _top_level_classes(after_mod)
    common = before_classes.keys() & after_classes.keys()
    if not common:
        return False, 0.0

    best_confidence = 0.0
    for name in common:
        b = before_classes[name]
        a = after_classes[name]
        before_fields = _init_with_plain_self_assignments(b)
        if before_fields is None:
            continue
        if len(before_fields) < 2:
            # RFC-0002 §D6: conservative — require ≥ 2 fields.
            continue
        if _has_init(a):
            # Migration should have removed the hand-written __init__.
            continue
        if not _has_dataclass_decorator(a):
            continue
        after_fields = _annotated_fields(a)
        if not after_fields:
            continue
        overlap = set(before_fields) & set(after_fields)
        if not overlap:
            continue
        if set(before_fields) == set(after_fields):
            best_confidence = max(best_confidence, 0.9)
        else:
            best_confidence = max(best_confidence, 0.7)

    if best_confidence == 0.0:
        return False, 0.0
    return True, best_confidence
