"""Labeler for `none-typing-modernization` (RFC-0002 §D6 / issue #48).

Detects PEP 604 (`X | Y`) / PEP 585 (builtin generics) modernization:
    Optional[X]           → X | None
    Union[A, B]           → A | B
    typing.Optional[X]    → X | None
    typing.Union[A, B]    → A | B
    typing.List[X]        → list[X]   (Dict/Tuple/Set/FrozenSet)
    from typing import List → builtin list

Conservative: requires that some legacy marker present in `before` is gone in
`after`, that *some* modern marker appears in `after`, and that no new legacy
marker was introduced.
"""

from __future__ import annotations

import libcst as cst

LEGACY_GENERIC_MARKERS: tuple[str, ...] = (
    "Optional[",
    "Union[",
    "typing.Optional[",
    "typing.Union[",
    "typing.List[",
    "typing.Dict[",
    "typing.Set[",
    "typing.Tuple[",
    "typing.FrozenSet[",
    "typing.Type[",
    "typing.DefaultDict[",
)

MODERN_GENERIC_MARKERS: tuple[str, ...] = (
    "| None",
    "list[",
    "dict[",
    "set[",
    "tuple[",
    "frozenset[",
    "type[",
)


def _can_parse(source: str) -> bool:
    if not source.strip():
        return False
    try:
        cst.parse_module(source)
    except Exception:
        return False
    return True


def _count_markers(source: str, markers: tuple[str, ...]) -> int:
    return sum(source.count(m) for m in markers)


def _count_pipe_union(source: str) -> int:
    """Count `X | Y` style union usages in annotations.

    Heuristic: bare ' | ' tokens not in comments. Pipe also appears in bit-or
    expressions outside annotations, so this is approximate — we only use it
    as a tiebreaker.
    """

    return source.count(" | ")


def none_typing_modernization_labeler(before: str, after: str) -> tuple[bool, float]:
    """Return (matched, confidence) for the none-typing-modernization intent."""

    if not _can_parse(before) or not _can_parse(after):
        return False, 0.0

    legacy_before = _count_markers(before, LEGACY_GENERIC_MARKERS)
    legacy_after = _count_markers(after, LEGACY_GENERIC_MARKERS)
    modern_before = _count_markers(after, MODERN_GENERIC_MARKERS) - _count_markers(
        before, MODERN_GENERIC_MARKERS
    )
    modern_pipe_gain = _count_pipe_union(after) - _count_pipe_union(before)

    # Conservative: require legacy markers to strictly decrease.
    if legacy_after >= legacy_before:
        return False, 0.0

    # Require some modern marker to have appeared, OR new `|` unions.
    if modern_before <= 0 and modern_pipe_gain <= 0:
        return False, 0.0

    # If every legacy marker disappeared cleanly → high confidence.
    full_clearance = legacy_after == 0 and legacy_before > 0
    if full_clearance and (modern_before > 0 or modern_pipe_gain > 0):
        return True, 0.95

    # Partial migration (some legacy left, some modern gained) → lower confidence.
    return True, 0.8
