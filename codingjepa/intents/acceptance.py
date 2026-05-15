"""Single source of truth for per-intent acceptance rules. Placeholder; #40 implements."""

from __future__ import annotations

from typing import Any


def acceptance_check(intent: str, before: Any, after: Any) -> bool:
    """Per-intent acceptance rule (RFC-0004 §D2). Placeholder; #40 implements.

    Consumed by labelers (#41–#48), the inference rerank filter (#85), the
    eval harness scoring (#110), and the safety property tests (#98).
    `before` and `after` are `libcst.Module` instances at the real surface;
    the placeholder accepts Any to avoid importing libcst at scaffold time.
    """
    raise NotImplementedError


__all__ = ["acceptance_check"]
