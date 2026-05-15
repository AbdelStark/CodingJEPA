"""Intent vocabulary and the acceptance-rule single source of truth (RFC-0004)."""

from __future__ import annotations

from codingjepa.intents.acceptance import acceptance_check

INTENTS: tuple[str, ...] = (
    "extract-helper",
    "inline-helper",
    "comprehension-rewrite",
    "dataclass-migration",
    "exception-handling-cleanup",
    "loop-to-vectorized",
    "argument-defaulting",
    "none-typing-modernization",
)

INTENT_NONE: str = "NONE"


def intent_index(intent: str) -> int:
    if intent == INTENT_NONE:
        return -1
    return INTENTS.index(intent)


def intent_name(idx: int) -> str:
    if idx == -1:
        return INTENT_NONE
    return INTENTS[idx]


__all__ = [
    "INTENT_NONE",
    "INTENTS",
    "acceptance_check",
    "intent_index",
    "intent_name",
]
