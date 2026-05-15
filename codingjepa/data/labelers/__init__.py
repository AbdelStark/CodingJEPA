"""Intent labelers for the CodingJEPA data pipeline.

Each labeler takes (before, after) source strings and returns
`(matched: bool, confidence: float in [0, 1])`. Per RFC-0002 §D6, labelers
are *conservative* — they prefer false negatives over false positives — and
are stricter than the permissive acceptance heuristics in
`codingjepa.intents.acceptance` (see RFC-0004 §D2).

This package exposes:

- `Labeler` protocol (the function signature).
- `LABELERS`: registry mapping intent name → labeler callable.
- `label_pair(before, after)`: run all 8 labelers and return
  `(intent_name, confidence)` for the highest-confidence match, or
  `("NONE", 0.0)` if nothing fires.
"""

from __future__ import annotations

from typing import Protocol

from codingjepa.data.labelers.argument_defaulting import argument_defaulting_labeler
from codingjepa.data.labelers.comprehension_rewrite import comprehension_rewrite_labeler
from codingjepa.data.labelers.dataclass_migration import dataclass_migration_labeler
from codingjepa.data.labelers.exception_handling_cleanup import (
    exception_handling_cleanup_labeler,
)
from codingjepa.data.labelers.extract_helper import extract_helper_labeler
from codingjepa.data.labelers.inline_helper import inline_helper_labeler
from codingjepa.data.labelers.loop_to_vectorized import loop_to_vectorized_labeler
from codingjepa.data.labelers.none_typing_modernization import (
    none_typing_modernization_labeler,
)


class Labeler(Protocol):
    """Callable signature for an intent labeler."""

    def __call__(self, before: str, after: str) -> tuple[bool, float]:
        """Return (matched, confidence ∈ [0,1])."""

        ...


LABELERS: dict[str, Labeler] = {
    "extract-helper": extract_helper_labeler,
    "inline-helper": inline_helper_labeler,
    "comprehension-rewrite": comprehension_rewrite_labeler,
    "dataclass-migration": dataclass_migration_labeler,
    "exception-handling-cleanup": exception_handling_cleanup_labeler,
    "loop-to-vectorized": loop_to_vectorized_labeler,
    "argument-defaulting": argument_defaulting_labeler,
    "none-typing-modernization": none_typing_modernization_labeler,
}


# Deterministic iteration order for tie-breaking in `label_pair`. Equal
# confidence resolves in this order (first wins). The order mirrors
# `codingjepa.intents.INTENTS`.
_DETERMINISTIC_ORDER: tuple[str, ...] = (
    "extract-helper",
    "inline-helper",
    "comprehension-rewrite",
    "dataclass-migration",
    "exception-handling-cleanup",
    "loop-to-vectorized",
    "argument-defaulting",
    "none-typing-modernization",
)


def label_pair(before: str, after: str) -> tuple[str, float]:
    """Run all 8 labelers and return (intent_name, confidence).

    Returns `("NONE", 0.0)` if no labeler fires. Ties on confidence are
    broken by `_DETERMINISTIC_ORDER` (first wins) so the function is
    deterministic.
    """

    best_intent = "NONE"
    best_conf = 0.0
    for intent in _DETERMINISTIC_ORDER:
        labeler = LABELERS[intent]
        matched, conf = labeler(before, after)
        if matched and conf > best_conf:
            best_intent = intent
            best_conf = conf
    return best_intent, best_conf


__all__ = ["LABELERS", "Labeler", "label_pair"]
