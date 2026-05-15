"""User-facing failure messages for the demo (RFC-0006 §D6)."""

from __future__ import annotations

DEMO_MESSAGES: dict[str, str] = {
    "source_too_long": ("Input too long. CodingJEPA v1 only supports chunks ≤ 512 BPE tokens."),
    "source_parse_failed": ("Source does not parse. Returning verbatim. (Refusal per RFC-0007.)"),
    "no_candidate_accepted": (
        "No acceptable candidate for this intent. "
        "The top-1 candidate failed the acceptance rule."
    ),
    "confidence_below_threshold": (
        "Confidence below threshold (τ=0.55). No recommendation returned."
    ),
    "all_candidates_rejected_by_safety": (
        "Every candidate was rejected by a safety checker. " "Try a different intent or snippet."
    ),
}


def get_demo_message(key: str) -> str:
    """Return the verbatim user-facing message for a demo failure key.

    Raises KeyError on unknown key.
    """
    return DEMO_MESSAGES[key]


__all__ = ["DEMO_MESSAGES", "get_demo_message"]
