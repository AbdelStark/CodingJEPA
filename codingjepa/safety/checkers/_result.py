"""Shared result dataclass for all safety checkers (RFC-0007 §D1)."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["CheckerResult"]


@dataclass
class CheckerResult:
    """Result returned by every safety checker.

    Attributes
    ----------
    passed:
        True when the check is satisfied (candidate is safe to use).
    reason:
        Human-readable explanation; empty string when *passed* is True.
    refusal_code:
        Machine-readable refusal code (one of the R0NN_* codes); empty string
        when *passed* is True.
    """

    passed: bool
    reason: str = field(default="")
    refusal_code: str = field(default="")
