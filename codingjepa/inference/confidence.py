"""Softmax-at-τ calibrated confidence (RFC-0007 §D3, RFC-0009 §D5)."""

from __future__ import annotations

import math


def calibrate(top_k_cosines: list[float], tau: float = 0.1) -> list[float]:
    """Return softmax-calibrated confidence scores.

    Uses numerically stable softmax: subtract max before exp.
    Returns uniform distribution when all cosines are equal.
    Returns empty list when input is empty.

    Parameters
    ----------
    top_k_cosines:
        Cosine similarity scores, one per retrieved candidate.
    tau:
        Temperature for softmax sharpening. Default 0.1.
    """
    if not top_k_cosines:
        return []

    n = len(top_k_cosines)
    if n == 1:
        return [1.0]

    max_val = max(top_k_cosines)
    exps = [math.exp((x - max_val) / tau) for x in top_k_cosines]
    total = sum(exps)

    if total == 0.0:
        # All cosines equal after subtraction: uniform
        return [1.0 / n] * n

    return [e / total for e in exps]


__all__ = ["calibrate"]
