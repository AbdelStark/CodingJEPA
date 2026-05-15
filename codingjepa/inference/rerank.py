"""Acceptance + safety filter + cosine rerank + confidence (RFC-0009 §D5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # noqa: F401 — kept for type clarity in callers

from codingjepa.inference import Candidate, Provenance
from codingjepa.inference.confidence import calibrate
from codingjepa.safety.filter import run as safety_run

from .retrieve import RetrievalResult


@dataclass
class RerankConfig:
    """Configuration for the reranking step."""

    k: int = 10
    mmr: bool = False


def rerank(
    retrieval: RetrievalResult,
    intent: str,
    source_before: str,
    *,
    k: int = 10,
) -> list[Candidate]:
    """Rerank: safety filter + cosine score + confidence calibration.

    For each retrieved candidate (up to ``k``):
    1. Retrieve source from meta by FAISS index.
    2. Run the safety filter comparing ``source_before`` against the candidate.
    3. Assign calibrated confidence to passing candidates.

    Parameters
    ----------
    retrieval:
        Raw retrieval result from :func:`~codingjepa.inference.retrieve.retrieve`.
    intent:
        Intent string (e.g. ``"extract-helper"`` or ``"NONE"``).
    source_before:
        Original source code used as the "before" reference for safety checks.
    k:
        Maximum number of candidates to include in the result (hard cap on
        the number returned, applied before confidence calibration).

    Returns
    -------
    list[Candidate]
        Up to ``k`` candidates sorted by cosine (FAISS already returns them
        in descending order). Each passing candidate carries a calibrated
        confidence; rejected candidates have ``confidence=0.0``.
    """
    top_pairs = list(zip(retrieval.indices, retrieval.cosines, strict=True))[:k]

    # Gather info for each candidate
    entries: list[tuple[int, float, str, str | None]] = []
    for idx, cos in top_pairs:
        source = retrieval.meta.sources[idx]
        safety_result = safety_run(source_before, source)
        rejected_reason: str | None = None if safety_result.passed else safety_result.reason
        entries.append((idx, cos, source, rejected_reason))

    # Calibrate confidence only over passing cosines
    passing_cosines = [cos for _, cos, _, rej in entries if rej is None]
    confidences = calibrate(passing_cosines) if passing_cosines else []
    conf_iter = iter(confidences)

    candidates: list[Candidate] = []
    for idx, cos, source, rejected_reason in entries:
        conf = next(conf_iter, 0.0) if rejected_reason is None else 0.0
        candidates.append(
            Candidate(
                chunk_id=retrieval.meta.chunk_ids[idx],
                source=source,
                cosine=cos,
                confidence=conf,
                accepted_by_intent=True,
                rejected_reason=rejected_reason,
                provenance=Provenance(
                    repo="",
                    commit_sha="",
                    file_path="",
                    node_qualname="",
                ),
            )
        )

    return candidates


__all__ = ["RerankConfig", "rerank"]
