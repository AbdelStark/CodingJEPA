"""Public inference surface (RFC-0009, spec/02 §codingjepa.inference)."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Provenance:
    repo: str
    commit_sha: str
    file_path: str
    node_qualname: str


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    source: str
    cosine: float
    confidence: float
    accepted_by_intent: bool
    rejected_reason: str | None
    provenance: Provenance


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_path: pathlib.Path
    index_path: pathlib.Path
    intent: str
    k: int = 10
    threshold: float = 0.55
    deterministic: bool = True


@dataclass(frozen=True)
class InferenceResult:
    candidates: list[Candidate]
    refusal: str | None


# Refusal codes (RFC-0007 §D3)
_R001 = "R001"  # parse failure
_R002 = "R002"  # normalization failure
_R003 = "R003"  # token cap exceeded
_R004 = "R004"  # no candidates retrieved
_R005 = "R005"  # confidence below threshold
_R006 = "R006"  # all candidates safety-rejected


def infer(
    source: str,
    cfg: InferenceConfig,
    *,
    model: Any | None = None,
    index: Any | None = None,
    index_meta: Any | None = None,
    tokenizer: Any | None = None,
) -> InferenceResult:
    """Run embed → retrieve → rerank, return InferenceResult.

    Parameters
    ----------
    source:
        Raw Python source chunk to find refactoring candidates for.
    cfg:
        Inference configuration (checkpoint, index paths, intent, k,
        threshold, deterministic flag).
    model:
        Optional pre-loaded :class:`~codingjepa.model.CodingJEPA`. When
        None, must be loaded from ``cfg.checkpoint_path`` by the caller
        before calling this function.
    index:
        Optional pre-loaded ``faiss.Index``.
    index_meta:
        Optional pre-loaded :class:`~codingjepa.inference.index.IndexMeta`.
    tokenizer:
        Optional pre-loaded :class:`~codingjepa.data.tokenizer.Tokenizer`.

    Returns
    -------
    InferenceResult
        ``refusal`` is non-None when no usable candidates were found.

    Raises
    ------
    NotImplementedError
        When any of ``model``, ``index``, ``index_meta``, or ``tokenizer``
        are None (caller must supply pre-loaded objects).
    """
    if model is None or index is None or index_meta is None or tokenizer is None:
        raise NotImplementedError(
            "infer() requires pre-loaded model, index, index_meta, and tokenizer. "
            "Load them from cfg.checkpoint_path / cfg.index_path before calling."
        )

    from codingjepa.inference.embed import embed_chunk
    from codingjepa.inference.rerank import rerank
    from codingjepa.inference.retrieve import retrieve
    from codingjepa.intents import INTENT_NONE, intent_index

    # Embed the source chunk
    emb = embed_chunk(source, model, tokenizer)
    if emb is None:
        return InferenceResult(candidates=[], refusal=_R001)

    # Map intent string to index
    intent_idx = intent_index(cfg.intent) if cfg.intent != INTENT_NONE else -1

    # Retrieve top-M candidates via predictor-expanded FAISS search
    retrieval = retrieve(
        source_emb=emb,
        intent_idx=intent_idx,
        model=model,
        index=index,
        index_meta=index_meta,
        top_m=max(cfg.k * 10, 100),
    )

    if not retrieval.indices:
        return InferenceResult(candidates=[], refusal=_R004)

    # Rerank with safety filter and confidence calibration
    candidates = rerank(
        retrieval=retrieval,
        intent=cfg.intent,
        source_before=source,
        k=cfg.k,
    )

    passing = [c for c in candidates if c.rejected_reason is None]

    if not passing:
        return InferenceResult(candidates=candidates, refusal=_R006)

    # Confidence threshold gate
    best_confidence = max((c.confidence for c in passing), default=0.0)
    if best_confidence < cfg.threshold:
        return InferenceResult(candidates=candidates, refusal=_R005)

    return InferenceResult(candidates=candidates, refusal=None)


__all__ = [
    "Candidate",
    "InferenceConfig",
    "InferenceResult",
    "Provenance",
    "infer",
]
