"""Public inference surface (RFC-0009, spec/02 §codingjepa.inference). Placeholder stubs."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass


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


def infer(source: str, cfg: InferenceConfig) -> InferenceResult:
    """End-to-end refactor inference. Placeholder; the pipeline lands across #82–#90."""
    raise NotImplementedError


__all__ = [
    "Candidate",
    "InferenceConfig",
    "InferenceResult",
    "Provenance",
    "infer",
]
