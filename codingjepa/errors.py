"""Closed exception taxonomy for codingjepa. See docs/spec/04-error-model.md.

Every error in the package is one of the classes below. Adding a new class requires
an RFC amendment; `tests/test_errors.py` enforces the closure.

Each class carries:

* ``code``: stable ``E_*`` identifier used in structured logs and HTTP bodies.
* ``context``: dict of identifiers that locate the failure (paths, hashes, ids).
"""

from __future__ import annotations

import re
from typing import ClassVar


class CodingJEPAError(Exception):
    """Root of the closed exception taxonomy."""

    code: ClassVar[str] = "E_CODINGJEPA"

    def __init__(self, message: str = "", **context: object) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, object] = dict(context)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(message={self.message!r}, **{self.context!r})"


# ---------- Operator / config -------------------------------------------------


class UsageError(CodingJEPAError):
    """Bad CLI flags, malformed HTTP body, unknown intent."""

    code: ClassVar[str] = "E_USAGE"


class ConfigError(CodingJEPAError):
    """Missing or unresolvable Hydra config."""

    code: ClassVar[str] = "E_CONFIG"


# ---------- Artifact / hash mismatch ------------------------------------------


class ArtifactError(CodingJEPAError):
    """Missing checkpoint, missing index, or generic artifact problem."""

    code: ClassVar[str] = "E_ARTIFACT"


class ManifestHashMismatch(ArtifactError):
    code: ClassVar[str] = "E_MANIFEST_HASH_MISMATCH"


class CheckpointHashMismatch(ArtifactError):
    code: ClassVar[str] = "E_CHECKPOINT_HASH_MISMATCH"


class IndexHashMismatch(ArtifactError):
    code: ClassVar[str] = "E_INDEX_HASH_MISMATCH"


# ---------- Data --------------------------------------------------------------


class DataError(CodingJEPAError):
    code: ClassVar[str] = "E_DATA"


class SchemaVersionMismatch(DataError):
    code: ClassVar[str] = "E_SCHEMA_VERSION_MISMATCH"


class ProvenanceMissing(DataError):
    code: ClassVar[str] = "E_PROVENANCE_MISSING"


class DedupContractViolation(DataError):
    code: ClassVar[str] = "E_DEDUP_CONTRACT_VIOLATION"


class SplitContractViolation(DataError):
    code: ClassVar[str] = "E_SPLIT_CONTRACT_VIOLATION"


# ---------- Model -------------------------------------------------------------


class ModelError(CodingJEPAError):
    code: ClassVar[str] = "E_MODEL"


class EmbeddingCollapse(ModelError):
    """Rank diagnostic gate (RFC-0008)."""

    code: ClassVar[str] = "E_EMBEDDING_COLLAPSE"


class LossDivergence(ModelError):
    """pred_loss not monotone within the first 5k steps."""

    code: ClassVar[str] = "E_LOSS_DIVERGENCE"


class ParamCountMismatch(ModelError):
    code: ClassVar[str] = "E_PARAM_COUNT_MISMATCH"


# ---------- Inference-time determinism ----------------------------------------


class DeterminismViolation(CodingJEPAError):
    """`torch.use_deterministic_algorithms(True)` tripped at inference."""

    code: ClassVar[str] = "E_DETERMINISM_VIOLATION"


# ---------- Sandbox (execution-preservation eval) -----------------------------


class SandboxError(CodingJEPAError):
    code: ClassVar[str] = "E_SANDBOX"


class SandboxTimeout(SandboxError):
    code: ClassVar[str] = "E_SANDBOX_TIMEOUT"


class SandboxMemoryExceeded(SandboxError):
    code: ClassVar[str] = "E_SANDBOX_MEMORY_EXCEEDED"


class SandboxNetworkAttempted(SandboxError):
    code: ClassVar[str] = "E_SANDBOX_NETWORK_ATTEMPTED"


# ---------- Internal ----------------------------------------------------------


class InternalError(CodingJEPAError):
    """Wraps an unexpected exception with context; bug, not operator-fixable."""

    code: ClassVar[str] = "E_INTERNAL"


# ---------- Closed taxonomy ---------------------------------------------------

CLOSED_TAXONOMY: tuple[type[CodingJEPAError], ...] = (
    CodingJEPAError,
    UsageError,
    ConfigError,
    ArtifactError,
    ManifestHashMismatch,
    CheckpointHashMismatch,
    IndexHashMismatch,
    DataError,
    SchemaVersionMismatch,
    ProvenanceMissing,
    DedupContractViolation,
    SplitContractViolation,
    ModelError,
    EmbeddingCollapse,
    LossDivergence,
    ParamCountMismatch,
    DeterminismViolation,
    SandboxError,
    SandboxTimeout,
    SandboxMemoryExceeded,
    SandboxNetworkAttempted,
    InternalError,
)

_CODE_PATTERN = re.compile(r"^E_[A-Z0-9_]+$")


def _all_subclasses(root: type) -> set[type]:
    seen: set[type] = set()
    stack = [root]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return seen


__all__ = [
    "CLOSED_TAXONOMY",
    "ArtifactError",
    "CheckpointHashMismatch",
    "CodingJEPAError",
    "ConfigError",
    "DataError",
    "DedupContractViolation",
    "DeterminismViolation",
    "EmbeddingCollapse",
    "IndexHashMismatch",
    "InternalError",
    "LossDivergence",
    "ManifestHashMismatch",
    "ModelError",
    "ParamCountMismatch",
    "ProvenanceMissing",
    "SandboxError",
    "SandboxMemoryExceeded",
    "SandboxNetworkAttempted",
    "SandboxTimeout",
    "SchemaVersionMismatch",
    "SplitContractViolation",
    "UsageError",
    "_CODE_PATTERN",
    "_all_subclasses",
]
