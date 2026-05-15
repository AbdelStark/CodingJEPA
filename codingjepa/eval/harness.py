"""Eval harness hash-check enforcer. RFC-0010 §D1, spec/04 exit code 4.

`make eval` refuses to run if any of the manifest, checkpoint, or index hashes
drift from the values recorded in `MODEL_CARD.md`. The diff is printed before
the CLI exits with code 4 (mapped from any of the `*HashMismatch` exceptions
in `codingjepa.errors`).
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from codingjepa.errors import (
    ArtifactError,
    CheckpointHashMismatch,
    IndexHashMismatch,
    ManifestHashMismatch,
    UsageError,
)

# ---- Public API -------------------------------------------------------------


@dataclass(frozen=True)
class HashTriple:
    """Identifiers compared between the model card and the runtime artifacts."""

    manifest_hash: str
    checkpoint_hash: str
    index_id: str


def check_hashes(model_card: HashTriple, runtime: HashTriple) -> None:
    """Raise a specific HashMismatch if any of the three hashes drift.

    Order of checks (deterministic for the diff printer):
    1. manifest_hash → ManifestHashMismatch
    2. checkpoint_hash → CheckpointHashMismatch
    3. index_id → IndexHashMismatch
    """

    if model_card.manifest_hash != runtime.manifest_hash:
        raise ManifestHashMismatch(
            "manifest_hash drift",
            expected=model_card.manifest_hash,
            actual=runtime.manifest_hash,
        )
    if model_card.checkpoint_hash != runtime.checkpoint_hash:
        raise CheckpointHashMismatch(
            "checkpoint_hash drift",
            expected=model_card.checkpoint_hash,
            actual=runtime.checkpoint_hash,
        )
    if model_card.index_id != runtime.index_id:
        raise IndexHashMismatch(
            "index_id drift",
            expected=model_card.index_id,
            actual=runtime.index_id,
        )


def format_hash_diff(exc: ArtifactError) -> str:
    """Render the operator-facing diff line printed before exit code 4."""

    expected = exc.context.get("expected", "<unknown>")
    actual = exc.context.get("actual", "<unknown>")
    return f"{exc.code}: expected={expected!r} actual={actual!r}"


# ---- MODEL_CARD.md front-matter parser --------------------------------------

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$")


def parse_model_card_front_matter(path: pathlib.Path | str) -> dict[str, str]:
    """Extract the YAML front-matter block from `MODEL_CARD.md` as a flat dict.

    We restrict to flat `key: value` lines so we do not need the YAML library
    (RFC-0013 §D3 dependency-creep rule). Comments, lists, and nested objects
    are rejected with UsageError.
    """

    path = pathlib.Path(path)
    text = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        raise UsageError("MODEL_CARD.md missing YAML front-matter", path=str(path))

    out: dict[str, str] = {}
    for line_no, raw in enumerate(match.group(1).splitlines(), start=2):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        kv = _KV_RE.match(line)
        if kv is None:
            raise UsageError(
                "unsupported front-matter line; expected flat `key: value`",
                path=str(path),
                line=line_no,
                content=line,
            )
        out[kv.group(1)] = _unquote(kv.group(2))
    return out


def _unquote(value: str) -> str:
    """Strip a single layer of matching single or double quotes."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def hash_triple_from_model_card(path: pathlib.Path | str) -> HashTriple:
    """Read MODEL_CARD.md and extract the three identifiers that gate `make eval`."""

    front = parse_model_card_front_matter(path)
    required = {"manifest_hash", "checkpoint_hash", "index_id"}
    missing = required - front.keys()
    if missing:
        raise UsageError(
            "MODEL_CARD.md missing required hash fields",
            path=str(path),
            missing=sorted(missing),
        )
    return HashTriple(
        manifest_hash=front["manifest_hash"],
        checkpoint_hash=front["checkpoint_hash"],
        index_id=front["index_id"],
    )


__all__ = [
    "HashTriple",
    "check_hashes",
    "format_hash_diff",
    "hash_triple_from_model_card",
    "parse_model_card_front_matter",
]
