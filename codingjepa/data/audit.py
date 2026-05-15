"""Per-repo audit + hard gates (RFC-0002 §D2, RFC-0014 §D10).

For each repo in the locked corpus we compute a single
:class:`AuditResult` and write it to ``data/audit/<owner>__<name>.json``.
The result records descriptive stats (file/chunk counts, drop rates,
duplication rate, intent breakdown) and evaluates the four hard gates:

* **compile_gate**: ``compile_ok_rate >= 0.95``
* **dedup_gate**:   ``duplication_rate < 0.30``
* **license_gate**: ``license_spdx in ALLOWED_LICENSES``
* **secrets_gate**: ``secret_scanner_hits == 0``

:func:`check_all_gates` is the pipeline's pre-flight: if any repo's
``passes_all_gates`` is ``False`` it raises
:class:`codingjepa.errors.AuditGateError` with the offending repo ids in the
context dict.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from codingjepa._jsonschema import load_schema, validate_record
from codingjepa.errors import AuditGateError

__all__ = [
    "ALLOWED_LICENSES",
    "AuditResult",
    "check_all_gates",
    "compute_audit",
    "load_audit",
    "write_audit",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "PSF-2.0",
        "BSD-3-Clause",
        "MIT",
        "Apache-2.0",
    }
)

_COMPILE_GATE_MIN = 0.95
_DEDUP_GATE_MAX = 0.30
_SCHEMA_VERSION = "v1"
_TOKEN_LEN_CAP = 512


# ---------------------------------------------------------------------------
# AuditResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class AuditResult:
    """Per-repo audit + gate evaluation.

    The first eleven fields map 1:1 onto ``data/schemas/audit.schema.json``.
    The four ``passes_*_gate`` fields plus ``passes_all_gates`` are
    derived booleans the pipeline uses to short-circuit downstream stages
    on failure; they are *not* serialized to the audit JSON.
    """

    repo: str
    commit_sha: str
    license_spdx: str
    py_files_in_scope: int
    chunk_count: int
    median_chunk_token_len: int
    drop_rate_over_cap: float
    drop_rate_parse_fail: float
    duplication_rate: float
    secret_scanner_hits: int
    compile_ok_rate: float
    per_intent_pair_count: dict[str, int] = field(default_factory=dict)

    # Derived gates -----------------------------------------------------
    passes_compile_gate: bool = False
    passes_dedup_gate: bool = False
    passes_license_gate: bool = False
    passes_secrets_gate: bool = False
    passes_all_gates: bool = False


# ---------------------------------------------------------------------------
# compute_audit
# ---------------------------------------------------------------------------


def compute_audit(
    repo: str,
    commit_sha: str,
    license_spdx: str,
    *,
    chunks: list[Any],
    pairs: list[Any] | None = None,
    dedup_result: Any | None = None,
    secret_hits: list[Any] | None = None,
    token_lengths: list[int] | None = None,
    py_files_in_scope: int | None = None,
    parse_fail_count: int = 0,
    over_cap_count: int = 0,
) -> AuditResult:
    """Compute the audit stats for one repo and evaluate the four gates.

    The function accepts loosely-typed inputs (``chunks``, ``pairs``,
    ``dedup_result``) so it can be wired into the pipeline without forcing
    a single canonical dataclass for each upstream stage. The shape we
    rely on:

    * each chunk exposes ``token_ids: list[int]`` (used to derive lengths
      when ``token_lengths`` is not provided);
    * each pair (if any) exposes a ``intent: str`` attribute or key;
    * ``dedup_result`` exposes ``duplication_rate: float``.

    ``parse_fail_count`` and ``over_cap_count`` are the totals the chunker
    accumulated *before* the surviving chunks were handed off to this
    function — i.e., the denominator is ``len(chunks) + parse_fail_count``.
    """

    chunk_count = len(chunks)
    secret_hits = secret_hits or []
    pairs = pairs or []

    # Token-length stats ------------------------------------------------
    if token_lengths is None:
        token_lengths = _token_lengths_from_chunks(chunks)
    median_len = int(statistics.median(token_lengths)) if token_lengths else 0

    # Drop rates --------------------------------------------------------
    total_candidates = chunk_count + parse_fail_count
    drop_rate_parse_fail = parse_fail_count / total_candidates if total_candidates > 0 else 0.0
    over_cap_total = chunk_count + over_cap_count
    drop_rate_over_cap = over_cap_count / over_cap_total if over_cap_total > 0 else 0.0
    compile_ok_rate = 1.0 - drop_rate_parse_fail

    # Duplication rate --------------------------------------------------
    duplication_rate = 0.0
    if dedup_result is not None:
        rate = getattr(dedup_result, "duplication_rate", None)
        if rate is None and isinstance(dedup_result, dict):
            rate = dedup_result.get("duplication_rate", 0.0)
        duplication_rate = float(rate or 0.0)

    # Secret hits -------------------------------------------------------
    secret_count = len(secret_hits)

    # Per-intent pair counts -------------------------------------------
    per_intent = _count_by_intent(pairs)

    # py_files_in_scope: caller may provide explicitly, else infer from
    # chunks' file_path attribute (distinct count).
    files_in_scope = (
        py_files_in_scope if py_files_in_scope is not None else _distinct_file_count(chunks)
    )

    # Gates -------------------------------------------------------------
    passes_compile = compile_ok_rate >= _COMPILE_GATE_MIN
    passes_dedup = duplication_rate < _DEDUP_GATE_MAX
    passes_license = license_spdx in ALLOWED_LICENSES
    passes_secrets = secret_count == 0
    passes_all = passes_compile and passes_dedup and passes_license and passes_secrets

    return AuditResult(
        repo=repo,
        commit_sha=commit_sha,
        license_spdx=license_spdx,
        py_files_in_scope=files_in_scope,
        chunk_count=chunk_count,
        median_chunk_token_len=median_len,
        drop_rate_over_cap=drop_rate_over_cap,
        drop_rate_parse_fail=drop_rate_parse_fail,
        duplication_rate=duplication_rate,
        secret_scanner_hits=secret_count,
        compile_ok_rate=compile_ok_rate,
        per_intent_pair_count=per_intent,
        passes_compile_gate=passes_compile,
        passes_dedup_gate=passes_dedup,
        passes_license_gate=passes_license,
        passes_secrets_gate=passes_secrets,
        passes_all_gates=passes_all,
    )


# ---------------------------------------------------------------------------
# write_audit / load_audit
# ---------------------------------------------------------------------------


def write_audit(
    result: AuditResult,
    output_dir: Path = Path("data/audit"),
) -> Path:
    """Write the audit JSON for one repo. Returns the path written.

    File name: ``<owner>__<name>.json`` (the manifest references the same
    pattern; double-underscore avoids collisions with repo names that
    contain a dot).
    """

    payload = _audit_to_payload(result)
    validate_record(payload, load_schema("audit"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / _audit_filename(result.repo)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def load_audit(path: Path) -> AuditResult:
    """Read an audit JSON back into an :class:`AuditResult` and re-evaluate gates."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_record(payload, load_schema("audit"))

    license_spdx = payload["license_spdx"]
    compile_ok = payload["compile_ok_rate"]
    dup_rate = payload["duplication_rate"]
    secret_hits = payload["secret_scanner_hits"]

    passes_compile = compile_ok >= _COMPILE_GATE_MIN
    passes_dedup = dup_rate < _DEDUP_GATE_MAX
    passes_license = license_spdx in ALLOWED_LICENSES
    passes_secrets = secret_hits == 0

    return AuditResult(
        repo=payload["repo"],
        commit_sha=payload["commit_sha"],
        license_spdx=license_spdx,
        py_files_in_scope=payload["py_files_in_scope"],
        chunk_count=payload["chunk_count"],
        median_chunk_token_len=payload["median_chunk_token_len"],
        drop_rate_over_cap=payload["drop_rate_over_cap"],
        drop_rate_parse_fail=payload["drop_rate_parse_fail"],
        duplication_rate=dup_rate,
        secret_scanner_hits=secret_hits,
        compile_ok_rate=compile_ok,
        per_intent_pair_count=dict(payload["per_intent_pair_count"]),
        passes_compile_gate=passes_compile,
        passes_dedup_gate=passes_dedup,
        passes_license_gate=passes_license,
        passes_secrets_gate=passes_secrets,
        passes_all_gates=passes_compile and passes_dedup and passes_license and passes_secrets,
    )


# ---------------------------------------------------------------------------
# check_all_gates
# ---------------------------------------------------------------------------


def check_all_gates(audits: list[AuditResult]) -> bool:
    """Return ``True`` if every repo passes every gate; raise otherwise.

    :raises AuditGateError: at least one repo's :attr:`AuditResult.passes_all_gates`
        is ``False``. The exception's ``context`` carries:

        * ``failing_repos``: sorted list of offending repo ids.
        * ``failures``: list of ``{repo, gate}`` dicts naming which gate
          tripped for each repo.
    """

    failing_repos: list[str] = []
    failures: list[dict[str, str]] = []
    for audit in audits:
        if audit.passes_all_gates:
            continue
        failing_repos.append(audit.repo)
        for gate_name, ok in (
            ("compile", audit.passes_compile_gate),
            ("dedup", audit.passes_dedup_gate),
            ("license", audit.passes_license_gate),
            ("secrets", audit.passes_secrets_gate),
        ):
            if not ok:
                failures.append({"repo": audit.repo, "gate": gate_name})

    if failing_repos:
        raise AuditGateError(
            f"audit gates failed for {len(failing_repos)} repo(s): "
            + ", ".join(sorted(failing_repos)),
            failing_repos=sorted(failing_repos),
            failures=failures,
        )
    return True


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _audit_filename(repo: str) -> str:
    """``owner/name`` -> ``owner__name.json``; falls back to a flat encoding."""

    if "/" in repo:
        owner, name = repo.split("/", 1)
        return f"{owner}__{name}.json"
    return f"{repo}.json"


def _audit_to_payload(result: AuditResult) -> dict[str, Any]:
    """Serialize an :class:`AuditResult` into the schema-conformant payload.

    The derived gate fields are stripped — the schema does not declare them
    and ``additionalProperties: false`` would reject the file otherwise.
    """

    blob = asdict(result)
    for derived in (
        "passes_compile_gate",
        "passes_dedup_gate",
        "passes_license_gate",
        "passes_secrets_gate",
        "passes_all_gates",
    ):
        blob.pop(derived, None)
    payload: dict[str, Any] = {"schema_version": _SCHEMA_VERSION, **blob}
    # Schema mandates exact float types for rates; coerce.
    payload["drop_rate_over_cap"] = float(payload["drop_rate_over_cap"])
    payload["drop_rate_parse_fail"] = float(payload["drop_rate_parse_fail"])
    payload["duplication_rate"] = float(payload["duplication_rate"])
    payload["compile_ok_rate"] = float(payload["compile_ok_rate"])
    return payload


def _token_lengths_from_chunks(chunks: list[Any]) -> list[int]:
    """Best-effort token length extraction from chunk objects."""

    lengths: list[int] = []
    for chunk in chunks:
        ids = getattr(chunk, "token_ids", None)
        if ids is not None:
            lengths.append(len(ids))
            continue
        source = getattr(chunk, "source_normalized", None) or ""
        if source:
            lengths.append(len(source.split()))
    return lengths


def _count_by_intent(pairs: list[Any]) -> dict[str, int]:
    """Bucket pairs by their ``intent`` attribute/key. Missing → ``unknown``."""

    counts: dict[str, int] = {}
    for pair in pairs:
        intent = getattr(pair, "intent", None)
        if intent is None and isinstance(pair, dict):
            intent = pair.get("intent")
        intent = str(intent) if intent is not None else "unknown"
        counts[intent] = counts.get(intent, 0) + 1
    return counts


def _distinct_file_count(chunks: list[Any]) -> int:
    """Count distinct ``file_path`` values across the chunks. 0 if none have it."""

    paths: set[str] = set()
    for chunk in chunks:
        file_path = getattr(chunk, "file_path", None)
        if file_path is None and isinstance(chunk, dict):
            file_path = chunk.get("file_path")
        if file_path:
            paths.add(str(file_path))
    return len(paths)
