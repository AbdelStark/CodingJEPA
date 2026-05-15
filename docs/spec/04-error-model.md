# 04 — Error model

This document specifies how the system represents, propagates, and reports failure. The error taxonomy is closed; adding a class requires an RFC amendment.

## Goals

- Every failure is named, has a stable identifier, and a documented response.
- The system never silently emits a wrong refactor: refusal is a first-class outcome, not an error.
- Operator-facing failures use stable strings so that downstream tooling, logs, and integration tests are robust.
- Internal exceptions carry enough context to root-cause without re-running.

## Three classes of outcome

| Class | Meaning | Surface |
|---|---|---|
| **Success** | A candidate set was produced and at least one passed acceptance + safety + threshold. | CLI exit 0; HTTP 200 with `candidates`. |
| **Refusal** | The system declined to recommend; the source is well-formed but no acceptable candidate was found. | CLI exit 2; HTTP 200 with `refusal`. *Not an error.* |
| **Error** | The system could not run to completion (bad input, missing artifact, internal bug, sandbox failure). | CLI exit 1/3/4; HTTP 4xx/5xx. |

Refusal vs. error is a load-bearing distinction. Refusals are *expected* and are part of the contract; errors are *defects* (operational or programmer).

## Refusal taxonomy

All refusals come from the stable copy table at `codingjepa/safety/messages.py`. The set is enumerated below and is the closed contract for v1. Each refusal has a stable `code` (machine-readable) and a `message` (human-readable, displayed verbatim).

| Code | When | Source |
|---|---|---|
| `R001_SOURCE_PARSE_FAILED` | Source does not `compile()` under Python 3.12. | RFC-0007 §D2, RFC-0009 §D2 |
| `R002_SOURCE_TOO_LONG` | Source exceeds the 512-BPE-token chunk cap. | RFC-0007 §D2 |
| `R003_SOURCE_EMPTY` | Source is empty after the chunker normalizes it. | RFC-0007 §D2 |
| `R004_NO_CANDIDATE_PASSED_ACCEPTANCE` | All top-`k` candidates failed the chosen intent's acceptance rule. | RFC-0004 §D4 |
| `R005_CONFIDENCE_BELOW_THRESHOLD` | Top-1 cosine after rerank is below `τ_refuse = 0.55`. | RFC-0007 §D2 |
| `R006_SAFETY_CHECKER_REJECTED_ALL` | All candidates were filtered by safety checkers (RFC-0007 §D1). | RFC-0007 §D1, §D2 |
| `R007_INTENT_UNSUPPORTED_ON_SOURCE` | The chosen intent's acceptance rule cannot in principle apply (e.g., `comprehension-rewrite` on a source with `break`). | RFC-0004 §D2 |

**Operator response on refusal:** the UI/CLI surfaces the refusal verbatim plus the offending intent and a one-line "why this happened" derived from the failed checker. The user is never auto-redirected to a different intent; that is their decision.

**Audit:** every refusal is recorded in `.runs/demo-log.jsonl` (RFC-0007 §D5) as `{ts, source_hash, intent, refusal_code}`. The log is gitignored but is the input to "shadow eval" runs.

## Error taxonomy

All errors raise an exception in the `codingjepa.errors` module. Top-level exceptions inherit from `CodingJEPAError`. The taxonomy is closed; adding a class is an RFC amendment.

```
CodingJEPAError
├── UsageError                  # bad CLI flags, malformed HTTP body, unknown intent
├── ConfigError                 # missing or unresolvable Hydra config
├── ArtifactError               # missing checkpoint, missing index, hash mismatch
│   ├── ManifestHashMismatch
│   ├── CheckpointHashMismatch
│   └── IndexHashMismatch
├── DataError
│   ├── SchemaVersionMismatch
│   ├── ProvenanceMissing
│   ├── DedupContractViolation
│   └── SplitContractViolation
├── ModelError
│   ├── EmbeddingCollapse       # rank diagnostic gate
│   ├── LossDivergence          # pred_loss not monotone in first 5k steps
│   └── ParamCountMismatch
├── DeterminismViolation        # torch.use_deterministic_algorithms tripped at inference
├── SandboxError                # execution-preservation sandbox failures
│   ├── SandboxTimeout
│   ├── SandboxMemoryExceeded
│   └── SandboxNetworkAttempted
└── InternalError               # bug; wraps the original exception with context
```

Each class:

- carries a `code` attribute (e.g., `E_MANIFEST_HASH_MISMATCH`) for log/event correlation;
- carries a `context` dict with the relevant identifiers (paths, hashes, batch indices);
- formats as a single line on the CLI plus a structured payload in the JSON log.

## Exit codes (CLI contract)

| Exit | Meaning |
|---|---|
| 0 | Success. |
| 1 | `UsageError`. |
| 2 | Refusal. |
| 3 | `InternalError` or any unhandled exception (the CLI catches and re-raises as 3 with context). |
| 4 | Hash-mismatch in `make eval` (`ManifestHashMismatch` / `CheckpointHashMismatch` / `IndexHashMismatch`). |

Exit code 1 vs. 3 is the operator/programmer distinction. Tests assert exit codes, not stderr text.

## HTTP status mapping

| Status | When |
|---|---|
| `200` | Success **or** refusal (refusal is in the response body, not the status). |
| `400` | `UsageError`: malformed body, unknown intent. |
| `413` | Source body > 64 KB. |
| `409` | `ArtifactError`: hash mismatch detected at startup. |
| `503` | Model/index failed to load at startup. |
| `500` | `InternalError`. |

The body always includes `{error_code, message, request_id}` for non-2xx responses; `request_id` is a UUIDv7 generated per request and logged.

## Error propagation rules

1. **At system boundaries**, errors are translated to the user-facing form (CLI exit code or HTTP status).
2. **Inside the package**, errors are typed exceptions from `codingjepa.errors`. Bare `RuntimeError`/`ValueError` from underlying libraries are wrapped at the boundary they cross into `codingjepa.*` code.
3. **Never swallow.** No `except Exception: pass` outside of the demo log writer (which catches once, logs the swallowed exception with `code=E_LOG_WRITE_FAILED`, and continues).
4. **Context first.** Every wrap includes the inputs that triggered the failure: file paths, chunk ids, hashes, hyperparameters. Stack traces are kept; they do not replace context.
5. **Determinism violations are errors.** If `torch.use_deterministic_algorithms(True)` raises at inference time, we surface `DeterminismViolation` rather than silently fall back to non-determinism.

## Recovery rules

| Failure | Recovery |
|---|---|
| `ManifestHashMismatch` (eval) | Print the diff (expected vs. actual hashes); exit 4. Operator must explicitly re-pin or re-run upstream pipeline. No auto-rebuild. |
| Mid-training `EmbeddingCollapse` | Halt the run (`SIGTERM` worker). Dump rank diagnostics to `runs/<id>/collapse.json`. Operator re-launches with adjusted config. |
| Mid-training `LossDivergence` | Same as collapse: halt, dump, do not auto-recover. |
| `SandboxTimeout` in CJ-EXEC | Mark the pair as `passed=false, reason="timeout"`. Continue the benchmark. The aggregated pass-rate includes timeouts as failures. |
| `SandboxNetworkAttempted` | Same: mark `passed=false`, continue. |
| FAISS load failure at startup | HTTP 503; CLI exit 3. The process does not start. |
| Tokenizer hash mismatch | Treated as `ArtifactError`; refuse to proceed. |

The principle: when the system is in an unknown state, stop and surface; do not patch and continue.

## Logging contract for errors

Every error class emits one structured log record at the boundary where it surfaces, with at minimum:

```json
{
  "ts": "ISO-8601 UTC",
  "level": "error",
  "request_id": "<uuid7|null>",
  "error_code": "E_*",
  "message": "human-readable",
  "context": { ... }
}
```

See `docs/spec/05-observability.md` for the full logging contract.

## Tests

- `tests/test_errors.py` enumerates the closed error taxonomy and asserts every class exposes `code` and `context`.
- `tests/test_refusal_messages.py` enumerates the closed refusal copy table.
- `tests/test_cli_exit_codes.py` asserts exit codes for each operator-facing scenario.
- `tests/test_http_errors.py` asserts the HTTP status mapping with `httpx.AsyncClient`.
- Property test in `tests/test_safety_chain.py` verifies the safety filter never returns a candidate that violates its checker (RFC-0007 §D6).
