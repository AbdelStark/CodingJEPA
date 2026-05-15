# RFC-0007 — Failure modes and safety rails

## Status
Locked (2026-05-15)

## Problem

Define classes of unsound or low-confidence transformations and what the system does about them, so we never silently emit a bad refactor.

## Decisions locked

### D1 — Unsafe transform classes (refuse, never auto-apply)
- **Side-effect introduction.** Candidate adds new IO, network, or mutation that was not in `chunk_before`.
- **Side-effect elimination.** Candidate removes an observable side effect (`print`, `logger.*`, file write) that existed in `chunk_before`. (Even if the user *wanted* this, v1 does not auto-apply it.)
- **Exception-contract change.** Candidate changes the set of exceptions raised. Detected by a coarse heuristic: AST diff on `Raise` nodes and on the `except` clauses that handle them.
- **Public-API change.** Candidate changes function name, parameter list, or return-type annotation when the chunk has `__all__` or is a module-level public name.
- **Async/sync boundary change.** Candidate makes a sync function `async` or vice versa.

Each class has a static checker in `codingjepa/safety/checkers/*.py`. A candidate that fails any checker is filtered out *before* it reaches the user.

### D2 — Refusal / no-op rules
The system returns "no acceptable candidate" (CLI exit 2) when:
- the top-1 cosine similarity after rerank is below `τ_refuse = 0.55`;
- no top-`k` candidate passes the intent acceptance rule (RFC-0004 §D2);
- the source chunk fails to `compile()` under Python 3.12;
- the source chunk exceeds the 512-BPE-token cap;
- the source chunk is empty after the chunker normalizes it.

### D3 — Confidence and uncertainty display
The UI/CLI displays:
- raw cosine similarity (0–1);
- a calibrated confidence in `[0, 1]` derived from a temperature-`τ = 0.1` softmax over the top-`k` cosine distances;
- an explicit warning band (red) when confidence `< 0.5`, (yellow) `< 0.75`, (green) otherwise.

Confidence is **not** a probabilistic guarantee; it is reported as a heuristic with documented limits.

### D4 — Scope boundaries for v1
v1 does **not** attempt:
- multi-step refactors (chained intents);
- cross-file refactors;
- automatic test execution as a guard before showing a candidate (this is done only in the offline execution-preservation eval);
- automatic application of a refactor to a file on disk — the demo *displays* diffs, it does not write them.

A `--apply` flag is reserved for v2.

### D5 — Auditability
- Every refactor query writes a JSONL record to `.runs/demo-log.jsonl`:
  `{ts, source_hash, intent, top_k_hashes, scores, accepted_idx, user_marked}`.
- The log is gitignored but is essential for failure-mode analysis.
- The eval harness can be re-run against this log to compute "shadow" metrics from real usage.

### D6 — Test coverage requirements
- Each safety checker has unit tests with ≥ 5 positive and ≥ 5 negative examples.
- The full safety filter chain is property-tested: random-mutate `chunk_before` such that the chosen intent's invariant is violated, and confirm the filter catches it.

### D7 — Failure documentation
- Every refusal reason is a stable string from `codingjepa/safety/messages.py`.
- The set of stable strings is enumerated in this RFC's appendix table (see commit history); adding a new reason requires an RFC amendment.

## Deferred items
- A learned calibration head over confidence.
- An "explain why I refused" surface beyond the static one-line reason.
- Integration with project linters/type checkers as a pre-acceptance gate.

## Acceptance condition

Locked when:
- all checkers in D1 exist and pass their unit tests;
- the property test for the filter chain runs in CI;
- the refusal message table is committed.
