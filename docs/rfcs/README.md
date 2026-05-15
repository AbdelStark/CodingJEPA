# RFC index — CodingJEPA

These RFCs lock the design before implementation grows. Read them in order. Each RFC has a `Status` (Draft / Proposed / Locked / Superseded).

## Foundation

- [RFC-0001 — Problem thesis and MVP scope](RFC-0001-problem-thesis-and-mvp-scope.md)
- [RFC-0002 — Dataset mining and intent labeling](RFC-0002-dataset-mining-and-intent-labeling.md)
- [RFC-0012 — Code chunking and tokenization](RFC-0012-code-chunking-and-tokenization.md)
- [RFC-0014 — Licensing, deduplication, data ethics](RFC-0014-licensing-dedup-data-ethics.md)

## Modeling

- [RFC-0003 — Encoder, predictor, projector stack](RFC-0003-encoder-predictor-decoder-stack.md)
- [RFC-0004 — Refactor intent contract](RFC-0004-refactor-intent-contract.md)
- [RFC-0008 — Training recipe and pipeline (LeWorldModel-derived)](RFC-0008-training-recipe-and-pipeline.md)

## Inference and demo

- [RFC-0009 — Inference pipeline and serving](RFC-0009-inference-pipeline.md)
- [RFC-0006 — Demo and developer workflow](RFC-0006-demo-and-developer-workflow.md)
- [RFC-0007 — Failure modes and safety rails](RFC-0007-failure-modes-and-safety-rails.md)

## Evaluation and publication

- [RFC-0005 — Evaluation and baselines](RFC-0005-evaluation-and-baselines.md)
- [RFC-0010 — Evaluation harness and benchmarks](RFC-0010-evaluation-harness-and-benchmarks.md)
- [RFC-0011 — Paper outline and target venue](RFC-0011-paper-outline-and-venue.md)

## Infrastructure

- [RFC-0013 — Compute, infrastructure, reproducibility](RFC-0013-compute-infrastructure-reproducibility.md)

## Conventions

- An RFC moves to `Status: Locked` only when every decision in the "Decisions locked" section has a concrete value (no "TBD").
- New decisions amend the RFC with a dated entry rather than rewriting silently.
- A superseded RFC stays in the repo with a pointer to the replacement.

## RFC structure

The RFCs in this directory predate the spec corpus expansion of 2026-05-15 and use a project-local template:

| Section | Equivalent in the canonical RFC template |
|---|---|
| `Status` | `Status` |
| `Problem` | `Summary` + `Motivation` |
| `Context` (when present) | extension of `Motivation` |
| `Decisions locked` | `Proposed Design` (with decisions explicitly named D1, D2, …) |
| `Deferred items` | `Non-Goals` + `Open Questions` |
| `Acceptance condition` | `Testing Strategy` (the conditions that prove the RFC is implemented correctly) |

The two templates are functionally equivalent. `Alternatives Considered` and explicit `Drawbacks` sections are folded into the `Decisions locked` rationale paragraphs of each RFC; where a load-bearing alternative was rejected, the rejection is named (see RFC-0008 §D15 for the LeWM-vs-CodingJEPA comparison and RFC-0012 §D3 for chunk-granularity rationale).

New RFCs (`RFC-0015+`) follow the canonical template documented in the project bootstrap goal (`docs/goal.md` §Phase 1.4). Existing RFCs will be amended with `Alternatives Considered` and `Drawbacks` sections only when a concrete reviewer finds a gap; we do not retrofit form for form's sake.
