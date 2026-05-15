# CodingJEPA — Canonical specification

CodingJEPA is a single-GPU research artifact: a Joint-Embedding Predictive Architecture trained on Python code with a next-embedding prediction loss across code chunks and a sliced isotropic Gaussian regularizer over the embedding distribution. This document is the entry point. Detail lives in `docs/spec/` and `docs/rfcs/`.

## Thesis

A JEPA trained on Python code with (a) a next-embedding prediction loss across code chunks and (b) a sliced isotropic Gaussian regularizer over the embedding distribution learns semantic representations that beat parameter- and FLOP-matched token-level baselines on narrow refactoring and code-understanding tasks.

## Scope of v1

Eight refactor intents, ten curated source repositories, ~44–46M-parameter encoder + autoregressive predictor + projector stack, two-stage training (unconditional pretraining → intent-conditioned fine-tune), retrieval-and-rerank inference, deterministic eval harness, single-command reproduction. No multi-language. No general-purpose code generation. No frontier-scale claims.

## Document map

### Top-level

- `SPEC.md` — this file. Index and executive summary.
- `README.md` — project overview, RFC stack, contributor entry point.
- `docs/prd/PRD.md` — product requirements (what the artifact must be).

### Per-section spec (`docs/spec/`)

- `00-overview.md` — thesis, goals, non-goals, success criteria, audience.
- `01-architecture.md` — system layers, module boundaries, data flow.
- `02-public-api.md` — Python package surface, CLI surface, HTTP surface, versioning policy.
- `03-data-model.md` — schemas, on-disk formats, invariants, schema versioning.
- `04-error-model.md` — error taxonomy, exit codes, refusal contract, recovery.
- `05-observability.md` — logging, metrics, tracing, redaction.
- `06-security.md` — threat model, sandbox boundaries, secrets handling, supply chain.
- `07-testing-strategy.md` — pyramid, ML-specific tests, property tests, CI gates.
- `08-performance-budget.md` — training compute, inference latency, memory.
- `09-release-and-versioning.md` — semver, deprecation, changelog, support policy.
- `10-glossary.md` — canonical terms.

### Supporting design (referenced from spec sections)

- `docs/spec/SYSTEM-SPEC.md` — the long-form system spec. Section anchors are referenced from `01-architecture.md` and `03-data-model.md`.
- `docs/spec/RESEARCH.md` — research notes, prior art, hypotheses.
- `docs/spec/IMPLEMENTATION-PLAN.md` — phased execution plan; the issue tracker derives from this.
- `docs/spec/SCHEDULE.md` — milestones and budgets.
- `docs/spec/MVP-STATUS.md` — current implementation state.
- `docs/data/CANDIDATE_REPOS.md` — the 10 source repositories.

### RFCs (`docs/rfcs/`)

- `RFC-0001` — Problem thesis and MVP scope.
- `RFC-0002` — Dataset mining and intent labeling.
- `RFC-0003` — Encoder, predictor, projector stack.
- `RFC-0004` — Refactor intent contract.
- `RFC-0005` — Evaluation and baselines.
- `RFC-0006` — Demo and developer workflow.
- `RFC-0007` — Failure modes and safety rails.
- `RFC-0008` — Training recipe and pipeline (LeWorldModel-derived).
- `RFC-0009` — Inference pipeline and serving.
- `RFC-0010` — Evaluation harness and benchmarks.
- `RFC-0011` — Paper outline and target venue.
- `RFC-0012` — Code chunking and tokenization.
- `RFC-0013` — Compute, infrastructure, reproducibility.
- `RFC-0014` — Licensing, deduplication, data ethics.

### Roadmap (`docs/roadmap/`)

- `IMPLEMENTATION.md` — the implementation tracker. Every implementable unit of work is a GitHub issue; this doc is the index keyed by milestone, area, RFC, and dependencies.

## Acceptance bar (from RFC-0001 §D6)

| Metric | Threshold |
|---|---|
| Retrieval@10 | ≥ 1.5× strongest baseline (frozen CodeBERT) |
| Intent-conditioned hit rate | ≥ 2× unconditional latent baseline |
| Execution-preservation pass rate (500-pair subset) | ≥ 70% |
| Formatting-invariance rank change | < 5% under PEP-8-preserving perturbations |

If any threshold misses at end of Phase 3, the project halts and re-opens RFC-0003 / RFC-0008.

## Open questions still owned by the spec

The locked RFCs leave a small number of open questions, owned and dated. They are tracked in the per-RFC "Deferred items" sections and surfaced in the implementation tracker as `priority:p2` issues if they touch v1 scope. None block the v1 critical path.

## Conventions

- Specs lock decisions; they do not list options. An option that has not been chosen is an `OPEN QUESTION` with an owner.
- Diagrams supplement prose; they never replace it.
- Every cross-reference uses a stable file path and section anchor, not "see above."
- No tool/vendor branding, no marketing language, no emojis.
- Numbers are reproducible from a single command (`make eval`) or they are not reported.
