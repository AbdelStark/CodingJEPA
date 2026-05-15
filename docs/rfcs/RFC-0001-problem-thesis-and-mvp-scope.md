# RFC-0001 — Problem thesis and MVP scope

## Status
Locked (2026-05-15)

## Problem

Fix the project's thesis, scope boundary, and acceptance bar so downstream RFCs and implementation choices are decidable.

## Context

The project was renamed from RustJEPA to CodingJEPA and retargeted from Rust to Python on 2026-05-15. The motivation:
- a larger and more diverse corpus of refactor pairs in Python;
- richer dynamic-semantics surface (so the gap between syntactic and semantic similarity is sharper);
- existing high-quality CST tooling (`libcst`, `parso`, `ast`);
- executable unit tests across most of the corpus, enabling execution-preservation eval.

## Decisions locked

### D1 — One-line thesis
A JEPA trained on Python code with a next-embedding prediction loss across chunks and a sliced Gaussian regularizer over the embedding distribution learns semantic representations that beat parameter- and FLOP-matched token-level baselines on narrow refactoring and code-understanding tasks.

### D2 — Supported task family
Two task families only, in v1:
1. **Intent-conditioned latent retrieval.** Given `(source_chunk, intent_label)`, retrieve the latent of the post-refactor chunk from a candidate pool.
2. **Linear-probe code understanding.** Standard linear probes on the frozen encoder for: function-name prediction, defect-detection (Devign), code-pair clone detection (BigCloneBench Python subset, where available).

Anything else (free-form generation, multi-step planning, IDE integration) is **out of scope** in v1.

### D3 — One-line product framing
A single-GPU, single-language, single-corpus research artifact that produces (a) a checkpoint, (b) a written results memo, (c) a paper draft. Not a platform.

### D4 — Demo loop (one and only one)
Paste a Python snippet → choose one of 8 intents → see ranked candidate target chunks with diff inspection and per-candidate latent distance + confidence.

### D5 — Anti-goals
- No general-purpose code generation.
- No multi-language support in v1.
- No claim of replacing compiler, linter, type checker, or IDE refactor tooling.
- No from-scratch decoder-only LM pretraining.
- No comparison against frontier models on absolute task performance (the claim is matched-compute, matched-params).

### D6 — Acceptance bar for v1
- **Retrieval@10** ≥ 1.5× the strongest baseline (frozen CodeBERT).
- **Intent hit rate** ≥ 2× the unconditional latent baseline.
- **Execution-preservation pass rate** ≥ 70% on the 500-pair executable subset.
- **Formatting-invariance:** rank change < 5% under PEP-8-preserving perturbations.

If any of these are missed at end of Phase 3, the project halts and re-opens RFC-0003 and RFC-0008 rather than continuing into demo/paper phases.

### D7 — What would count as a fake win
- Beating BM25 only because of corpus contamination (mitigated by RFC-0014 dedup).
- High retrieval@k against a tiny pool (we report at N=100 *and* N=1000).
- Cherry-picked intents (we report per-intent and macro-average).
- Best-of-seeds (we report mean ± std over ≥ 3 seeds).

## Deferred items

- Multi-step latent planning (chained refactors).
- Cross-file refactors.
- Multi-language extension.
- A generative decoder head over the frozen latents.

## Acceptance condition

Locked when every decision (D1–D7) is concrete enough that the next RFC author does not have to guess what the project is trying to prove.
