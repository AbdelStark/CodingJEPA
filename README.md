# RustJEPA

> JEPA-style semantic refactoring for Rust code

RustJEPA explores whether refactoring is better treated as latent prediction than token generation. The project focuses on semantic code transformation for Rust, where types, ownership, data flow, and API intent matter more than exact tokens, formatting, or naming.

## Core thesis

A JEPA-style model trained on before/after Rust refactor pairs can learn semantic invariants and predict intent-conditioned target representations that are better suited to narrow refactoring tasks than token-level generation alone.

## Problem statement

Existing code assistants treat refactoring as next-token generation or retrieval with weak semantic grounding. For Rust, small surface edits can hide large semantic changes, while semantically equivalent refactors can look very different at the token level. Generic code generation is too broad for a credible small project and usually optimizes for the wrong target.

## What v1 is

- Mine and filter high-quality Rust refactor pairs from real repositories.
- Support a narrow intent taxonomy such as extract-trait, allocation-reduction, iterator-ification, and error-handling cleanup.
- Use a compact code encoder plus JEPA-style predictor and a practical decode path.
- Ship a demo that shows code input, selected intent, ranked refactor proposals, and diff inspection.

## What v1 is not

- No general-purpose code generation system.
- No broad multi-language support in v1.
- No claim of replacing compiler, lints, or full IDE workflows.
- No full-scale code-model pretraining from scratch.

## Candidate data sources

- Filtered Rust commit pairs from stable OSS repositories
- Intent-labeled subset for narrow refactor tasks
- Optional synthetic augmentation for small controlled transformations

## Success criteria

- Higher semantic preservation than token-level baselines on the chosen refactor tasks.
- Compilation- and test-preserving transforms on the curated benchmark subset.
- Demo outputs that make latent refactoring feel materially different from autocomplete.

## Milestones

1. Repo and commit mining pipeline
2. Intent taxonomy and gold subset
3. Encoder/predictor baseline
4. Decode / retrieval / reranking path
5. Demo UI and evaluation harness

## Repository structure

- `docs/prd/PRD.md` — product requirements
- `docs/spec/SYSTEM-SPEC.md` — system architecture and design contracts
- `docs/spec/RESEARCH.md` — research notes and open questions
- `docs/spec/IMPLEMENTATION-PLAN.md` — phased execution plan
- `docs/spec/MVP-STATUS.md` — current implementation state
- `docs/rfcs/` — decision records that lock the design before code expands

## Build principle

The repository is spec-first by design.
Implementation should follow only after the PRD, system spec, and RFC stack are coherent enough to make the first build phase mechanical.
