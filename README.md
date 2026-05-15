# CodingJEPA

> A Joint-Embedding Predictive Architecture for Python code.

CodingJEPA is a small-scale, single-GPU-friendly research project that asks: can a JEPA-style model — trained on Python code with a next-embedding prediction loss and a distributional regularizer — learn semantic representations that beat token-level baselines on narrow code understanding and refactoring tasks?

The project is anchored in the **LeWorldModel** training recipe (stable end-to-end JEPA from raw inputs, no EMA target encoder, two-term loss) but adapted to **code as the modality** and **edit intents as the action conditioning**.

## Core thesis

> A JEPA trained on Python code with (a) a next-embedding prediction loss across code chunks and (b) a sliced Gaussian regularizer over the embedding distribution, learns semantic representations that transfer to narrow refactoring and code understanding tasks better than token-level baselines of comparable scale.

## What v1 is

- A **two-stage training pipeline**:
  1. **Unconditional pretraining** on Python files mined from 10 curated repositories. The predictor learns to forecast the next code chunk's latent from a short history of chunks.
  2. **Intent-conditioned fine-tuning** on `(before, intent, after)` refactor pairs, where the intent embedding plays the role of the "action" in LeWorldModel.
- A **compact model** (encoder + autoregressive predictor, ~40–60M parameters) trainable on a single 80GB GPU in days, not weeks.
- A **narrow refactor intent taxonomy** (8 intents) with deterministic acceptance rules.
- A **demo** that takes a Python snippet + intent, retrieves and reranks candidate target chunks in the learned latent space, and presents a diff.
- An **evaluation harness** with cheap baselines, semantic and syntactic robustness probes, and a curated benchmark.

## What v1 is not

- Not a general-purpose code generation system.
- Not a multi-language model (Python only).
- Not a from-scratch attempt at frontier-scale pretraining.
- Not a replacement for compilers, linters, or IDE refactoring tools.

## Project structure

- `docs/prd/PRD.md` — product requirements
- `docs/spec/SYSTEM-SPEC.md` — system architecture and design contracts
- `docs/spec/RESEARCH.md` — research notes, prior art, open questions
- `docs/spec/IMPLEMENTATION-PLAN.md` — phased execution plan
- `docs/spec/SCHEDULE.md` — milestones
- `docs/spec/MVP-STATUS.md` — current implementation state
- `docs/data/CANDIDATE_REPOS.md` — the 10 curated Python source repositories
- `docs/rfcs/` — decision records that lock the design before code expands

## RFC stack

| RFC | Topic |
|---|---|
| 0001 | Problem thesis and MVP scope |
| 0002 | Dataset mining, deduplication, intent labeling |
| 0003 | Encoder, predictor, projector stack |
| 0004 | Refactor intent contract |
| 0005 | Evaluation and baselines |
| 0006 | Demo and developer workflow |
| 0007 | Failure modes and safety rails |
| 0008 | Training recipe and pipeline (LeWorldModel-derived) |
| 0009 | Inference pipeline and serving |
| 0010 | Evaluation harness and benchmarks |
| 0011 | Paper outline and target venue |
| 0012 | Code chunking and tokenization |
| 0013 | Compute, infrastructure, reproducibility |
| 0014 | Licensing, data ethics, deduplication |

## Build principle

The repository is **spec-first**. Implementation begins only after the PRD, system spec, and RFC stack form a self-consistent contract that makes the first build phase mechanical. Each RFC locks specific decisions; nothing is left as "to be decided" by the implementer.

## Inheritance from LeWorldModel

CodingJEPA reuses the LeWorldModel architectural skeleton — a frozen-graph encoder/predictor/projector stack with a sliced-Gaussian regularizer instead of an EMA target — and replaces the visual backbone with a Python-aware code transformer. The full mapping is documented in `docs/rfcs/RFC-0008-training-recipe-and-pipeline.md`.
