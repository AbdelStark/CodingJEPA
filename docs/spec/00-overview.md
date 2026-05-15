# 00 — Overview

## Purpose

CodingJEPA is a research artifact that tests one claim at small scale on Python code: a Joint-Embedding Predictive Architecture trained with next-embedding prediction plus a sliced isotropic Gaussian regularizer learns representations that transfer to narrow refactoring and code-understanding tasks better than parameter- and FLOP-matched token-level baselines.

It is not a platform. It is not a product. It is a checkpoint, a results memo, a paper draft, and a demo that lets a reader inspect the claim end-to-end.

## Goals (v1)

- **Reproducible artifact.** A single command (`make eval`) regenerates every reported number from a released checkpoint and a content-addressed data manifest.
- **Honest comparison.** Three baselines from distinct families (lexical / MLM / pretrained encoder) on the same pool, tokenization, and split.
- **Narrow but credible task.** Eight refactor intents with deterministic acceptance rules; intent-conditioned latent retrieval as the primary metric.
- **Single-GPU training.** Pretrain ≤ 7 days, fine-tune ≤ 24 h on 1× H100 (80 GB).
- **Auditable demo.** CLI + minimal web UI displays diffs, latent distance, confidence, and explicit refusal reasons; never silently auto-applies a refactor.

## Non-goals (v1)

- General-purpose code generation.
- Multi-language support (Python only; see RFC-0001 §D5).
- Replacing compilers, linters, type checkers, or IDE refactor tools.
- From-scratch decoder-only LM pretraining.
- Beating frontier models on absolute task performance; the comparison is matched-compute, matched-params.
- Multi-step latent planning, cross-file refactors, generative head — explicitly deferred to v2 (see RFC-0001 deferred items).

## Success criteria (PRD §8, RFC-0001 §D6)

| Axis | Threshold | Source |
|---|---|---|
| Retrieval@10 | ≥ 1.5× best baseline | PRD §8 |
| Intent-conditioned hit rate | ≥ 2× unconditional baseline | PRD §8 |
| Execution-preservation rate (500-pair subset) | ≥ 70% | PRD §8 |
| Formatting-invariance rank change | < 5% | PRD §8 |
| Statistical significance | p < 0.05 paired bootstrap, ≥ 3 seeds | RFC-0005 §D7 |
| Inter-rater agreement on gold subset | Cohen's κ ≥ 0.7 | RFC-0002 §D7 |

If any of these are missed at end of Phase 3, the project halts and re-opens RFC-0003 / RFC-0008 rather than proceeding to demo/paper phases.

## Audience

- **Primary:** researchers and engineers evaluating whether the JEPA framing improves the target problem at small scale.
- **Secondary:** open-source contributors joining after the prototype; technical readers who want the design decisions before implementation grows; reviewers verifying a written claim against released artifacts.

## What "done" looks like

- `make data && make pretrain && make finetune && make eval` runs end-to-end on a 1× H100 host with the released data manifest.
- `results/RESULTS-MEMO.md` reports every metric in PRD §8 with confidence intervals and ≥ 3 seeds.
- `paper/main.tex` builds a workshop-grade PDF whose every figure and table traces to `results/results.json`.
- `make demo` launches the web UI; the deterministic example in `examples/` produces the same top-1 diff across runs.

## What "fail" looks like (and how we handle it)

- A PRD §8 threshold is missed → halt, re-open RFC-0003/0008 rather than ship a soft claim.
- An RFC's acceptance condition cannot be satisfied → amend the RFC (dated entry), re-derive the affected issues, do not silently lower the bar.
- A safety checker fires inappropriately on real refactors → file a bug, freeze the demo path on the offending checker until resolved.
- The compute budget overruns → decrease `B` and `S_A` proportionally, document in `docs/notes/PHASE-2.md` (RFC-0008 §D10).

## How this document set fits together

- **PRD** (`docs/prd/PRD.md`) — what the artifact must be.
- **System spec** (`docs/spec/SYSTEM-SPEC.md`) — the long-form architecture.
- **Per-section specs** (`docs/spec/0N-*.md`, this directory) — focused contracts: API, data model, errors, observability, security, testing, performance, release, glossary.
- **RFCs** (`docs/rfcs/RFC-NNNN-*.md`) — locked decisions, one per load-bearing concern.
- **Implementation tracker** (`docs/roadmap/IMPLEMENTATION.md`) — the issue index that, when closed, equals a shipped v1.

This file is the orientation. Everything specific is one link away.
