# RustJEPA — PRD

## 1. Product thesis

A JEPA-style model trained on before/after Rust refactor pairs can learn semantic invariants and predict intent-conditioned target representations that are better suited to narrow refactoring tasks than token-level generation alone.

## 2. Product goal

Build a compact, technically credible v1 that demonstrates the thesis with a working demo and evaluation surface, without requiring large-lab training budgets or broad platform scope.

## 3. Problem

Existing code assistants treat refactoring as next-token generation or retrieval with weak semantic grounding. For Rust, small surface edits can hide large semantic changes, while semantically equivalent refactors can look very different at the token level. Generic code generation is too broad for a credible small project and usually optimizes for the wrong target.

## 4. v1 scope

- Mine and filter high-quality Rust refactor pairs from real repositories.
- Support a narrow intent taxonomy such as extract-trait, allocation-reduction, iterator-ification, and error-handling cleanup.
- Use a compact code encoder plus JEPA-style predictor and a practical decode path.
- Ship a demo that shows code input, selected intent, ranked refactor proposals, and diff inspection.

## 5. Explicit non-goals

- No general-purpose code generation system.
- No broad multi-language support in v1.
- No claim of replacing compiler, lints, or full IDE workflows.
- No full-scale code-model pretraining from scratch.

## 6. Users

Primary users:
- researchers and engineers evaluating whether the JEPA framing actually improves the target problem
- developers/operators who need a concrete, inspectable demo rather than a vague claim

Secondary users:
- open-source contributors joining after the initial prototype
- technical readers who want the core design decisions spelled out before implementation grows

## 7. Demo requirement

The project must support a short demo that makes the thesis obvious without hidden manual setup or cloud-only dependencies.

## 8. Success criteria

- Higher semantic preservation than token-level baselines on the chosen refactor tasks.
- Compilation- and test-preserving transforms on the curated benchmark subset.
- Demo outputs that make latent refactoring feel materially different from autocomplete.

## 9. Main risks

- weak task definition creates a fake win
- model scope expands faster than the data/eval contract becomes clear
- demo polish arrives before the underlying claim is actually supported
- baselines are too weak and make the result look better than it is

## 10. Deliverables for MVP

- reproducible data/task contract
- baseline system
- JEPA-based representation/training path
- downstream scoring or decode path
- demo surface
- evaluation report and exported artifacts
