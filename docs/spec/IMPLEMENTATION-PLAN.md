# RustJEPA — implementation plan

## Goal

Build the smallest credible v1 that proves this thesis:

> A JEPA-style model trained on before/after Rust refactor pairs can learn semantic invariants and predict intent-conditioned target representations that are better suited to narrow refactoring tasks than token-level generation alone.

## Phase 0 — lock the design
1. Read PRD.
2. Read system spec.
3. Read all RFCs in order.
4. Update `docs/spec/MVP-STATUS.md` with any corrected assumptions.
5. Refuse implementation work that expands scope before the v1 contracts are explicit.

## Phase 1 — data and task definition
1. Audit the candidate data sources.
2. Define the exact sample / window / pair format.
3. Build a tiny gold subset for manual inspection.
4. Implement one cheap baseline before training anything expensive.
5. Write a short data/task note that captures assumptions and leakage risks.

## Phase 2 — minimal model path
1. Lock the exact backbone and model size.
2. Implement preprocessing and batching.
3. Run a tiny-slice training pass.
4. Verify that embeddings are non-collapsed and inspectable.
5. Add one robustness / invariance check tied to the project thesis.

## Phase 3 — downstream task path
1. Implement the smallest scoring/forecast/decode path that makes the latent useful.
2. Compare against the baseline.
3. Export one artifact a human can inspect.
4. Tighten the failure cases before adding interface polish.

## Phase 4 — demo surface
1. Build a narrow happy-path demo.
2. Make outputs screenshot/video friendly.
3. Eliminate hidden manual steps.
4. Add one deterministic example flow in the repository.

## Phase 5 — evaluation and packaging
1. Run the chosen metrics on the demo subset.
2. Compare against baseline clearly.
3. Write a short results memo.
4. Decide go / no-go for deeper implementation.

## Project-specific caution

The project must stay anchored to this objective:

> JEPA-style semantic refactoring for Rust code

If implementation drifts away from that objective, treat it as scope failure rather than creativity.
