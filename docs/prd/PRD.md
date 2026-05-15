# CodingJEPA — PRD

## 1. Product thesis

A JEPA trained on Python code with a next-embedding prediction loss across code chunks and a sliced Gaussian regularizer over the embedding distribution learns semantic representations that transfer to narrow refactoring and code understanding tasks better than token-level baselines of comparable scale.

## 2. Product goal

Build a compact, technically credible v1 that demonstrates the thesis on Python code, with a working demo, an evaluation surface, and a written results memo — without requiring large-lab training budgets or broad platform scope. The training pipeline must inherit from the LeWorldModel recipe and be reproducible on a single 80GB GPU.

## 3. Problem

Existing code assistants treat refactoring as token-level next-token generation or retrieval with weak semantic grounding. For Python, small surface edits can hide large semantic changes (mutable defaults, name resolution, descriptor protocol), while semantically equivalent refactors can look very different at the token level (comprehensions vs. loops, `map`/`filter` vs. explicit iteration, dataclass vs. `__init__`). A latent-space objective — predicting the embedding of the next or refactored chunk rather than the tokens themselves — is a more natural fit for this asymmetry and can be tested credibly at small scale.

## 4. v1 scope

- **Data:** mine, dedupe, and filter Python files and refactor pairs from 10 curated open-source repositories.
- **Intent taxonomy:** 8 narrow refactor intents with deterministic acceptance rules.
- **Model:** compact ViT-style code encoder (~30M params) + autoregressive predictor (~10–20M params) + projector heads.
- **Training:** two-stage — unconditional next-chunk-embedding pretraining, then intent-conditioned fine-tuning on refactor pairs.
- **Loss:** LeWM two-term loss — MSE on predicted vs. target embeddings + SIGReg (sliced isotropic Gaussian regularizer).
- **Inference:** retrieval-and-rerank decode path. Given source + intent, predict target latent, score candidate target chunks from a precomputed pool.
- **Demo:** CLI + small web UI; takes Python input + intent, returns ranked candidates with diff inspection.
- **Evaluation:** primary metric (latent retrieval accuracy on held-out pairs) + at least one syntactic-nuisance robustness probe + at least one execution-preservation check.

## 5. Explicit non-goals

- No general-purpose code generation system.
- No multi-language support in v1 (Python only).
- No claim of replacing compiler, linter, type checker, or full IDE refactoring workflows.
- No from-scratch decoder-only LM training; we use small decoder-free retrieval/rerank.
- No claim of beating large frontier models on absolute task performance; the claim is about *parameter-matched and FLOP-matched* baselines.

## 6. Users

Primary users:
- researchers and engineers evaluating whether the JEPA framing actually improves the target problem at small scale;
- developers/operators who need an inspectable demo and a written results memo rather than a vague claim.

Secondary users:
- open-source contributors joining after the initial prototype;
- technical readers who want the core design decisions spelled out before implementation grows.

## 7. Demo requirement

The project must support a short demo flow:
1. Paste a Python snippet (function or class up to ~400 tokens).
2. Select an intent from the 8-intent taxonomy.
3. The system returns the top-k ranked candidate target snippets with a diff view, latent distance, and per-intent confidence score.
4. The user can mark any candidate as accepted/rejected; rejections are logged.

Failure cases must be visibly surfaced (e.g., "no candidate met the confidence threshold for intent X"), never silently hidden.

## 8. Success criteria

Quantitative:
- **Retrieval@10 (latent target chunk recall):** ≥ 1.5× the strongest cheap baseline (BM25 over tokens, BPE token MLM, or CodeBERT embeddings).
- **Intent-conditioned hit rate:** ≥ 2× over an unconditional latent baseline (proves the intent conditioning matters).
- **Execution-preservation rate** on the curated benchmark subset (snippets with executable unit tests): ≥ 70%.
- **Nuisance robustness:** retrieval ranks must change by < 5% under formatting-only edits (whitespace, comments, identifier alpha-renames that preserve resolution).

Qualitative:
- Human reviewers (n ≥ 3) rate top-1 latent retrieval as semantically preserving more often than baselines on a held-out gold set of 100 pairs.
- A short written results memo (≤ 8 pages) is reproducible from the artifacts in the repo.

## 9. Main risks

- **Weak task definition creates a fake win.** Mitigated by RFC-0001 locking the task family.
- **Model scope expands faster than data/eval contract.** Mitigated by RFC-0002 / RFC-0010 freezing benchmarks first.
- **Demo polish arrives before claim is supported.** Mitigated by RFC-0006 requiring the eval report before the UI.
- **Baselines too weak make the result look better than it is.** Mitigated by RFC-0005 requiring three baselines of distinct families (lexical, MLM, contrastive).
- **Data leakage from large repos into eval.** Mitigated by RFC-0014 deduplication and held-out repo splits.

## 10. Deliverables for MVP

- **D1:** Reproducible data/task contract: scripts that download, filter, dedupe, chunk, and intent-label the 10 source repos.
- **D2:** Baseline systems (3): BM25, MLM-encoder, CodeBERT-frozen.
- **D3:** CodingJEPA model checkpoint: encoder + predictor + projector + sigreg.
- **D4:** Retrieval-and-rerank inference path with deterministic seeding.
- **D5:** Demo surface (CLI + minimal web UI).
- **D6:** Evaluation harness, benchmark splits, and a written results memo with all numbers reproducible from a single `make eval` command.
- **D7:** Paper draft (workshop-grade, ≤ 8 pages) following the structure in RFC-0011.
