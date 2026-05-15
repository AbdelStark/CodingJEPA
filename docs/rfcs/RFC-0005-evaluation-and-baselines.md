# RFC-0005 — Evaluation and baselines

## Status
Locked (2026-05-15)

## Problem

Define what we measure, how we measure it, and which baselines we are obliged to beat before claiming the thesis.

This RFC is *companion* to RFC-0010, which specifies the harness mechanics. RFC-0005 specifies *what counts as a valid measurement*.

## Decisions locked

### D1 — Primary metric
**Retrieval@k of `chunk_after`** given `(chunk_before, intent)` from a pool of N candidates.

- Reported at `k ∈ {1, 5, 10}` and `N ∈ {100, 1000}` simultaneously.
- The pool is constructed per-test-set: all `chunk_after`s in the test split, optionally augmented with `M − |test|` distractors sampled from the same split.
- Cosine similarity in the L2-normalized projected embedding space.

### D2 — Secondary metrics
- **MRR (Mean Reciprocal Rank)** of `chunk_after` in the test pool.
- **Intent-conditioned hit rate** vs. unconditional baseline (proves that intent conditioning is doing work).
- **Per-intent breakdown** (macro-average, then per-intent table).
- **Execution-preservation pass rate** on the 500-pair executable subset.
- **Linear-probe scores** for: function-name prediction, defect detection (Devign Python subset), clone detection.

### D3 — Robustness probes (mandatory)
- **B-1 Formatting invariance.** Apply `black`, `isort`, random PEP-8-preserving whitespace edits, and random comment perturbations. Measure cosine drift between original and perturbed embedding, and rank change of the correct target.
  - Pass bar: rank change < 5%, cosine drift < 0.02.
- **B-2 Alpha-rename invariance.** Randomly rename free local variables in `chunk_before` preserving resolution. Same metrics as B-1.
  - Pass bar: rank change < 10%, cosine drift < 0.05.
- **B-3 Docstring perturbation.** Replace docstrings with random sentences of the same length. Same metrics.
  - Pass bar: rank change < 5%, cosine drift < 0.02.

### D4 — Execution-preservation check
- A subset of ~500 refactor pairs whose `chunk_after` is associated with an executable unit test (the chunk lives in a module covered by a `pytest` suite in the source repo).
- For each, retrieve top-1 candidate and substitute it into the file. Run the relevant test subset under a sandbox (subprocess, 30s timeout, no network, read-only FS except `/tmp`).
- Report pass-rate. PRD acceptance bar: ≥ 70%.

### D5 — Baseline systems (3, each from a different family)
- **B1 — BM25 over BPE tokens.** Lexical baseline using `rank_bm25`. No training.
- **B2 — MLM-encoder.** Same architecture as CodingJEPA encoder (6 layers, hidden 512, ~30M params), trained with masked-LM (15% mask, BERT-style) on the same corpus and compute budget. Compares JEPA loss to MLM loss at matched scale.
- **B3 — Frozen CodeBERT.** `microsoft/codebert-base`. Used as a published representation baseline.

Optional but encouraged:
- **B4 — CodeT5+ embeddings** (110M variant).

All baselines use the same tokenization at retrieval-time and the same FAISS index format.

### D6 — Ablation matrix (CodingJEPA internal)
For the ablation table in the paper:
1. Full CodingJEPA.
2. — no SIGReg (`λ = 0`).
3. — no intent conditioning (always `[I_NONE]`).
4. — no projector / pred_proj (identity).
5. — predictor history `H ∈ {1, 3, 6}`.
6. — encoder size `{small=18M, base=30M, large=70M}` (only if compute permits).
7. — λ ∈ `{0.005, 0.05, 0.5}`.

Each ablation is run with ≥ 3 seeds. We report mean ± std.

### D7 — Statistical reporting rules
- ≥ 3 seeds for every reported number.
- Confidence intervals via 1000-resample bootstrap; report 95% CI.
- Significance tests: paired bootstrap on per-example retrieval rank (CodingJEPA vs. best baseline). `p < 0.05` required for "beats" claims.

### D8 — Human review rubric (top-k candidate inspection)
For the gold subset (200 pairs) we record human ratings of the top-1 retrieved candidate on three Likert-1-to-5 scales:
- **Semantic preservation** (does it do the same thing as `chunk_after`?).
- **Stylistic plausibility** (would a reviewer accept it?).
- **Intent fidelity** (does it actually match the selected intent?).

≥ 2 raters per pair, Cohen's κ reported.

### D9 — Cheap-baseline-first rule
**No CodingJEPA training run is launched until all three baselines (B1–B3) produce a metrics JSON on the gold subset.** This rule is enforced at the Phase 1 → Phase 2 gate.

## Deferred items
- Comparison against frontier (StarCoder, CodeLlama) generative baselines — orthogonal to our claim about matched-compute representation learning.
- Cross-corpus generalization (train on this 10-repo set, test on a different repo) — v2.

## Acceptance condition

Locked when:
- the three baselines have a documented run command and committed metric JSON;
- the eval harness produces a single `results.json` covering D1–D6 in one invocation;
- the robustness probes have unit tests that confirm the perturbations preserve resolution (we don't want a buggy perturber to dominate the result).
