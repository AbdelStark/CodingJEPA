# CodingJEPA — implementation plan

## Goal

Build the smallest credible v1 that proves the thesis:

> A JEPA trained on Python code with a next-embedding prediction loss across code chunks and a sliced Gaussian regularizer over the embedding distribution learns semantic representations that transfer to narrow refactoring and code understanding tasks better than parameter- and FLOP-matched token-level baselines.

Implementation is **phased**. Each phase ends with a frozen artifact and a one-page note. No phase begins until the prior phase's artifact exists in the repository.

---

## Phase 0 — lock the design (week 0)

**Inputs:** none.
**Outputs:** the spec stack in this repository.

1. Read `docs/prd/PRD.md`.
2. Read `docs/spec/SYSTEM-SPEC.md`.
3. Read all RFCs 0001–0014 in order.
4. Update `docs/spec/MVP-STATUS.md` with any corrected assumptions.
5. Refuse implementation that expands scope before the v1 contracts are explicit.

**Gate to phase 1:** every RFC marked `Status: Locked`.

---

## Phase 1 — data and task definition (weeks 1–2)

**Inputs:** RFC-0002, RFC-0012, RFC-0014, `docs/data/CANDIDATE_REPOS.md`.
**Outputs:** `data/raw/`, `data/parsed/`, `data/splits/v1.lock.json`, `data/pairs/v1.parquet`, `data/manifest.lock.json`, `data/gold/v1.jsonl`.

1. Mirror the 10 source repositories at pinned commit hashes.
2. Walk commit histories with PyDriller; extract candidate `(before, after)` chunk pairs.
3. Run the 8-intent heuristic labelers; produce labeled refactor pairs.
4. Parse all files with `libcst`; produce chunks per RFC-0012.
5. Run MinHash LSH deduplication per RFC-0014; resolve cross-split overlaps.
6. Manually inspect and curate a 200-pair **gold subset** for evaluation.
7. Implement the three baselines (BM25, MLM-encoder, frozen CodeBERT) end-to-end on the gold subset before any CodingJEPA training begins.
8. Write `docs/notes/PHASE-1.md` capturing data assumptions and leakage audit.

**Gate to phase 2:** baselines produce a metrics JSON; manifest is content-addressed and committed.

---

## Phase 2 — minimal model path (weeks 3–5)

**Inputs:** Phase 1 artifacts, RFC-0003, RFC-0008, RFC-0013.
**Outputs:** `codingjepa/` package, `runs/pretrain-v1/`, ablation matrix.

1. Lock backbone and parameter counts (encoder ≈ 30M, predictor ≈ 12M).
2. Implement preprocessing, BPE tokenizer training, and chunk batching.
3. Implement encoder, `ARPredictor`, projector, pred_proj, intent embedder, `SIGReg`.
4. Run a tiny-slice training pass (10k steps, 1k pairs) and verify:
   - Loss decreases.
   - Embeddings are not collapsed (rank ≥ 0.9 × embed_dim on a 10k-sample matrix).
   - SIGReg loss decreases monotonically when enabled.
5. Add the **formatting-invariance** robustness check tied to H1.
6. Run full pretraining run per RFC-0008.
7. Write `docs/notes/PHASE-2.md` covering training curves, rank metrics, and any departures from the RFC config.

**Gate to phase 3:** pretraining converged; baselines beat by ≥ 1.2× on linear-probe retrieval.

---

## Phase 3 — downstream task path (weeks 6–7)

**Inputs:** Phase 2 checkpoint, RFC-0004, RFC-0009.
**Outputs:** `runs/finetune-v1/`, FAISS index, `retrieval_metrics.json`.

1. Fine-tune CodingJEPA on intent-labeled refactor pairs (RFC-0008 §3).
2. Build the FAISS retrieval index over the test-pool target chunks.
3. Implement the retrieval-rerank decode path per RFC-0009.
4. Compare against the three baselines on Retrieval@k, intent hit rate, robustness.
5. Export the diff gallery for the gold subset.
6. Tighten failure cases (low-confidence refusal, no-op surfacing).
7. Write `docs/notes/PHASE-3.md` with full metric table and failure analysis.

**Gate to phase 4:** primary metric meets PRD §8 thresholds.

---

## Phase 4 — demo surface (week 8)

**Inputs:** Phase 3 artifacts, RFC-0006.
**Outputs:** `codingjepa/cli.py`, `codingjepa/web/`, demo screenshots/video.

1. Build narrow happy-path CLI (`python -m codingjepa refactor --file x.py --intent extract-helper`).
2. Build a minimal FastAPI + HTMX UI (one route, one form, one diff view).
3. Eliminate hidden manual steps.
4. Add one deterministic example flow committed to `examples/`.
5. Record a screenshot/video walkthrough.

**Gate to phase 5:** demo runs cleanly from a fresh clone via `make demo`.

---

## Phase 5 — evaluation, paper, packaging (weeks 9–10)

**Inputs:** all prior artifacts, RFC-0010, RFC-0011.
**Outputs:** `results/RESULTS-MEMO.md`, paper draft, frozen release.

1. Run the full eval harness end-to-end with `make eval`.
2. Produce the ablation table (loss terms, intent conditioning, sequence history `H`, projector vs. no-projector, SIGReg λ sweep).
3. Compare against all three baselines.
4. Write the results memo per RFC-0010.
5. Draft the paper per RFC-0011.
6. Tag `v1.0`, freeze data manifests, release checkpoints.
7. Decide go/no-go on v2 work (planning, multi-step refactors, multi-language).

**Gate to release:** every number in the paper is reproducible from the released checkpoint via a single command.

---

## Project-specific caution

The project must stay anchored to this objective:

> **JEPA-style semantic refactoring and representation learning for Python code.**

If implementation drifts away from this objective, treat it as scope failure rather than creativity.
