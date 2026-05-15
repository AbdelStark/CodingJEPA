# 10 — Glossary

Canonical terms used across the spec corpus. Definitions are normative; the rest of the corpus and any RFC must use these terms consistently.

## A

**Acceptance rule.** A boolean predicate per intent that decides whether a `(before, after)` pair is a valid instance of that intent. Single source of truth: `codingjepa.intents.acceptance`. Used at data-mining (RFC-0002 §D6), inference rerank (RFC-0009 §D5), and eval scoring (RFC-0010). See RFC-0004 §D2.

**Action encoder.** The intent-embedder module. Inherits its name from LeWorldModel where the input is a continuous control vector; in CodingJEPA the input is a discrete intent index. See RFC-0003 §D5.

**Ablation.** A run with one design choice swapped out — `λ=0`, no intent conditioning, no projector, varied history `H`, varied encoder size. The ablation matrix is locked in RFC-0005 §D6.

**Audit (per repo).** A JSON record at `data/audit/<repo>.json` capturing license, chunk count, drop rate, dedup rate, and secret-scanner hits. Gates: `compile_ok_rate ≥ 0.95`, `duplication_rate < 0.30`, `secret_scanner_hits == 0`. RFC-0014 §D10.

## B

**Baseline.** One of the three reference systems (B1 BM25, B2 MLM-encoder, B3 frozen CodeBERT) that CodingJEPA must compare against. Optional B4: CodeT5+. RFC-0005 §D5. The cheap-baseline-first rule (§D9) gates Phase 1 → Phase 2.

**Benchmark.** A named, scored evaluation in the harness. Each benchmark has a `prepare`, `run`, `score` method and produces a per-benchmark JSON aggregated into `results.json`. RFC-0010 §D8. The full suite's IDs (CJ-RET-100, CJ-RET-1k, CJ-INTENT, CJ-EXEC, CJ-ROB-FMT, CJ-ROB-RENAME, CJ-ROB-DOC, CJ-OOD, CJ-PROBE-NAME, CJ-PROBE-DEFECT, CJ-PROBE-CLONE, CJ-HUMAN) are listed in RFC-0010 §D3.

## C

**Candidate.** A single retrieved chunk paired with its score, confidence, acceptance verdict, and provenance. See `Candidate` in `docs/spec/02-public-api.md`.

**Checkpoint.** A `.ckpt` file containing the model weights (and optionally the full training-state object). Pinned by `checkpoint_hash`. RFC-0008 §D13.

**Chunk.** A contiguous AST unit from a Python file. One of: `FunctionDef`, `AsyncFunctionDef`, `ClassDef`, or interstitial module-level statement block. Capped at 512 BPE tokens after normalization. See RFC-0012 §D1, §D2.

**Chunk-id.** Stable identifier `sha256(repo + commit_sha + file_path + node_qualname + source_normalized)[:16]`. Persisted in `chunks.parquet`.

**CJ-* benchmark IDs.** The closed set of benchmark identifiers in RFC-0010 §D3.

**CodingJEPA.** This project. Renamed from RustJEPA on 2026-05-15.

**Confidence.** A heuristic value in [0, 1] computed as `softmax(top_k_cosine / τ)` with `τ = 0.1`. Not a probabilistic guarantee. Displayed alongside cosine in the demo. RFC-0007 §D3, RFC-0009 §D5.

**Content addressing.** Identifying an artifact by the hash of its content (e.g., `manifest_hash`, `pool_hash`). Forbids silent overwrites; bumps re-version the artifact.

**Context window.** The `H` chunk embeddings the predictor consumes. Default `H = 3`. See RFC-0003 §D3, §D7.

**Cross-split anti-leakage.** Procedure that ensures no chunk's near-duplicates appear in two splits. RFC-0014 §D6. The audit `data/audit/cross_split_leakage.json` must report zero crossings.

## D

**Dataclass-migration intent.** An intent whose acceptance rule converts a class with hand-written `__init__` to `@dataclass`. RFC-0004 §D2.

**Dedup.** Two-stage procedure: exact `sha256` of normalized chunk source, then MinHash LSH with Jaccard threshold 0.85 over 5-gram BPE shingles. RFC-0014 §D6.

**Determinism.** A property of pipeline stages: same inputs → same outputs. Inference is deterministic via `torch.use_deterministic_algorithms(True)`; training is non-deterministic for performance and we report ≥ 3 seeds. RFC-0009 §D7, RFC-0013 §D5.

**Drop reason.** One of `{over_cap, parse_failed, secret_hit, near_dup, boilerplate}`. Recorded in the chunk row. See `docs/spec/03-data-model.md`.

## E

**Effective rank.** `exp(H(σ))` where `σ` is the normalized singular-value spectrum of a sample of embeddings. The collapse gate requires effective rank ≥ `0.9 × embed_dim`. RFC-0008 §D7.

**Embed dim (`embed_dim`).** Dimensionality of the projected embedding space. Default 512.

**Embed mode.** Inference mode that turns a chunk into a `(D,)` projected, L2-normalized vector. RFC-0009 §D1.

**Eval pool.** The set of candidate chunks a retrieval benchmark scores against. Recorded in `eval/pools/<benchmark>.lock.json`. Sizes N ∈ {100, 1000} for CJ-RET-100/1k.

**Execution-preservation.** A check that runs the original test suite against the file with the top-1 retrieved candidate substituted in. Pass-rate is the metric. RFC-0010 §D2 / E3.

**Extract-helper intent.** Intent 0 in the vocabulary. Acceptance: ≥ 2 contiguous statements move into a new top-level `FunctionDef` and a `Call` replaces the extracted block. RFC-0004 §D2.

## F

**FAISS index.** `IndexFlatIP` over L2-normalized embeddings. Sidecar Parquet carries `chunk_id`, position, per-intent acceptance metadata, provenance. RFC-0009 §D3.

**Fine-tune (Stage B).** The intent-conditioned training stage on labeled refactor pairs. RFC-0008 §D2.

**Formatting invariance.** A robustness probe: apply `black` + `isort` + random whitespace; embedding cosine drift must be < 0.02; rank change < 5%. RFC-0005 §D3.

## G

**Gold subset.** 200 hand-curated refactor pairs with κ ≥ 0.7 inter-rater agreement. Stratified 25 per intent. Used for human review and the diff gallery. RFC-0002 §D7.

## H

**Heuristic labeler.** A conservative function that takes `(before_cst, after_cst)` and returns `(matched, confidence)` for one of the 8 intents. Implemented per intent in `codingjepa/data/labelers/`. RFC-0002 §D6.

**History (`H`).** Number of context chunks the predictor sees. Default 3.

## I

**Index id.** `f"{checkpoint_hash[:8]}-{manifest_hash[:8]}"`. Identifies a FAISS index uniquely. RFC-0009 §D3.

**Intent.** One of the 8 closed refactor classes plus `NONE`. Drives the action embedding at training and inference. RFC-0004.

**Intent-conditioned hit rate.** R@5 with vs. without intent conditioning. RFC-0010 CJ-INTENT.

**Intent embedder.** `nn.Embedding(9, embed_dim)`. Index 0–7 are the intents; index 8 is `[I_NONE]`. RFC-0003 §D5.

## J

**JEPA.** Joint-Embedding Predictive Architecture. The class of models that predict in embedding space rather than token space.

## L

**LeWorldModel (LeWM).** The architectural and training-recipe substrate (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026). End-to-end JEPA without EMA, two-term loss (MSE + SIGReg). See RESEARCH.md and RFC-0008 §D1.

**Linear probe.** A frozen-encoder evaluation: train only a linear head on a downstream task. Used for function-name prediction, defect detection, clone detection. RFC-0010 E6.

**Loss.** `loss = pred_loss + λ · sigreg_loss` where `pred_loss = MSE(pred_emb, tgt_emb.detach())` and `sigreg_loss = SIGReg(emb)`. RFC-0008 §D3.

## M

**Manifest.** `data/manifest.lock.json`. Records pinned commits, chunker version, tokenizer hash, splits path, per-repo audit. Content-addressed by `manifest_hash`. RFC-0014 §D10.

**MinHash LSH.** Locality-sensitive hashing for near-duplicate detection. Jaccard threshold 0.85 over 5-gram BPE shingles. RFC-0014 §D6.

**Mixed precision.** `bf16` autocast at training; FP16 weights at inference. RFC-0008 §D4.

**MRR.** Mean Reciprocal Rank. Secondary metric. RFC-0005 §D2.

## N

**`[I_NONE]`.** The "no intent" embedding (index 8 in the intent embedder). Used during pretraining and as the refusal label. RFC-0003 §D5.

**Normalization.** The chunker-side preprocessing: black, isort, docstring sentinel, pragma stripping, line-ending and whitespace cleanup. RFC-0012 §D5.

**Nuisance variable.** A surface feature the latent should be invariant under (formatting, identifier rename, comment edits, import order). RFC-0010 robustness probes test invariance.

## O

**OOD probe.** 200 hand-curated pairs from cpython `Lib/` 3.11 → 3.13 refactors. Held out from all training. RFC-0014 §D8.

## P

**Pair (refactor pair).** `(chunk_before, chunk_after, intent_label, provenance)`. RFC-0002 §D10.

**`pred_proj`.** A projector mirror applied after the predictor. Same shape as the projector. RFC-0003 §D4.

**Predictor.** `ARPredictor`. 4-layer transformer over chunk embeddings + action embeddings. Outputs `(B, n_preds, D)`. RFC-0003 §D3.

**Pretraining (Stage A).** Unconditional next-chunk-embedding prediction on chunk sequences. Intent always `[I_NONE]`. RFC-0008 §D2.

**Projector.** MLP mapping encoder CLS → `embed_dim`. Same projector for context and target. No separate target projector; no EMA. RFC-0003 §D2.

**Provenance.** `(repo, commit_sha, file_path, node_qualname)`. Carried with every chunk, pair, embedding, retrieval result.

## R

**RAG / retrieval-and-rerank.** The v1 decode path. Predict latent, retrieve top-`M`, filter by intent acceptance and safety checkers, rerank by cosine + confidence. RFC-0009.

**Refusal.** A first-class outcome. The system declined to recommend; not an error. Stable code from `codingjepa/safety/messages.py`. See `docs/spec/04-error-model.md`.

**Retrieval@k.** Hit rate of the true `chunk_after` in the top-`k` retrieved candidates. `k ∈ {1, 5, 10}`, pool size `N ∈ {100, 1000}`. Primary metric. RFC-0005 §D1.

**Rerank.** The post-retrieval step that filters and reorders the top-`M` candidates. RFC-0009 §D5.

**RoPE.** Rotary positional encoding. Used by the encoder; no learned absolute positions. RFC-0003 §D1.

## S

**Safety checker.** A per-class static checker that filters candidates: side-effect introduction/elimination, exception-contract change, public-API change, async/sync boundary change. RFC-0007 §D1.

**Schema version.** A field on every persisted artifact. Major bump re-versions the artifact directory; minor bumps are backwards-compatible additions. See `docs/spec/03-data-model.md`.

**Sequence (training).** Sliding window of `S = H + n_preds + 1` chunks within a single file. RFC-0002 §D9.

**SIGReg.** Sliced Isotropic Gaussian Regularizer. Penalizes deviation from `N(0, 1/d)` along K random unit directions. Default `K = 256`, `λ = 0.05`. RFC-0003 §D6.

**Source repos.** The 10 curated Python repositories at pinned commits. `docs/data/CANDIDATE_REPOS.md`.

**Special tokens.** `[PAD]` (0), `[UNK]` (1), `[CLS]` (2), `[SEP]` (3), `[CHUNK]` (4), `<DOC>` (5), `[I_0]…[I_7]` (6–13), `[I_NONE]` (14). RFC-0012 §D7.

**Splits.** Train (6 repos) / val (2 repos) / test (2 repos) / OOD (200 cpython pairs). Frozen in `data/splits/v1.lock.json`. RFC-0014 §D7.

**Stop-gradient.** `tgt_emb.detach()` in the prediction loss. The target encoder is the same encoder; we forbid gradient flow into the target through the loss. RFC-0008 §D3.

## T

**`τ_refuse`.** Cosine-similarity threshold below which the system refuses. `τ_refuse = 0.55`. RFC-0007 §D2.

**`τ`.** Softmax temperature for confidence calibration. `τ = 0.1`. RFC-0007 §D3, RFC-0009 §D5.

**Tokenizer.** SentencePiece BPE, vocab 32k, pinned by content hash. Trained on the 6 train repos only. RFC-0012 §D7.

## U

**Unconditional pretraining.** Stage A. See "Pretraining."

## V

**Versioning.** SemVer for the package; per-schema versioning for artifacts. See `docs/spec/09-release-and-versioning.md`.

## W

**WandB.** Weights & Biases. Training logger. Project `codingjepa-v1`. Offline mode acceptable. RFC-0008 §D12.

---

Terms not defined here may be used loosely in conversational notes (`docs/notes/PHASE-N.md`), but specifications and RFCs use only the definitions above. Adding or amending a term requires a PR that touches this file.
