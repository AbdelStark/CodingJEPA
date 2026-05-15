# CodingJEPA — system spec

## 1. Architecture summary

The system is built in seven layers. Each layer has a documented input/output contract; downstream layers must not reach upstream past the contract.

1. **Raw data ingestion** — mirror, filter, and license-check the 10 source repositories (see `docs/data/CANDIDATE_REPOS.md`).
2. **Sample/window construction** — parse Python via `ast` + `libcst`, chunk into semantic units (function, class, top-level block), tokenize, build sequences of chunks per file and per refactor pair.
3. **JEPA encoder + predictor training** — two-stage training: (a) unconditional next-chunk-latent prediction; (b) intent-conditioned latent prediction on refactor pairs. Loss = `MSE(pred_emb, target_emb) + λ · SIGReg(emb)`.
4. **Downstream decode path** — latent retrieval-and-rerank against a precomputed pool of candidate target chunks indexed by FAISS.
5. **Demo-facing application surface** — CLI and a thin FastAPI/HTMX UI.
6. **Evaluation and artifact export** — deterministic eval runner producing JSON metrics, diff galleries, and a written results memo.
7. **Reproducibility and infra** — Hydra configs, deterministic seeds, manifest-pinned data snapshots, single-command `make eval`.

## 2. System constraints

- **Single-machine friendly.** v1 training fits a single 80GB GPU. Distributed training is allowed but never required.
- **Bounded training budget.** Pretraining ≤ 7 days on 1×H100 or equivalent. Fine-tune ≤ 24 hours.
- **Explicit baselines before model work.** No CodingJEPA training run is launched until the three baselines run end-to-end and produce metric JSON.
- **Deterministic enough to reproduce demo artifacts.** All RNG seeded; data snapshots are content-addressed.
- **Narrow enough to evaluate honestly.** Eight intents, ten repos, frozen splits.

## 3. Data contract

### 3.1 Definitions

- **Chunk:** a contiguous AST unit (`FunctionDef`, `AsyncFunctionDef`, `ClassDef`, module-level statement block) up to a token budget. See RFC-0012.
- **Sequence:** an ordered list of chunks within a file, or across files in a topologically-sorted import graph, capped at `S` chunks.
- **Refactor pair:** a `(chunk_before, chunk_after, intent_label, provenance)` tuple where `chunk_before` and `chunk_after` are chunks from the same file at adjacent commits, and the diff matches one of the 8 intents per RFC-0002 heuristics.

### 3.2 Splits

- **Train / val / test splits are by repository × time.**
  - 6 repos → train.
  - 2 repos → val.
  - 2 repos → test.
- A separate **OOD probe set** holds out 200 hand-curated pairs from `python/cpython` standard-library refactors.
- Splits are frozen in `data/splits/v1.lock.json` and content-addressed. Any new split is `v2`, never an overwrite.

### 3.3 Nuisance variables

Suppressed (the latent should be invariant under these):
- whitespace and indentation style (within PEP 8);
- comment content and docstring wording;
- alpha-renames of locals that preserve resolution;
- import ordering (where it does not affect resolution).

Preserved (the latent must encode these):
- API shape (parameter names that are part of the public signature, return type);
- control flow structure;
- side-effect surface (mutation, IO, exceptions raised);
- the set of names resolved at module scope.

### 3.4 Anti-leakage

- Near-duplicate detection via MinHash LSH on token n-grams before splitting.
- Cross-split fork detection: any file whose content hash or whose import-graph neighborhood overlaps ≥ θ with files in another split is pinned to one split.
- The 8-intent taxonomy is balanced across splits; any intent with < 100 pairs in test is dropped from the primary metric and reported as supplementary.

## 4. Model contract

### 4.1 Encoder

- Backbone: 6-layer Transformer encoder over BPE tokens, hidden 512, 8 heads, FFN 2048, dropout 0.1.
- Output: per-chunk embedding via a learned `[CLS]` token (analogous to LeWM's CLS extraction in `JEPA.encode`).
- Parameter budget: ≈ 30M.
- Tokenizer: BPE trained on the 10-repo Python corpus (vocab 32k); special tokens `[CLS]`, `[SEP]`, `[PAD]`, `[CHUNK]`, intent tokens `[I_*]`.

### 4.2 Predictor

- Autoregressive predictor (`ARPredictor` analog): 4 layers, hidden 512, 8 heads, cross-attends over a history of `H = 3` chunk embeddings; conditions on an intent embedding when present.
- Outputs a single predicted embedding per step, projected through `pred_proj` (MLP with BatchNorm1d, hidden 2048).
- Parameter budget: ≈ 10–15M.

### 4.3 Intent encoder ("action encoder")

- For unconditional pretraining: an all-zeros intent vector (or `[I_NONE]` token).
- For intent-conditioned fine-tuning: a small `Embedder` (one-hot intent index → embedding) with the same output dim as `projector`.
- Eight intent slots + one `NONE` slot.

### 4.4 Projector

- `MLP(hidden=2048, norm=BatchNorm1d)` mapping encoder CLS → embed dim (same as LeWM).
- Identical projector applied to both context embeddings and prediction targets (no separate target projector, no EMA).

### 4.5 Latent dimensions

- `embed_dim = 512` (default). Configurable through Hydra.
- Predictor history `H = 3`. Number of predicted steps `n_preds = 1` for v1.

### 4.6 Training objective

`L = pred_loss + λ · sigreg_loss`

- `pred_loss = MSE(pred_emb, target_emb.detach())`
  - For pretraining: `target_emb` is the next chunk's encoder embedding within the same file/sequence.
  - For fine-tuning: `target_emb` is the encoder embedding of `chunk_after` from a refactor pair.
- `sigreg_loss = SIGReg(emb)` — sliced isotropic Gaussian regularizer; encourages embedding distribution to be approximately N(0, I/d) along random projections. Single hyperparameter `λ`.
- **No EMA target encoder.** The target encoder is the same encoder, with stop-gradient on the target embedding.
- **No teacher–student decoupling.** Single network, single projector.

### 4.7 Embedding export and downstream interface

- The encoder + projector are the export surface. They produce a deterministic `(D,)` embedding for any input chunk.
- A FAISS `IndexFlatIP` over L2-normalized embeddings is the inference-time retrieval primitive.

## 5. Evaluation contract

The eval harness must expose, on a deterministic test split:

### 5.1 Primary task metric

- **Retrieval@k (k ∈ {1, 5, 10})** of the true `chunk_after` from a pool of N candidates (N ∈ {100, 1000}), given `chunk_before` and intent.

### 5.2 Robustness probes (mandatory)

- **Formatting-invariance:** apply `black`, `isort`, and random whitespace perturbations; measure embedding cosine drift.
- **Alpha-rename invariance:** randomly rename free locals (preserving resolution); measure rank change of correct target.

### 5.3 Execution-preservation check

- For chunks that come with executable unit tests (subset, ~500 pairs), the top-1 retrieved target chunk must produce passing tests with the same suite as `chunk_after`. Report pass-rate.

### 5.4 Baselines (mandatory)

- **B1 — BM25** over BPE tokens (cheap lexical).
- **B2 — MLM-encoder** of the same architecture as CodingJEPA's encoder, trained with masked-LM (no JEPA loss); same compute budget.
- **B3 — Frozen CodeBERT** retrieval (`microsoft/codebert-base`).

### 5.5 Human-inspectable artifact

- A `diffs/` gallery with top-1 retrievals per intent, side-by-side, exported to HTML.
- A `confusions/` page with the worst 50 errors per intent for manual inspection.

## 6. Demo contract

The demo must:
- Show the thesis in one interaction (`python -m codingjepa demo`).
- Take a Python snippet + intent; return ranked candidates with latent distance and confidence.
- Produce screenshot/video-friendly outputs (HTML diff cards, JSON traces).
- Surface failure cases ("no candidate above confidence threshold T"), never silently fall back to top-1.
- Avoid hidden manual data cleanup in the live path: the demo runs against the same artifacts the eval harness uses.

## 7. Reproducibility contract

- **Configs:** all training/eval runs are launched from Hydra YAML; the resolved config is dumped to the run directory.
- **Seeds:** `seed` is set for Python `random`, `numpy`, `torch`, and `torch.cuda`. Dataloader workers are seeded deterministically.
- **Data manifest:** `data/manifest.lock.json` records commit hashes of source repos and content hashes of all parsed chunks.
- **`make eval`:** a single command reproduces all reported metrics from the released checkpoint and data snapshot.

## 8. Implementation principle

Build the narrowest meaningful path first. The first version must prove the thesis, not become a platform. Any feature that does not move a metric on the frozen eval contract is deferred to v2.
