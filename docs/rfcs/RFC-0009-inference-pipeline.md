# RFC-0009 — Inference pipeline and serving

## Status
Locked (2026-05-15)

## Problem

Specify how a trained CodingJEPA checkpoint is used at inference: encoding, prediction, retrieval, reranking, refusal, and serving.

## Decisions locked

### D1 — Inference modes
Three supported modes:
1. **Embed.** Take a chunk → produce a `(D,)` projected embedding. Used for indexing.
2. **Retrieve.** Take `(chunk_before, intent) → top-k candidate chunks` from a precomputed pool.
3. **Score.** Take `(chunk_before, intent, candidate_chunk_after) → score ∈ [0, 1]`. Used for offline reranking and eval.

There is no "generate" mode in v1.

### D2 — Embedding pipeline
For one chunk:
1. Parse with `libcst`; verify it `compile()`-checks under Python 3.12.
2. Normalize: strip trailing whitespace, normalize line endings, drop comments (per RFC-0012 D7).
3. Tokenize with the committed BPE tokenizer.
4. Truncate to 512 tokens (longer chunks return `None`, surfaced as refusal).
5. Pass through `encoder` → CLS hidden state → `projector` → L2-normalize.

### D3 — Index build
- **Format:** FAISS `IndexFlatIP` (exact inner-product over L2-normalized vectors = cosine).
- **Source:** all `chunk_after`s in the relevant split (test pool for eval; user-provided pool for demo).
- **Sidecar:** `index.meta.parquet` with `chunk_id`, `provenance`, `acceptance_meta` (precomputed per-intent acceptance booleans).
- **Index versioning:** keyed by `(model_checkpoint_hash, data_manifest_hash)`. A new model or data snapshot creates a new index; we never silently mix versions.

### D4 — Retrieve mode

Given `(chunk_before, intent)`:
1. Embed `chunk_before` (D2).
2. Compute `pred_emb = pred_proj(predictor(projector(emb_before).unsqueeze(1).expand(-1, H, -1), act_emb))[:, -1]`.
   - The "expand" repeats the single source embedding `H` times to fill the predictor history; this is the v1 simplification because we do not have a multi-chunk source history at inference. (A multi-chunk source history is a v2 extension.)
3. L2-normalize `pred_emb`.
4. FAISS top-`M = 100` over `pred_emb`.

### D5 — Rerank mode

The retrieved top-`M`:
1. Filter through the intent acceptance rule (RFC-0004 §D2). Candidates that fail are kept but marked `rejected_by_acceptance`.
2. Filter through safety checkers (RFC-0007 §D1). Candidates that fail are removed entirely.
3. Score candidates by cosine similarity to `pred_emb`.
4. Optionally rerank further with a "diversity" pass (MMR with λ=0.5) so the top-`k` aren't near-duplicates of one another; off by default.
5. Compute calibrated confidence: softmax over the top-`k` cosine distances at temperature `τ = 0.1`.

### D6 — Refusal
Per RFC-0007 §D2: if no candidate passes acceptance, or if top-1 confidence `< τ_refuse = 0.55`, return a refusal with a stable message string.

### D7 — Determinism
- All inference is deterministic for a given `(checkpoint, index, input)`.
- `torch.use_deterministic_algorithms(True)` is set at startup. (We accept a small perf hit for reproducibility.)
- Floating-point order is fixed by the FAISS exact index; we do not use approximate indexes in v1.

### D8 — Serving (web)
- FastAPI app `codingjepa.web.app`. Single host, no auth in v1.
- Endpoints:
  - `GET /` — the form.
  - `POST /refactor` — body: `{source, intent, k}`. Returns an HTMX fragment.
  - `GET /healthz` — returns `200` with checkpoint and index hashes.
- Worker model: a single GPU worker process; the index lives in RAM (≈ 200 MB for the test pool).

### D9 — Serving (CLI)
- `python -m codingjepa refactor …` (see RFC-0006 §D2).
- The CLI loads the model + index once and runs to exit; no daemon mode in v1.

### D10 — Latency targets
- P50 < 400 ms, P95 < 1.5 s on H100 for a 512-token chunk and a 100k-chunk index.
- Profiled in `tests/perf/test_latency.py`. CI regression threshold: +20% on P95 fails the test.

### D11 — Memory budgets
- Encoder + predictor + projector on H100: ~250 MB FP16 weights + ~1 GB activation memory at B=1.
- Index for v1 corpus (≈ 50k target chunks × 512 floats × 4 bytes): ≈ 100 MB.
- Total resident set ≤ 4 GB at inference time.

### D12 — Multi-tenant / batching
Not in v1. Requests are processed serially.

### D13 — Logging at inference
Every demo request logs to `.runs/demo-log.jsonl` per RFC-0007 §D5. The log is not exposed via the web endpoint.

## Deferred items
- Approximate indexes (`IndexHNSWFlat`, `IndexIVFFlat`) for larger pools.
- Multi-chunk source history at inference (matching the trained `H=3`).
- Streaming / batched serving.
- Authentication and rate limiting.
- TorchScript/ONNX export.

## Acceptance condition

Locked when:
- a saved checkpoint + index round-trips through `embed → retrieve → rerank` on a held-out chunk, deterministically, in CI;
- the P95 latency budget is met by the perf test;
- the refusal path is exercised by integration tests.
