# RFC-0003 — Encoder, predictor, projector stack

## Status
Locked (2026-05-15)

## Problem

Specify the exact architecture of the CodingJEPA model: encoder, predictor, projector, intent embedder, and the optional "decode" surface.

## Context

We inherit the structural skeleton from LeWorldModel — encoder + ARPredictor + action_encoder + projector + pred_proj, single network, no EMA — and adapt the visual ViT into a Python-token transformer.

## Decisions locked

### D1 — Encoder
- **Backbone:** Transformer encoder, pre-norm, 6 layers, hidden 512, 8 heads, FFN 2048, GELU, dropout 0.1, RoPE positional encoding (no learned absolute positions).
- **Input:** BPE token IDs from the CodingJEPA tokenizer (RFC-0012). Max sequence length 512 tokens.
- **Special tokens:** `[CLS]`, `[SEP]`, `[PAD]`, `[CHUNK]`, intent tokens `[I_0]…[I_7]` and `[I_NONE]`.
- **Output:** the hidden state at position `[CLS]` (analogous to LeWM's `output.last_hidden_state[:, 0]`).
- **Parameter count:** ≈ 30M.

### D2 — Projector
- **MLP:** `Linear(hidden_dim=512, 2048) → BatchNorm1d → ReLU → Linear(2048, embed_dim=512)`.
- One shared projector for both context embeddings and target embeddings (same as LeWM).
- No separate target projector.

### D3 — Predictor (`ARPredictor`)
- **Autoregressive transformer over chunk embeddings.**
- 4 layers, hidden 512, 8 heads, FFN 2048.
- Takes `(B, H, D)` context chunk embeddings + `(B, H, D)` action embeddings; emits `(B, n_preds, D)` predicted embeddings.
- Action embeddings are summed onto chunk embeddings position-wise before the predictor (same convention as LeWM's `predictor(emb, act_emb)`).
- Predictor history `H = 3`; number of predicted steps `n_preds = 1` for v1.
- **Parameter count:** ≈ 10–12M.

### D4 — `pred_proj`
- **MLP:** `Linear(hidden_dim=512, 2048) → BatchNorm1d → ReLU → Linear(2048, embed_dim=512)`.
- Same shape as `projector` (identical except the parameters). Applied after the predictor and before the loss.

### D5 — Intent embedder ("action encoder")
- Discrete intent index `∈ {0, …, 7, NONE}` → embedding of size `embed_dim`.
- Implemented as `nn.Embedding(9, embed_dim)`; `[I_NONE]` is index 8.
- During unconditional pretraining all positions use `[I_NONE]`.
- During intent-conditioned fine-tuning the intent is passed for the target position.

### D6 — SIGReg
- Sliced Isotropic Gaussian Regularizer.
- Projects the batch of embeddings onto `K` random unit directions; for each direction, penalizes deviation from `N(0, 1/d)`.
- Default: `K = 256` slices, refreshed each step.
- Single hyperparameter `λ = 0.05` (calibrated in Phase 2; sweep over `{0.005, 0.05, 0.5}`).
- Implementation in `codingjepa/modules/sigreg.py`. Reference: LeWM `module.py:SIGReg`.

### D7 — Forward / loss
Identical structure to LeWM's `lejepa_forward`:
```
emb_all = encoder([CLS] + tokens for each chunk in sequence)   # (B, S, D)
ctx_emb  = emb_all[:, :H]                                       # context
tgt_emb  = emb_all[:, n_preds:]                                 # targets (stop-grad)
pred_emb = pred_proj(ARPredictor(projector(ctx_emb), act_emb))
pred_loss   = MSE(pred_emb, tgt_emb.detach())
sigreg_loss = SIGReg(projector(emb_all))
loss        = pred_loss + λ * sigreg_loss
```

### D8 — Decode path
The v1 "decoder" is **retrieval-and-rerank**, not generative:
1. Encode `chunk_before` + intent. Predict `pred_emb`.
2. Retrieve top-`M` candidates from a FAISS index of pre-encoded chunks (cosine over L2-normalized projections).
3. Rerank by:
   - cosine similarity to `pred_emb`;
   - per-intent acceptance rules from RFC-0004 (boolean filter);
   - confidence = softmax over top-M cosine distances at temperature `τ = 0.1`.

We **do not train a generative decoder** in v1. A generative head is on the v2 roadmap.

### D9 — Parameter budget summary
| Component | Params |
|---|---|
| Encoder | ~30M |
| Projector | ~1.6M |
| ARPredictor | ~10–12M |
| pred_proj | ~1.6M |
| Intent embedder | < 1k |
| **Total** | **~44–46M** |

### D10 — Interfaces
- `codingjepa.model.CodingJEPA(encoder, predictor, action_encoder, projector, pred_proj, sigreg)` — top-level module mirroring LeWM's `JEPA` class.
- `model.encode(chunk_tokens) → emb`
- `model.predict(ctx_emb, act_emb) → pred_emb`
- `model.embed(chunk_tokens) → projected_emb` (used by the retrieval index)
- All forward passes return a dict that includes losses, mirroring LeWM's convention.

## Deferred items
- Generative decoder head.
- Multi-step predictor (`n_preds > 1`).
- Larger encoder (12-layer, hidden 768) — kept for v1.5 scaling experiments.

## Acceptance condition

Locked when:
- a tiny-slice training pass produces a non-collapsed embedding matrix (rank ≥ 0.9 × `embed_dim` over 10k samples);
- the model loads and round-trips a chunk through `encode → predict → embed` with no shape errors;
- the parameter-count assertion in `tests/test_param_count.py` passes within ±5% of the stated total.
