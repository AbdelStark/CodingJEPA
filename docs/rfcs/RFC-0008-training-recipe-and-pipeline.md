# RFC-0008 — Training recipe and pipeline (LeWorldModel-derived)

## Status
Locked (2026-05-15)

## Problem

Specify the exact training pipeline for CodingJEPA: the recipe inherited from LeWorldModel, the two-stage schedule (pretrain → intent fine-tune), the optimizer, the data loader, and the verification gates.

## Context

This RFC is the implementation contract for the model defined in RFC-0003. It is intentionally written at "config-level" precision so that the training pipeline can be implemented mechanically from this document plus the LeWorldModel reference code at `https://github.com/lucas-maes/le-wm`.

## Decisions locked

### D1 — Inheritance from LeWorldModel

We reuse:
- the `JEPA` module skeleton (`encoder`, `predictor`, `action_encoder`, `projector`, `pred_proj`);
- the `lejepa_forward` loss layout: `loss = pred_loss + λ · sigreg_loss`;
- the `ARPredictor`, `Embedder`, `MLP`, `SIGReg` module shapes;
- the Hydra-based config layout (`config/train/lewm.yaml` analog);
- the `stable_pretraining.Module` / `Manager` training loop (or a thin reimplementation if dependency creep is a concern).

We replace:
- the ViT image encoder with a token-level Transformer (RFC-0003 §D1);
- the pixel preprocessing with the chunk tokenizer (RFC-0012);
- the HDF5 video dataset with our Python chunk dataset (RFC-0002);
- the action vector (continuous control) with a discrete intent embedding (RFC-0003 §D5).

We remove:
- the image augmentation pipeline (`get_img_preprocessor`);
- the per-feature column normalizers (`get_column_normalizer`);
- the planning/MPC inference path (we have retrieval-rerank instead, RFC-0009).

### D2 — Two-stage schedule

**Stage A — Unconditional pretraining.**
- Data: chunk sequences from the 6 train repos (RFC-0014 splits).
- Sequence length: `S = H + n_preds + 1 = 5` chunks (default `H=3`, `n_preds=1`).
- Intent for every position: `[I_NONE]`.
- Target: encoder embedding of the next chunk in the sequence.
- Steps: `S_A = 200,000` optimizer steps.
- Effective batch: 1024 chunk-sequences (B=64 with grad accum = 16) on a single H100.

**Stage B — Intent-conditioned fine-tuning.**
- Data: labeled refactor pairs (RFC-0002).
- Per pair: a 2-chunk sequence `(chunk_before, chunk_after)`, with `act_emb` set to the intent embedding.
- Initialization: Stage A checkpoint.
- Steps: `S_B = 50,000` optimizer steps.
- Effective batch: 512 pairs (B=64, grad accum = 8).

### D3 — Loss

Following LeWM's `lejepa_forward`:

```
output = self.model.encode(batch)
emb     = output["emb"]              # (B, S, D)
act_emb = output["act_emb"]          # (B, S, D)

ctx_emb = emb[:, :H]                 # context
ctx_act = act_emb[:, :H]
tgt_emb = emb[:, n_preds:]           # targets

pred_emb = self.model.predict(ctx_emb, ctx_act)

pred_loss   = (pred_emb - tgt_emb).pow(2).mean()
sigreg_loss = self.sigreg(emb.transpose(0, 1))   # SIGReg over the batch of embeddings
loss        = pred_loss + λ · sigreg_loss
```

- The target is **stop-gradient implicit** via `.detach()` on the prediction MSE (we will use `tgt_emb.detach()` to match LeWM's pattern, even though the LeWM code does not always detach explicitly — we make it explicit here because we share parameters and we want unambiguous semantics).
- `λ = 0.05` by default; the sweep `{0.005, 0.05, 0.5}` runs in Phase 2.

### D4 — Optimizer

- **AdamW**, `lr = 3e-4`, `weight_decay = 0.05`, `betas = (0.9, 0.95)`, `eps = 1e-8`.
- **LinearWarmupCosineAnnealingLR**, 5,000 warmup steps, cosine to `1e-5` over the remaining steps. (Matches LeWM `cfg.optimizer.scheduler`.)
- **Gradient clipping**: `1.0` (global L2 norm).
- **Mixed precision**: `bf16` (`torch.amp.autocast(dtype=torch.bfloat16)`).
- **EMA on weights**: not used (LeWM omits it; we follow).

### D5 — Data loader

- Source: `data/pairs/v1.parquet` for fine-tune, `data/sequences/v1.parquet` for pretrain.
- Workers: 8.
- `pin_memory=True`, `prefetch_factor=4`, `persistent_workers=True`.
- Sampling: uniform random per epoch for pretrain; **intent-balanced** for fine-tune (each minibatch contains roughly equal proportions of the 8 intents, with class weights compensating residual imbalance).
- Sequence assembly is offline (precomputed), not on-the-fly, to ensure determinism.

### D6 — Tokenization

Per RFC-0012:
- SentencePiece BPE, vocab 32k, trained on the 6-repo training corpus.
- Max length 512 tokens per chunk.
- `[CLS]` prepended; `[SEP]` appended.
- Padding to the batch max (no fixed pad to 512).

### D7 — Collapse / sanity gates

We run **embedding-rank diagnostics** at the end of each epoch (or every 5k steps):
1. Sample 10k random chunks from the val split.
2. Compute the projected embeddings.
3. Compute the effective rank: `exp(H(σ))` where `σ` is the normalized singular-value spectrum.
4. **Gate:** effective rank ≥ `0.9 · embed_dim`. If violated, halt and dump diagnostics.

A second gate:
- `pred_loss` must decrease monotonically (averaged over 100 steps) for the first 5k steps. If not, halt.

### D8 — Stage A → Stage B transition

- Load the Stage A checkpoint.
- Reset the optimizer state (fresh AdamW).
- Reset LR warmup (5k steps).
- Keep all module parameters trainable. No frozen-encoder fine-tune in v1; we revisit in v1.5.

### D9 — Reproducibility

- `seed` set globally (Python `random`, NumPy, PyTorch, CUDA).
- DataLoader worker seeding via `torch.utils.data.DataLoader(..., generator=g, worker_init_fn=seed_worker)`.
- The resolved Hydra config is dumped to `runs/<run_id>/config.yaml`.
- The data manifest hash is recorded in the same directory.

### D10 — Runtime budget

Target wall-clock on 1× H100:
- Stage A: ≤ 5 days at 200k steps.
- Stage B: ≤ 24h at 50k steps.

If pretraining exceeds the budget, decrease `B` and `S_A` proportionally and document it in `docs/notes/PHASE-2.md` rather than starting over.

### D11 — Hydra config layout

```
config/
  train/
    pretrain.yaml         # Stage A
    finetune.yaml         # Stage B
    ablations/
      no-sigreg.yaml
      no-intent.yaml
      no-projector.yaml
      lambda-sweep/{0p005,0p05,0p5}.yaml
      history/{H1,H3,H6}.yaml
      size/{small,base,large}.yaml
  data/
    pretrain.yaml
    finetune.yaml
  model/
    coding_jepa.yaml
  optimizer/
    adamw.yaml
  loss/
    sigreg.yaml
```

### D12 — Logging

- WandB integration mirroring LeWM (`WandbLogger`).
- Logged each step: `pred_loss`, `sigreg_loss`, `loss`, `grad_norm`, `lr`.
- Logged each 5k steps: rank diagnostic, retrieval@10 on a 200-pair eval probe.
- Logged each epoch: the full eval harness on the val split (no test access until release).

### D13 — Checkpointing

- Mirror LeWM's `ModelObjectCallBack`.
- Save weights every epoch (`<run_id>/weights_epoch_<N>.ckpt`).
- Save a full module object every epoch for direct loading (`<run_id>/object_epoch_<N>.ckpt`).
- Keep last 3 + best (by val retrieval@10).

### D14 — Hardware target

- **Primary:** 1× NVIDIA H100 (80 GB) on a single host.
- **Secondary:** 1× A100 (80 GB) — same recipe, longer wall-clock.
- **CPU fallback:** not supported.

### D15 — Differences from LeWM worth noting

| Item | LeWM | CodingJEPA |
|---|---|---|
| Modality | pixel frames | code chunks |
| Encoder | ViT (HF) | Transformer encoder over BPE tokens |
| Action | continuous control vector | discrete intent index |
| Sequence axis | time (video frames) | file order over chunks |
| Image normalization | yes (`get_img_preprocessor`) | none; tokenizer handles it |
| Frameskip | `cfg.data.dataset.frameskip` | not applicable |
| Decoder/Planning | MPC over actions (`get_cost`) | none in v1; retrieval-rerank only |
| EMA target | no (LeWM removes it) | no (we follow) |
| SIGReg | yes | yes |

### D16 — Pre-flight checklist before launching a run

1. `data/manifest.lock.json` exists and hashes match a known release.
2. Tokenizer artifact `tokenizer/v1/tokenizer.model` is committed.
3. Tiny-slice run (1k steps, 1k pairs) completes without NaNs.
4. Rank diagnostic on the tiny-slice run shows non-collapse.
5. WandB project is configured; offline mode is acceptable but the config is logged either way.

## Deferred items
- Multi-GPU and FSDP scaling.
- Larger encoder (12-layer, 768-hidden).
- A "soft" intent embedding sampled from a prior during pretraining to bridge stage A and B.
- Adversarial probes during training as auxiliary losses.

## Acceptance condition

Locked when:
- the `config/train/pretrain.yaml` and `config/train/finetune.yaml` configs exist and resolve;
- a tiny-slice run completes and produces a non-collapsed embedding matrix;
- the loss formula above is implemented in `codingjepa/training/forward.py` with a unit test that compares its outputs to a numerically-evaluated reference on a 4-example batch.
