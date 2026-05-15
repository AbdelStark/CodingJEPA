# 08 — Performance budget

This document specifies the compute, latency, throughput, and memory budgets the system must hit. Numbers are commitments, not aspirations: a regression past one of them is a blocking issue.

## Training compute (RFC-0008 §D10, RFC-0013 §D1)

| Item | Budget | Notes |
|---|---|---|
| Hardware | 1× H100 (80 GB) | Primary. A100 (80 GB) acceptable with longer wall-clock. |
| Stage A (pretrain) wall-clock | ≤ 5 days at 200k steps | If exceeded: decrease `B` and `S_A` proportionally and document in `docs/notes/PHASE-2.md`. |
| Stage B (intent fine-tune) wall-clock | ≤ 24 h at 50k steps | |
| Effective batch (Stage A) | 1024 chunk-sequences (B=64 × grad_accum=16) | RFC-0008 §D2 |
| Effective batch (Stage B) | 512 pairs (B=64 × grad_accum=8) | |
| Mixed precision | bf16 | RFC-0008 §D4 |
| Per-step time target (Stage A) | ≤ 2.0 s | At 200k steps × 2.0 s = ~111 h ≈ 4.6 days. Within 5-day budget. |
| Per-step time target (Stage B) | ≤ 1.5 s | At 50k × 1.5 s = ~21 h. |
| Memory (training) | ≤ 70 GB resident on H100 | bf16 + grad accum; checkpoint shards if violated. |

If per-step times regress by > 10% vs. a recent green run, an issue is opened with the run config and a profiler trace before launching a full run.

## Inference latency (RFC-0009 §D10)

| Percentile | Target | Test |
|---|---|---|
| P50 | < 400 ms | `tests/perf/test_latency.py` |
| P95 | < 1.5 s | same; CI fails on > 20% regression |
| P99 | < 2.5 s | observed; not a hard gate |

Conditions: 512-token chunk input, 100k-vector index, H100, single request, deterministic mode on. The latency budget is explicitly *with* `torch.use_deterministic_algorithms(True)`; we accept a small perf hit (≤ 15%) for reproducibility (RFC-0009 §D7).

### Inference latency budget breakdown (target P50)

| Stage | Budget | Notes |
|---|---|---|
| Parse + normalize + tokenize | < 30 ms | libcst on a 512-token chunk. |
| Encoder forward (B=1) | < 60 ms | 6-layer, 512-hidden, bf16 on H100. |
| Predictor forward | < 25 ms | 4-layer, expand-to-H. |
| FAISS top-100 over 100k | < 30 ms | `IndexFlatIP` exact. |
| Acceptance + safety filter (top-100) | < 100 ms | Per-candidate libcst parse (cached per chunk_id). |
| Diff render + response | < 50 ms | difflib + pygments. |
| Total | ≤ 295 ms | Headroom against the 400 ms P50 target. |

If any stage breaches its bracket consistently, the regression issue cites the stage and the profiler output.

## Inference memory (RFC-0009 §D11)

| Item | Target |
|---|---|
| Encoder + predictor + projector weights (FP16) | ~250 MB |
| Activation memory (B=1) | ~1 GB |
| FAISS index for v1 corpus (~50k chunks × 512 floats × 4 B) | ~100 MB |
| Acceptance metadata sidecar | ~50 MB |
| Resident set total | ≤ 4 GB |

CI does not enforce memory; nightly perf job records `psutil.Process().memory_info().rss` and trends it.

## Throughput (training)

| Item | Target | Notes |
|---|---|---|
| Tokens/sec (Stage A) | ≥ 65k tokens/s on H100 | bf16, B=64, S=5 chunks × 512 tokens. |
| Tokens/sec (Stage B) | ≥ 50k tokens/s on H100 | B=64, 2 chunks × 512 tokens. |
| GPU util | ≥ 70% sustained | Lower indicates dataloader starvation; investigate before re-launching. |

WandB time-series records these per step; the rank diagnostic and gate checks include them.

## Throughput (inference)

| Item | Target |
|---|---|
| Throughput (B=1, sustained) | ≥ 2 req/s on H100 |
| Throughput (batched, B=8, future) | ≥ 16 req/s | Out of scope for v1 (RFC-0009 §D12). |

Batched serving is deferred; v1 processes requests serially.

## Storage (RFC-0013 §D4)

| Item | Target |
|---|---|
| `data/raw/` | ~30 GB (10 repos at pinned commits) |
| `data/parsed/` | ~3 GB (parquet, BPE-tokenized) |
| `data/sequences/v1.parquet` | ~1 GB |
| `data/pairs/v1.parquet` | ~250 MB |
| Checkpoints (3 + best, per stage) | ~6 GB total |
| FAISS indices (per `index_id`) | ~150 MB |
| Total budget | ≤ 200 GB |

If storage exceeds budget, gitignored caches are pruned in this order: `runs/<old>/checkpoints/object_*` → `data/raw/` → ablation checkpoints. Released artifacts on HF Hub are never pruned locally without confirmation.

## Network

- Source mirroring: ~30 GB over `git clone --depth 1` against the pinned commits.
- HF Hub pulls (CodeBERT, model card, tokenizer): ~500 MB.
- WandB upload: ≤ 10 MB / day during training.
- No network during inference; no network in the eval sandbox (RFC-0013 §D7).

## Eval wall-clock (RFC-0010 §D10)

| Block | Target |
|---|---|
| E1 (12k pairs × encode + retrieve at N=100/1000) | ≤ 30 min |
| E3 (500 sandboxed pytest invocations) | ≤ 2 h |
| E4 (4k embeddings, 3 perturbers) | ≤ 10 min |
| E6 (linear probes) | ≤ 30 min |
| Full `make eval` per system (CodingJEPA + 3 baselines) | ≤ 4 h × 4 = 16 h |

## What we do NOT optimize for in v1

- Multi-GPU scaling (FSDP/DDP) — RFC-0013 deferred.
- `torch.compile` aggressive modes beyond `mode="reduce-overhead"` on the encoder.
- Quantized inference (int8 / fp8).
- Approximate-NN indexes (HNSW / IVF).
- Mobile / ONNX / TensorRT export.

Adding any of these is a v1.x amendment with a justification tied to a measured bottleneck.

## Regression gates

| Gate | Threshold | Workflow |
|---|---|---|
| Inference P95 latency | +20% vs. last green main | `perf` CI job |
| Stage A per-step | +10% vs. last green | nightly perf job; opens an issue |
| Stage B per-step | +10% vs. last green | nightly perf job |
| Eval E1 wall-clock | +25% vs. last green | nightly eval job |
| Resident-set memory at inference | +15% vs. last green | nightly perf job |

Issues opened by automated gates are labeled `priority:p1` and assigned to the contributor whose merge introduced the regression.
