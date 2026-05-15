# 01 — Architecture

This document specifies the system's module boundaries, data flow, and the contracts each layer enforces. The detailed design lives in `docs/spec/SYSTEM-SPEC.md` and in the RFCs; this document is the orientation.

## Layered architecture

The system is built in seven layers. Downstream layers must not reach upstream past the documented contract.

```
┌──────────────────────────────────────────────────────────────────────┐
│ L7 — Demo surface (CLI + web)                                        │  RFC-0006, RFC-0009
├──────────────────────────────────────────────────────────────────────┤
│ L6 — Inference pipeline (embed / retrieve / rerank / refuse)         │  RFC-0009, RFC-0007
├──────────────────────────────────────────────────────────────────────┤
│ L5 — Eval harness + benchmark suite                                  │  RFC-0010, RFC-0005
├──────────────────────────────────────────────────────────────────────┤
│ L4 — Training (Stage A pretrain + Stage B intent fine-tune)          │  RFC-0008
├──────────────────────────────────────────────────────────────────────┤
│ L3 — Model (encoder + ARPredictor + projector + intent + SIGReg)     │  RFC-0003
├──────────────────────────────────────────────────────────────────────┤
│ L2 — Data (chunk + tokenize + label + dedup + split)                 │  RFC-0002, RFC-0012, RFC-0014
├──────────────────────────────────────────────────────────────────────┤
│ L1 — Source mirror + manifest                                        │  RFC-0002, RFC-0014
└──────────────────────────────────────────────────────────────────────┘
```

## Module boundaries (Python package layout)

```
codingjepa/
├── data/                 # L1, L2 — mirroring, chunking, labelers, splits
│   ├── mirror.py
│   ├── chunker.py
│   ├── normalize.py
│   ├── tokenizer.py
│   ├── pairs.py
│   ├── splits.py
│   ├── dedup.py
│   ├── secrets_scan.py
│   ├── audit.py
│   └── labelers/         # one labeler per intent (RFC-0002 §D6)
├── intents/              # intent vocabulary, acceptance rules (RFC-0004)
│   └── acceptance.py     # SINGLE source of truth for acceptance rules
├── modules/              # L3 model components (RFC-0003)
│   ├── encoder.py
│   ├── projector.py
│   ├── ar_predictor.py
│   ├── intent_embedder.py
│   ├── pred_proj.py
│   └── sigreg.py
├── model.py              # CodingJEPA top-level (RFC-0003 §D10)
├── training/             # L4 (RFC-0008)
│   ├── module.py         # forward + loss
│   ├── manager.py        # training loop / Lightning module
│   ├── dataloader.py
│   ├── callbacks.py      # rank-diagnostic gate, monotonicity gate, checkpoints
│   └── logging.py        # WandB integration
├── inference/            # L6 (RFC-0009)
│   ├── embed.py
│   ├── index.py          # FAISS IndexFlatIP + sidecar
│   ├── retrieve.py
│   ├── rerank.py
│   └── confidence.py
├── safety/               # L6 — interlocked with inference (RFC-0007)
│   ├── checkers/         # one file per unsafe-transform class
│   ├── filter.py
│   └── messages.py       # stable refusal strings
├── eval/                 # L5 (RFC-0010)
│   ├── harness.py
│   ├── benchmarks/
│   │   ├── ret.py        # CJ-RET-100, CJ-RET-1k
│   │   ├── intent.py     # CJ-INTENT
│   │   ├── exec.py       # CJ-EXEC + sandbox
│   │   ├── robustness.py # CJ-ROB-FMT/RENAME/DOC
│   │   ├── ood.py        # CJ-OOD
│   │   ├── probes.py     # CJ-PROBE-NAME/DEFECT/CLONE
│   │   └── human.py      # CJ-HUMAN
│   ├── pools.py          # deterministic pool construction
│   ├── stats.py          # bootstrap CIs, paired-bootstrap p-values
│   ├── memo.py           # RESULTS-MEMO.md generator
│   └── schema.py         # results.json JSONSchema + validator
├── baselines/            # L5 (RFC-0005)
│   ├── bm25.py
│   ├── mlm_encoder.py
│   └── codebert.py
├── demo/                 # L7 (RFC-0006)
│   ├── cli.py
│   ├── web/              # FastAPI app + HTMX templates
│   ├── diff.py           # pygments-rendered diffs
│   └── messages.py       # demo copy table
└── cli.py                # `python -m codingjepa` entry
```

## Data flow

### Training (Stage A — unconditional pretraining)

```
data/raw/<repo>           (L1: mirror at pinned commit)
  └─> chunker             (L2: libcst → top-level chunks)
        └─> normalizer    (L2: black, isort, docstring sentinel, strip pragmas; RFC-0012 §D5)
              └─> tokenizer  (L2: SentencePiece BPE, vocab 32k, max 512; RFC-0012 §D7)
                    └─> sequences/v1.parquet  (L2: sliding window of S=H+n_preds+1; RFC-0002 §D9)
                          └─> dataloader  (L4: workers=8, deterministic seeding; RFC-0008 §D5)
                                └─> CodingJEPA.encode + predict + loss  (L3, L4)
                                      └─> AdamW + cosine LR, bf16  (L4: RFC-0008 §D4)
                                            └─> checkpoint (every epoch + best-by-val; RFC-0008 §D13)
```

### Training (Stage B — intent-conditioned fine-tune)

```
PyDriller commit walker  (L2: RFC-0002 §D3)
  └─> 8 heuristic labelers  (L2: RFC-0002 §D6)
        └─> pairs/v1.parquet  (L2)
              └─> intent-balanced dataloader  (L4: RFC-0008 §D5)
                    └─> Stage A checkpoint + intent embedding  (L3, L4)
                          └─> finetune checkpoint
```

### Inference (per request)

```
user input (snippet + intent)
  └─> embed (parse → normalize → tokenize → encoder → projector → L2-norm; RFC-0009 §D2)
        └─> predict (act_emb + ARPredictor + pred_proj; RFC-0009 §D4)
              └─> FAISS IndexFlatIP top-M=100  (RFC-0009 §D3)
                    └─> intent acceptance filter  (RFC-0004 §D2 via codingjepa.intents.acceptance)
                          └─> safety checkers  (RFC-0007 §D1 via codingjepa.safety.filter)
                                └─> rerank (cosine + softmax conf at τ=0.1)
                                      └─> refusal if no candidate ≥ τ_refuse=0.55  (RFC-0007 §D2)
                                            └─> diff render + provenance + log
```

### Evaluation (`make eval`)

```
hash check (manifest + checkpoint + index)  (RFC-0010 §D1)
  └─> benchmark runner (each Benchmark.prepare + run + score)
        └─> per-benchmark JSON
              └─> aggregator → results/results.json
                    └─> memo generator → results/RESULTS-MEMO.md
                          └─> diff gallery + confusions HTML  (RFC-0010 §D12)
```

## Cross-cutting contracts

### Intent acceptance — single source of truth

`codingjepa.intents.acceptance` exports one boolean function per intent. It is consumed by:

- the labeler at data-mining time (RFC-0002 §D6);
- the inference rerank filter (RFC-0009 §D5);
- the eval harness scoring (RFC-0010);
- the safety property tests (RFC-0007 §D6).

If any of these stops importing from `codingjepa.intents.acceptance`, that is a contract violation and a CI gate must catch it.

### Provenance

Every chunk, pair, embedding, retrieval result, and demo log row carries `(repo, commit_sha, file_path, node_qualname)`. The provenance is the join key for downstream auditing and right-to-removal (RFC-0014 §D11).

### Determinism

- Seeds: a single `seed` flows through `random`, NumPy, PyTorch (`manual_seed`, `cuda.manual_seed_all`), DataLoader generators (RFC-0013 §D5).
- Inference: `torch.use_deterministic_algorithms(True)` (RFC-0009 §D7).
- Training: non-deterministic for performance; ≥ 3 seeds for any reported number (RFC-0005 §D7).
- Tokenization: bit-exact (SentencePiece) (RFC-0012 §D10).
- Eval pools: content-addressed, frozen in `eval/pools/<benchmark>.lock.json` (RFC-0010 §D9).

### Versioning of artifacts

- Data manifest version → `v1`, `v2`, … never overwrite. Schema in `data/schemas/`.
- Tokenizer artifact pinned by content hash in the manifest.
- FAISS index keyed by `(model_checkpoint_hash, data_manifest_hash)` (RFC-0009 §D3).
- Eval pools content-addressed; an accidental change re-versions the benchmark.

See also `docs/spec/09-release-and-versioning.md`.

## Out of scope of this document

- Inter-process or multi-tenant concerns: not in v1 (RFC-0009 §D12).
- Distributed training: the architecture admits it but v1 is single-host (RFC-0013 §D1).
- Streaming / batched serving: deferred (RFC-0009 deferred items).
- Multi-chunk source history at inference: deferred (the v1 inference path expands a single source embedding `H` times to fill the predictor history; see RFC-0009 §D4).
