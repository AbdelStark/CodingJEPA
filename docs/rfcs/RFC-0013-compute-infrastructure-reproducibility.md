# RFC-0013 — Compute, infrastructure, reproducibility

## Status
Locked (2026-05-15)

## Problem

Pin the compute environment, the dependency stack, the storage layout, and the reproducibility contract so the artifact can be re-run by anyone with one H100 and the released checkpoints.

## Decisions locked

### D1 — Hardware target
- **Primary:** 1× NVIDIA H100 (80 GB SXM or PCIe), single host.
- **Secondary:** 1× A100 (80 GB). Same recipe with longer wall-clock.
- **Min for dev:** any modern GPU with 24 GB (for tiny-slice runs only).
- **CPU-only is not supported** for training. Inference is supported on CPU but with no latency guarantees.

### D2 — OS and CUDA
- Linux x86_64, kernel ≥ 5.15.
- CUDA 12.4, cuDNN 9.x.
- Driver ≥ 550.

### D3 — Python and dependencies

- **Python 3.12** (consistent with the chunker's `compile()` target).
- Dependency management: `uv` with `pyproject.toml` + `uv.lock`. The lockfile is the source of truth; pinning is exact.
- **Core dependencies (locked):**
  - `torch` (2.4+, with CUDA 12.4 build)
  - `lightning` (2.4+)
  - `hydra-core`
  - `omegaconf`
  - `wandb`
  - `libcst`
  - `sentencepiece`
  - `pydriller`
  - `faiss-cpu` (for indexes that fit in RAM) or `faiss-gpu` if available
  - `rank_bm25` (B1 baseline)
  - `transformers` (for B3 CodeBERT baseline only)
  - `pytest`, `pytest-xdist` (eval-time test sandbox)
  - `pygments` (demo diff rendering)
  - `fastapi`, `htmx` (demo)
- **Reference dependencies for the LeWM port:**
  - `stable_pretraining` (if we adopt it; or a thin reimplementation behind the same interface)
  - `stable_worldmodel` (only its `HDF5Dataset` and utilities — likely not needed; we keep this as an optional reference dep)

Decision: we keep an option to **re-implement the small slice of `stable_pretraining` we use** rather than carrying it as a hard dep. The interface boundary is `codingjepa.training.module` which exposes the LeWM-style `forward` callback and a minimal `Manager`. Dependency creep beyond this is rejected.

### D4 — Storage layout

```
/
├── codingjepa/             # source package
├── config/                 # Hydra configs
├── data/
│   ├── raw/<repo>/         # mirrored source repos (gitignored)
│   ├── parsed/             # chunked parquet
│   ├── sequences/          # pretraining sequences
│   ├── pairs/              # labeled refactor pairs
│   ├── splits/v1.lock.json
│   ├── manifest.lock.json
│   ├── gold/v1.jsonl
│   └── schemas/            # parquet schemas
├── tokenizer/v1/           # SentencePiece artifact
├── indices/                # FAISS indexes (gitignored, content-hashed)
├── runs/                   # training runs (gitignored)
├── results/                # eval outputs (committed)
├── paper/                  # LaTeX source
├── examples/               # demo determinism examples
├── tests/                  # pytest suite
├── docs/                   # PRD, spec, RFCs (this folder)
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

Gitignored: `data/raw/`, `runs/`, `indices/`, `.runs/`, `.artifacts/`, `__pycache__`, `.venv`. Everything else is committed.

### D5 — Reproducibility contract

- **Seeds.** A single `seed` flows through Python `random`, NumPy, PyTorch (`manual_seed`, `cuda.manual_seed_all`), and DataLoader generators.
- **Deterministic algorithms.** `torch.use_deterministic_algorithms(True)` at inference time. At training time we accept non-determinism for performance and re-run ≥ 3 seeds for any reported number.
- **Data manifest.** `data/manifest.lock.json` records: per-repo `commit_sha`, per-file content hash, chunker version, tokenizer hash. Any change re-versions the corpus.
- **Tokenizer pin.** Tokenizer artifact is content-addressed.
- **Run config dump.** Every run dumps the resolved Hydra config to `runs/<run_id>/config.yaml`.
- **Checkpoint metadata.** Every checkpoint carries `(model_hash, code_git_sha, config_hash, data_manifest_hash, tokenizer_hash, torch_version, cuda_version)`.

### D6 — CI

GitHub Actions:
- `lint`: ruff + black --check + mypy on `codingjepa/`.
- `unit`: pytest -m "not slow" on CPU.
- `safety`: runs RFC-0007 §D6 property tests.
- `eval-smoke`: a 10-example fixture run of `make eval` with a stub checkpoint.

We do **not** run training in CI. Training runs are launched manually and logged to WandB.

### D7 — Sandboxing for execution-preservation eval

Per RFC-0010 §D2 E3:
- Each test is run in a `subprocess` with:
  - 30-second wall-clock timeout (`SIGKILL` on overrun);
  - no network (firewall denylist or `nsjail`);
  - read-only filesystem except `/tmp/<run>`;
  - memory cap 4 GB;
  - CPU-only by default; GPU is not exposed to test sandboxes.
- The sandbox returns `{passed: bool, stdout, stderr, exit_code, wall_clock}`; this is the eval's authoritative outcome.

### D8 — Release artifacts

For v1.0:
- Tagged git release `v1.0`.
- Hugging Face Hub repo `<org>/coding-jepa-v1` containing:
  - encoder + projector weights (`encoder.safetensors`, `projector.safetensors`);
  - predictor + pred_proj weights;
  - tokenizer artifact;
  - `MODEL_CARD.md` per HF convention.
- Hugging Face Datasets repo `<org>/coding-jepa-v1-pairs` for the labeled refactor pairs (subject to license compliance in RFC-0014).
- Docker image `coding-jepa:v1.0-eval` pinned by digest.

### D9 — Maintenance posture
- Bug-fix releases (`v1.0.x`) are allowed without re-versioning data.
- Recipe or data changes bump to `v1.1` and require an RFC amendment.
- We do not promise long-term support for the demo UI; the eval harness and checkpoints are the durable artifacts.

## Deferred items
- Multi-host distributed training (FSDP / DDP across nodes).
- TorchInductor / `torch.compile` optimizations beyond `torch.compile(mode="reduce-overhead")` on the encoder.
- Quantized inference.
- Mobile / ONNX export.

## Acceptance condition

Locked when:
- `pyproject.toml` and `uv.lock` are committed and `uv sync` succeeds in a clean container;
- `make` runs a no-op smoke target that prints the resolved environment without errors;
- CI workflows pass on a clean clone.
