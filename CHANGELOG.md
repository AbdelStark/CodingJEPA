# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
per `docs/spec/09-release-and-versioning.md`.

## [Unreleased]

### Added
- **Phase 6 — Inference pipeline** (#82–#90, PR #171):
  - `codingjepa.inference.embed` — normalize→tokenize→encode→project→L2-norm, returns `None` on parse fail or >512 tokens.
  - `codingjepa.inference.index` — FAISS `IndexFlatIP` with `.meta.json` sidecar; `index_id = f"{checkpoint_hash[:8]}-{manifest_hash[:8]}"` versioning; `load_index()` raises `IndexHashMismatch` on drift.
  - `codingjepa.inference.retrieve` — predictor-history expand (H=3 repeats), FAISS top-M=100 query.
  - `codingjepa.inference.rerank` — safety filter per candidate + softmax-at-τ=0.1 calibrated confidence.
  - `codingjepa.inference.confidence` — numerically stable `calibrate()`.
  - `codingjepa.inference.infer()` — R001/R004/R005/R006 refusal codes.
  - `tests/inference/test_round_trip.py` — bit-equal two-run determinism test.
  - `tests/perf/test_latency.py` — P50/P95 skeleton (`@pytest.mark.slow`).
- **Phase 5 — Safety checkers** (#91–#98, PR #170):
  - `codingjepa.safety.checkers.side_effect_introduction` — detects new `print`/`logging`/`os`/`subprocess`/`requests`/etc. calls.
  - `codingjepa.safety.checkers.side_effect_elimination` — detects removed side-effect calls.
  - `codingjepa.safety.checkers.exception_contract_change` — Raise-node AST diff.
  - `codingjepa.safety.checkers.public_api_change` — rename/param/annotation detection.
  - `codingjepa.safety.checkers.async_sync_boundary` — sync↔async flip detection.
  - `codingjepa.safety.filter.run()` — short-circuit chain returning `R006_SAFETY_CHECKER_REJECTED_ALL`.
  - Hypothesis property tests: 60 tests, `max_examples=50` each.
- **Phase 4 — Baselines** (#78–#81, PR #169):
  - `codingjepa.baselines.bm25` — BM25Okapi over BPE token IDs; `run()` + `write_results()`.
  - `codingjepa.baselines.mlm_encoder` — `MLMEncoder` with 15% BERT-style masking; `embed()` returns L2-norm CLS.
  - `codingjepa.baselines.codebert` — frozen `microsoft/codebert-base` at pinned revision `3b6e86c`; `write_lock_file()`.
  - `codingjepa.training.preflight.check_baselines_first()` — raises `ConfigError("baseline missing")` if any `results.json` absent.
- **Phase 3 — Training infrastructure** (#66–#74, PR #168):
  - `codingjepa.training.optimizer` — AdamW (lr=3e-4, wd=0.05) + LinearWarmupCosine (5k warmup steps, cosine to 1e-5).
  - `codingjepa.training.module` — `TrainingModule` with bf16 AMP (CUDA only), grad-clip=1.0.
  - `codingjepa.training.dataloader` — `ChunkSequenceDataset`, `RefactorPairDataset`, intent-balanced `WeightedRandomSampler`.
  - `codingjepa.training.callbacks` — `RankDiagnostic` (effective-rank ≥ 0.9×d gate), `LossMonotonicity`, `Checkpoint` (keep_last=3 + best-by-val).
  - `codingjepa.training.manager` — `Manager.fit()` with validation cadence and callback dispatch.
  - `codingjepa.training.preflight` — `run_preflight()` gates on manifest, audits, baselines, model, GPU.
  - `codingjepa.training.logging` — `WandBLogger` with graceful fallback when wandb disabled.
- **Phase 2 — Model stack** (#58–#65, PR #167):
  - `codingjepa.modules.encoder` — 6-layer pre-norm Transformer, hand-rolled RoPE, 8 heads, hidden 512, GELU, dropout 0.1.
  - `codingjepa.modules.projector` — `Linear(512,2048)→BatchNorm1d→ReLU→Linear(2048,512)`.
  - `codingjepa.modules.ar_predictor` — 4-layer `nn.TransformerEncoder` with `norm_first=True`.
  - `codingjepa.modules.pred_proj` — mirror of projector, independent parameters.
  - `codingjepa.modules.intent_embedder` — `nn.Embedding(9,512)`, index 8 = `[I_NONE]`.
  - `codingjepa.modules.sigreg` — sliced isotropic Gaussian regularizer, K=256 random projections.
  - `codingjepa.model.CodingJEPA` — full `forward()`, `embed()`, `ForwardResult` dataclass; `build_model()`.
  - Tiny-slice training pass verified: loss decreasing in 10 steps.
- **Phase 1 — Data pipeline** (#34–#57, PRs #156–#166):
  - `codingjepa.data.mirror` — 10-repo REPO_REGISTRY with pinned 40-char SHAs; idempotent `git clone --filter=blob:none --depth=1`.
  - `codingjepa.data.normalize` — black + isort + docstring→`<DOC>` + pragma/email strip + compile() gate.
  - `codingjepa.data.chunker` — libcst FunctionDef/AsyncFunctionDef/ClassDef/interstitial; SHA-256 chunk IDs; `PositionProvider` line numbers.
  - `codingjepa.data.pairs` — PyDriller `Repository` walker; merge/bot/wip/whitespace-only filters.
  - 8 labelers (extract-helper through none-typing-modernization) — `LABELERS` registry, `label_pair()` dispatcher.
  - `codingjepa.data.tokenizer` — `Tokenizer` wrapping SentencePiece BPE 32k + 15 special tokens; `encode()`, `decode()`, `audit_coverage()`.
  - `codingjepa.data.dedup` — SHA-256 exact + MinHash LSH near-dedup (128 functions, 32 bands, Jaccard ≥ 0.85).
  - `codingjepa.data.splits` — by-repo assignment; `detect_leakage()` via MinHash Jaccard.
  - `codingjepa.data.secrets_scan` — extends `secret_patterns`, `scan_chunks()` drops any chunk with a hit.
  - `codingjepa.data.audit` — 4 hard gates; `AuditGateError` in closed taxonomy; schema-valid audit JSON.
  - `codingjepa.data.manifest` — `write_manifest()` with SHA-256 content address; `verify_manifest_hash()`.
  - `codingjepa.data.sequences` — H=3 context window builder; `apply_intent_quotas()` caps 12,000/intent.
  - `codingjepa.data.cli` — `add_data_subparser()` with mirror/chunk/pairs/dedup/splits/audit/manifest/all.
- Public package skeleton at `codingjepa/` matching `docs/spec/01-architecture.md`.
- `codingjepa.errors` closed exception taxonomy (`docs/spec/04-error-model.md`).
- `codingjepa.observability` structured-log writer, redactor, and UUIDv7
  `request_id` propagation (`docs/spec/05-observability.md`).
- `codingjepa.intents.acceptance` single-source-of-truth acceptance check
  for the 8 RFC-0004 §D2 intents.
- `codingjepa.safety.messages` closed refusal copy table R001–R007
  (RFC-0007 §D7).
- `codingjepa.safety.secret_patterns` full redactor/scanner pattern set
  from spec/05 §Redaction table.
- `data/schemas/` JSONSchemas for `manifest`, `splits`, `audit`, `dedup`,
  `cross_split_leakage`, `log`, `results`, `pool`, `gold`, `model_card`.
- Top-level `Makefile`, `pyproject.toml`, `uv.lock`.
- GitHub Actions workflows: `lint`, `unit`, `safety`, `changelog`.
- `codingjepa.eval.pools` deterministic, content-addressed eval pool
  construction; pools keyed by SHA-256 of sorted chunk IDs
  (`eval/pools/*.lock.json`).
- `codingjepa.eval.stats` bootstrap CI and paired-bootstrap p-value
  implementation (RFC-0005 §D6).
- `codingjepa.eval.schema.validate` — results.json schema validator with
  `jsonschema` enforcement (RFC-0010 §D5, spec/03).
- Hash-check enforcer in `codingjepa.eval.harness` — `make eval` refuses
  with exit code 4 on manifest / checkpoint / index hash drift
  (RFC-0010 §D1, spec/04).
- `MODEL_CARD.md` template with SHA-256 placeholders and
  `tools/model_card_update.py` updater script (RFC-0013 §D2).
- `docs/notes/RELEASE-RUNBOOK.md` v1.0.0 release checklist.
- Pre-commit hooks: `ruff`, `black`, `mypy`, `prettier` on `.yml`/`.yaml`/`.md`.
- `CONTRIBUTING.md`, GitHub PR template, and issue templates.
- `tests/test_invariants.py` — cross-artifact invariant suite verifying
  chunk_id ↔ pairs ↔ pools ↔ manifest consistency.

### Changed
- _none_

### Deprecated
- _none_

### Removed
- _none_

### Fixed
- _none_

### Security
- `SECURITY.md` with disclosure path and 90-day coordinated-disclosure window.

### Reproducibility
- `uv.lock` pins 129 packages on Python 3.12.
- `data/schemas/manifest.schema.json` requires sha256-shaped
  `manifest_hash` + `tokenizer_hash`; `tests/test_invariants.py` verifies
  the canonicalization rule.
- `eval/pools/*.lock.json` records the SHA-256 content address of each eval
  pool so `make eval` can detect drift before running benchmarks.

<!--
When adding an entry, place it under the matching heading above. Pick the
most accurate of:
  Added         — new public symbol, CLI flag, JSON schema, etc.
  Changed       — backwards-compatible change to existing public surface.
  Deprecated    — public surface scheduled for removal in a later release.
  Removed       — public surface removed (major-version-only).
  Fixed         — bug fixes that change observable behavior.
  Security      — fixes for vulnerabilities; references SECURITY.md.
  Reproducibility — data/schema/seed/lockfile/checkpoint changes; required
                    when public artifacts change per spec/09 §Changelog
                    discipline.

The `.github/workflows/changelog.yml` gate fails any PR touching
codingjepa/, data/schemas/, or docs/spec/02-public-api.md without an
[Unreleased] entry. Stub additions ("Add CHANGELOG entry") satisfy the
mechanical check but not the spec — write the actual entry.
-->

[Unreleased]: https://github.com/AbdelStark/CodingJEPA/compare/HEAD...HEAD
