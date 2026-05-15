# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
per `docs/spec/09-release-and-versioning.md`.

## [Unreleased]

### Added
- **Release tooling** (#126):
  - `tools/release/hf_model_upload.py` — `huggingface_hub`-based uploader for the four `*.safetensors` weights (encoder/projector/predictor/pred_proj), the `tokenizer/` directory, `MODEL_CARD.md` (renamed to `README.md` on Hub), and `LICENSES/`. Reads `HF_TOKEN` without ever logging it; idempotent via SHA-256 comparison against existing Hub blobs; `--dry-run` prints the full upload plan; default `--repo-id` is `CodingJEPA/coding-jepa-v1`.
  - `tests/tools/test_hf_model_upload.py` — covers dry-run enumeration, default repo-id, token-leak prevention, missing-file handling (exit 1 in real mode, exit 0 in dry-run), deterministic content hashing, and plan-builder targets.
- **License compliance** (#30):
  - `LICENSES/` — verbatim SPDX texts for every license used by the corpus (`Apache-2.0`, `BSD-3-Clause`, `MIT`, `PSF-2.0`).
  - `tools/notice_gen.py` — generate a deterministic `NOTICE` from `data/manifest.lock.json`, listing each upstream repo's SPDX identifier, commit SHA, and copyright.
  - `tests/tools/test_notice_gen.py` — unit tests covering missing-manifest exit code, copyright fallback, determinism, and default paths.
- `.github/workflows/perf.yml` — P50/P95 regression gate (±20%) for inference latency; skips gracefully on CPU-only runners (#22).
- `.github/workflows/nightly.yml` — daily 02:00 UTC slow-marker test run + ML diagnostics; auto-comments on #3 on failure (#23).
- **CI + docs** (#21, #124, #128, #129, PR #200):
  - `.github/workflows/eval-smoke.yml` — runs `pytest -m eval-smoke` on push/PR; times out at 5 min (#21).
  - `paper/main.tex` — full paper skeleton with all RFC-0011 §D4 sections (Abstract through Appendix A–E) (#124).
  - `paper/refs.bib` — BibTeX stubs for CodeBERT, GraphCodeBERT, UniXcoder, I-JEPA, V-JEPA.
  - `paper/Makefile` — `make -C paper` builds PDF with `latexmk`.
  - `docs/notes/PHASE-1.md` — data pipeline notes: corpus summary, audit gates, dedup, leakage (#128).
  - `docs/notes/PHASE-4.md` — demo subsystem notes (#129).
  - `docs/notes/PHASE-5.md` — eval harness notes (#129).
- **Dataset tools** (#174–#177, PR #200):
  - `codingjepa.data.pairs.COMMIT_CUTOFF` — `datetime(2024, 1, 1, UTC)`; commits at/after this date are skipped (RFC-0002 §D11).
  - `codingjepa data pairs --cutoff YYYY-MM-DD` — CLI argument, default `2023-12-31`.
  - `codingjepa.data.manifest` — now writes `commit_cutoff_utc = "2023-12-31T23:59:59Z"` into every manifest.
  - `data/schemas/manifest.schema.json` — `commit_cutoff_utc` is now a required field.
  - `tools/hf_convert.py` — build a HF `DatasetDict` from pairs parquet files and push to HF Hub.
  - `tools/assert_no_secrets.py` — scan corpus parquet files for secrets, exit 1 on any hit.
  - `tools/assert_trufflehog_clean.py` — run `trufflehog filesystem` scan; graceful skip if not installed.
- **Phase 8 — Evaluation harness** (#107–#123, PR #198):
  - `codingjepa.eval.harness` — `Benchmark` ABC with `prepare/run/score`, `BenchmarkResult` dataclass, `run_suite` orchestrator (writes per-benchmark JSON + `results/results.json`).
  - `codingjepa.eval.benchmarks.ret` — `CJ-RET-100` / `CJ-RET-1k`: FAISS `IndexFlatIP` retrieval, R@1/R@5/R@10/MRR.
  - `codingjepa.eval.benchmarks.intent` — `CJ-INTENT`: conditioned vs. unconditional R@5, delta_R5.
  - `codingjepa.eval.benchmarks.exec` — `CJ-EXEC` stub (returns `no_executable_pairs` until sandboxed data is provided).
  - `codingjepa.eval.sandbox` — `run_in_sandbox()` with nsjail/firejail/plain subprocess backends.
  - `codingjepa.eval.benchmarks.robustness` — `CJ-ROB-FMT` / `CJ-ROB-RENAME` / `CJ-ROB-DOC`: rank_change_pct + mean_cosine_drift.
  - `codingjepa.eval.benchmarks.ood` — `CJ-OOD`: R@10 on 200-pair pool.
  - `codingjepa.eval.benchmarks.probes` — `CJ-PROBE-NAME/DEFECT/CLONE` linear probes (sklearn-optional with graceful fallback).
  - `codingjepa.eval.benchmarks.human` — `CJ-HUMAN` stub (returns `no_human_annotations` until annotation file present).
  - `codingjepa.eval.memo` — `generate_memo()` writing RESULTS-MEMO.md with all 11 RFC-0010 §D6 sections.
  - `codingjepa.eval.diff_gallery` — HTML diff gallery for the gold subset.
  - `codingjepa.eval.confusions` — worst-50 error pages per intent.
  - `codingjepa.eval.figures` — matplotlib PDF figure generator (graceful no-op when matplotlib absent).
  - `tests/eval/test_harness.py` — 52-test suite on 10-example fixture covering every benchmark + orchestrator + memo + sandbox + gallery.
- **Phase 7 — Demo subsystem** (#100–#106, PR #172):
  - `codingjepa.demo.cli` — argparse `refactor` command with `--source/--file/--intent/--k/--threshold/--out`.
  - `codingjepa.demo.web` — FastAPI app: `GET /`, `POST /refactor`, `GET /healthz`, `GET /version`.
  - `codingjepa.demo.diff` — `render_diff_terminal()` (pygments) + `render_diff_html()` (self-contained page, monokai).
  - `codingjepa.demo.web.templates` — HTMX form + candidate rendering (no build step, htmx@1.9.12 from unpkg).
  - `codingjepa.demo.messages` — closed refusal copy table (5 demo-path keys).
  - `examples/demo-cpython-extract-helper.py` — deterministic extract-helper diff example (no checkpoint required).
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
