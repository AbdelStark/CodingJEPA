# CodingJEPA — MVP status

Current state: **complete implementation across all 6 phases.** 871 tests pass, 0 fail.

## Completed

- [x] README (rebranded to CodingJEPA, Python focus)
- [x] PRD
- [x] system spec (long-form): `docs/spec/SYSTEM-SPEC.md`
- [x] research notes (incl. LeWorldModel mapping)
- [x] implementation plan
- [x] schedule
- [x] RFC stack (0001–0014)
- [x] candidate Python repositories list (`docs/data/CANDIDATE_REPOS.md`)
- [x] top-level canonical spec (`SPEC.md`) and per-section corpus (`docs/spec/00-overview.md` through `docs/spec/10-glossary.md`)
- [x] implementation tracker at `docs/roadmap/IMPLEMENTATION.md`

### Phase 1 — Data pipeline (issues #34–#57)
- [x] `codingjepa.data.mirror` — 10-repo registry with pinned SHAs, idempotent clone
- [x] `codingjepa.data.normalize` — black + isort + docstring→`<DOC>` + pragma/email strip
- [x] `codingjepa.data.chunker` — libcst FunctionDef/AsyncFunctionDef/ClassDef/interstitial, SHA-256 chunk IDs
- [x] `codingjepa.data.pairs` — PyDriller commit walker, 4 commit-level + content-level filters
- [x] 8 labelers — extract-helper, inline-helper, comprehension-rewrite, dataclass-migration, exception-handling-cleanup, loop-to-vectorized, argument-defaulting, none-typing-modernization
- [x] `codingjepa.data.tokenizer` — SentencePiece BPE 32k + 15 special tokens, encode/decode/audit
- [x] `codingjepa.data.dedup` — SHA-256 exact + MinHash LSH near-dedup (Jaccard ≥ 0.85, 128 functions, 32 bands)
- [x] `codingjepa.data.splits` — by-repo assignment + MinHash cross-split leakage detector
- [x] `codingjepa.data.secrets_scan` — regex + `codingjepa.safety.secret_patterns`
- [x] `codingjepa.data.audit` — 4 gates: compile ≥ 0.95, dedup < 0.30, license ∈ allowed, secrets == 0
- [x] `codingjepa.data.manifest` — content-addressed SHA-256, schema-validated
- [x] `codingjepa.data.sequences` — sliding-window H=3 + n_preds=1 builder
- [x] `codingjepa.data.cli` — argparse subcommands: mirror/chunk/pairs/dedup/splits/audit/manifest/all
- [x] Per-intent quotas — 12,000 train pairs/intent cap with overflow parquet

### Phase 2 — Model stack (issues #58–#65)
- [x] `codingjepa.modules.encoder` — 6-layer pre-norm Transformer, RoPE, 8 heads, hidden 512, GELU
- [x] `codingjepa.modules.projector` — Linear(512,2048)→BatchNorm1d→ReLU→Linear(2048,512)
- [x] `codingjepa.modules.ar_predictor` — 4-layer norm_first TransformerEncoder
- [x] `codingjepa.modules.pred_proj` — mirror of projector, independent parameters
- [x] `codingjepa.modules.intent_embedder` — nn.Embedding(9,512), index 8 = NONE
- [x] `codingjepa.modules.sigreg` — sliced isotropic Gaussian regularizer, K=256 projections
- [x] `codingjepa.model.CodingJEPA` — full forward(), embed(), ForwardResult dataclass
- [x] Tiny-slice training pass (loss decreasing in 10 steps with random data)

### Phase 3 — Training infrastructure (issues #66–#74)
- [x] `codingjepa.training.optimizer` — AdamW + LinearWarmupCosine (5k warmup, cosine to 1e-5)
- [x] `codingjepa.training.module` — bf16 AMP, grad-clip=1.0, metric logging
- [x] `codingjepa.training.dataloader` — ChunkSequenceDataset + RefactorPairDataset, intent-balanced sampler
- [x] `codingjepa.training.callbacks` — RankDiagnostic, LossMonotonicity, Checkpoint (keep_last=3 + best-by-val)
- [x] `codingjepa.training.manager` — Manager.fit() loop with validation cadence
- [x] `codingjepa.training.preflight` — gates on manifest, audits, baselines, model, GPU
- [x] `codingjepa.training.logging` — WandBLogger with graceful fallback

### Phase 4 — Baselines (issues #78–#81)
- [x] BM25 over BPE tokens (`rank_bm25`)
- [x] MLM-encoder baseline (same arch, 15% mask)
- [x] Frozen CodeBERT at pinned revision `3b6e86c`
- [x] `check_baselines_first()` gate in preflight — refuses training without results JSONs

### Phase 5 — Safety checkers (issues #91–#98)
- [x] `side_effect_introduction`, `side_effect_elimination` (5 recognisable side-effect families)
- [x] `exception_contract_change` (Raise-node AST diff)
- [x] `public_api_change` (rename/param/annotation detection)
- [x] `async_sync_boundary` (sync↔async flip detection)
- [x] `codingjepa.safety.filter.run()` — short-circuit chain, R006 refusal code
- [x] Hypothesis property tests (60 tests, 50 examples each)

### Phase 6 — Inference pipeline (issues #82–#90)
- [x] `codingjepa.inference.embed` — full normalize→tokenize→encode→project→L2-norm pipeline
- [x] `codingjepa.inference.index` — FAISS IndexFlatIP + `.meta.json` sidecar, `index_id` versioning
- [x] `codingjepa.inference.retrieve` — predictor-history expand (H=3), FAISS top-M=100
- [x] `codingjepa.inference.rerank` — safety filter + softmax-at-τ=0.1 confidence
- [x] `codingjepa.inference.confidence` — numerically stable calibration
- [x] `codingjepa.inference.infer()` — R001/R004/R005/R006 refusal codes
- [x] `index_id` hash-drift enforcement (raises `IndexHashMismatch`)
- [x] Round-trip determinism test (`torch.use_deterministic_algorithms`)
- [x] Latency test stub (P50/P95 skeleton, `@pytest.mark.slow`)

### Phase 7 — Demo subsystem (issues #100–#106)
- [x] `codingjepa.demo.cli` — argparse refactor command, `--source/--file/--intent/--k/--threshold/--out`
- [x] `codingjepa.demo.web` — FastAPI app, GET /, POST /refactor, GET /healthz, GET /version
- [x] `codingjepa.demo.diff` — `render_diff_terminal()` (pygments) + `render_diff_html()` (self-contained HTML page)
- [x] `codingjepa.demo.web.templates` — HTMX form + candidate rendering (no build step)
- [x] `codingjepa.demo.messages` — closed refusal copy table (demo path, 5 keys)
- [x] `examples/demo-cpython-extract-helper.py` — deterministic extract-helper diff example (no checkpoint)
- [x] Hidden-step ban enforcement in `render_diff_html` (`data-hidden-step` assertion)

### Phase 8 — Evaluation harness (issues #107–#123)
- [x] `codingjepa.eval.harness` — `Benchmark` ABC + `BenchmarkResult` + `run_suite` orchestrator (writes `results/results.json`)
- [x] `codingjepa.eval.benchmarks.ret` — `CJ-RET-100` / `CJ-RET-1k`: FAISS IndexFlatIP, R@1/R@5/R@10/MRR
- [x] `codingjepa.eval.benchmarks.intent` — `CJ-INTENT`: conditioned vs. unconditional R@5, delta_R5
- [x] `codingjepa.eval.benchmarks.exec` — `CJ-EXEC` stub (no_executable_pairs until sandboxed data present)
- [x] `codingjepa.eval.sandbox` — `run_in_sandbox()` with nsjail/firejail/plain backends
- [x] `codingjepa.eval.benchmarks.robustness` — `CJ-ROB-FMT` / `CJ-ROB-RENAME` / `CJ-ROB-DOC`: rank_change_pct + cosine_drift
- [x] `codingjepa.eval.benchmarks.ood` — `CJ-OOD`: R@10 on 200-pair pool
- [x] `codingjepa.eval.benchmarks.probes` — `CJ-PROBE-NAME/DEFECT/CLONE` linear probes (sklearn-optional)
- [x] `codingjepa.eval.benchmarks.human` — `CJ-HUMAN` stub (no_human_annotations until annotation file present)
- [x] `codingjepa.eval.memo` — `generate_memo()`: RESULTS-MEMO.md with all 11 RFC-0010 §D6 sections
- [x] `codingjepa.eval.diff_gallery` — HTML diff gallery for gold subset
- [x] `codingjepa.eval.confusions` — worst-50 error pages per intent
- [x] `codingjepa.eval.figures` — matplotlib PDF generator (graceful no-op when matplotlib absent)
- [x] `tests/eval/test_harness.py` — 52 tests on 10-example fixture (RFC-0010 acceptance criteria)

## Not yet started

- [ ] Actual training runs (#75 Stage A pretrain, #76 Stage B fine-tune) — require GPU compute
- [ ] Paper draft + HF Hub upload (#124–#129)
- [ ] Hydra config tree (#17), Dockerfile.eval (#24), LICENSES/ (#30)
- [ ] Gold subset annotation tooling (#55)

## Rule

Do not mark implementation milestones complete until the repository contains runnable or inspectable artifacts proving them. Spec-only completeness is not implementation completeness.

## Renaming note

This project was previously named **RustJEPA** and targeted Rust refactoring. As of 2026-05-15 it has been renamed to **CodingJEPA** and retargeted to Python. The Rust scoping documents are obsolete; the current spec stack is authoritative.
