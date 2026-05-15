# Implementation Tracker — 2026-05-15

Generated from the spec corpus on branch `claude/follow-goal-docs-Z8iUv`. Every implementable unit of work in the spec is filed below. Each issue is independently shippable; cross-issue dependencies are noted inline.

The label / milestone convention from `docs/goal.md` §Phase 2.2 is recorded in each issue's `Metadata` block. The GitHub MCP available in the bootstrap environment did not expose label/milestone-creation endpoints; labels were auto-created via first-use on the `issue_write` API. Maintainers can apply milestones (`v0.1`–`v1.0`) to the issues using the milestone column below.

## Tracking issues (subsystem dashboards)

| # | Subsystem | RFCs |
|---|---|---|
| #2 | [meta] tracker: spec corpus + implementation issue set bootstrap | — |
| #3 | [Tracking] Infrastructure & reproducibility | RFC-0013, spec/02, spec/04, spec/05, spec/06, spec/07, spec/09 |
| #4 | [Tracking] Data pipeline | RFC-0002, RFC-0004, RFC-0012, RFC-0014 |
| #5 | [Tracking] Model stack | RFC-0003 |
| #6 | [Tracking] Training pipeline | RFC-0008 |
| #7 | [Tracking] Baselines | RFC-0005 |
| #8 | [Tracking] Inference pipeline | RFC-0009, RFC-0007 |
| #9 | [Tracking] Safety rails | RFC-0007 |
| #10 | [Tracking] Demo & developer workflow | RFC-0006, RFC-0009 |
| #11 | [Tracking] Evaluation harness | RFC-0010, RFC-0005 |
| #12 | [Tracking] Paper, release, packaging | RFC-0011, RFC-0013, RFC-0014, spec/09 |

## Milestone: v0.1 — data pipeline frozen + baselines + infrastructure ready

| # | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|-------|------|----------|--------|------------|----------|
| #13 | infra: initialize pyproject.toml | infra | p0 | s | RFC-0013 | #3 |
| #14 | infra: generate uv.lock; require uv sync --frozen in CI | infra | p0 | s | RFC-0013, spec/06 | #3 |
| #15 | infra: top-level Makefile | infra | p0 | s | RFC-0006, spec/02 | #3 |
| #16 | infra: scaffold codingjepa/ package skeleton | infra | p0 | m | spec/01, spec/02 | #3 |
| #18 | ci: lint workflow | infra | p0 | s | RFC-0013, spec/07 | #3 |
| #19 | ci: unit workflow | infra | p0 | s | RFC-0013, spec/07 | #3 |
| #20 | ci: safety workflow | infra | p0 | s | RFC-0013, RFC-0007, spec/06 | #3 |
| #25 | infra: codingjepa.errors module | infra | p0 | s | spec/04 | #3 |
| #26 | infra: codingjepa.observability | infra | p0 | m | spec/05, spec/06 | #3 |
| #27 | infra: CHANGELOG.md skeleton | infra | p1 | s | spec/09 | #3 |
| #28 | docs: CONTRIBUTING.md + PR / issue templates | infra | p1 | s | spec/09, spec/07 | #3 |
| #29 | security: SECURITY.md | infra | p1 | s | spec/06 | #3 |
| #31 | infra: data/schemas/ JSONSchemas | infra | p0 | m | spec/03 | #3 |
| #32 | tests: cross-artifact invariants | infra | p0 | m | spec/03 | #3 |
| #33 | infra: pre-commit hooks | infra | p2 | s | spec/07 | #3 |
| #34 | data: codingjepa.data.mirror | data | p0 | m | RFC-0002, RFC-0014 | #4 |
| #35 | data: codingjepa.data.chunker | data | p0 | l | RFC-0012, spec/03 | #4 |
| #36 | data: codingjepa.data.normalize | data | p0 | m | RFC-0012 | #4 |
| #37 | data: SentencePiece BPE tokenizer training | data | p0 | m | RFC-0012 | #4 |
| #38 | data: tokenizer coverage audit | data | p0 | s | RFC-0012 | #4 |
| #39 | data: PyDriller commit walker for refactor pairs | data | p0 | m | RFC-0002 | #4 |
| #40 | intents: codingjepa.intents.acceptance | data | p0 | l | RFC-0004, spec/01 | #4 |
| #41 | data: labeler — extract-helper | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #42 | data: labeler — inline-helper | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #43 | data: labeler — comprehension-rewrite | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #44 | data: labeler — dataclass-migration | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #45 | data: labeler — exception-handling-cleanup | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #46 | data: labeler — loop-to-vectorized | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #47 | data: labeler — argument-defaulting | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #48 | data: labeler — none-typing-modernization | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #49 | data: codingjepa.data.dedup | data | p0 | m | RFC-0014 | #4 |
| #50 | data: codingjepa.data.splits + cross-split leakage | data | p0 | m | RFC-0014 | #4 |
| #51 | data: codingjepa.data.secrets_scan | data | p0 | m | RFC-0014, spec/06 | #4 |
| #52 | data: codingjepa.data.audit | data | p0 | m | RFC-0002, RFC-0014 | #4 |
| #53 | data: codingjepa.data.manifest | data | p0 | s | RFC-0014, spec/03 | #4 |
| #54 | data: codingjepa.data.sequences | data | p0 | s | RFC-0002, RFC-0012 | #4 |
| #55 | data: gold subset annotation tooling + 200 pairs | data | p1 | l | RFC-0002 | #4 |
| #56 | data: codingjepa.data.cli wiring | data | p0 | s | spec/02 | #4 |
| #57 | data: per-intent quotas | data | p0 | s | RFC-0002 | #4 |
| #78 | baseline: B1 — BM25 over BPE tokens | baselines | p0 | m | RFC-0005 | #7 |
| #80 | baseline: B3 — frozen CodeBERT | baselines | p0 | m | RFC-0005, spec/06 | #7 |
| #81 | baseline: cheap-baseline-first gate | baselines | p0 | s | RFC-0005 | #7 |
| #97 | safety: stable refusal copy table | safety | p0 | s | RFC-0007, spec/04 | #9 |
| #99 | safety: secret pattern table | safety | p1 | s | spec/05, spec/06 | #9 |
| #128 | release: PHASE-1.md note (data freeze) | release | p1 | s | spec/IMPLEMENTATION-PLAN | #12 |

## Milestone: v0.2 — model stack + Stage A pretrain

| # | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|-------|------|----------|--------|------------|----------|
| #17 | infra: Hydra config tree skeleton | infra | p1 | m | RFC-0008 | #3 |
| #58 | model: codingjepa.modules.encoder | model | p0 | m | RFC-0003 | #5 |
| #59 | model: codingjepa.modules.projector | model | p0 | s | RFC-0003 | #5 |
| #60 | model: codingjepa.modules.ar_predictor | model | p0 | m | RFC-0003 | #5 |
| #61 | model: codingjepa.modules.pred_proj | model | p0 | s | RFC-0003 | #5 |
| #62 | model: codingjepa.modules.intent_embedder | model | p0 | s | RFC-0003 | #5 |
| #63 | model: codingjepa.modules.sigreg | model | p0 | m | RFC-0003 | #5 |
| #64 | model: codingjepa.model.CodingJEPA | model | p0 | m | RFC-0003, RFC-0008, spec/02 | #5 |
| #65 | model: tiny-slice training pass | model | p0 | m | RFC-0003, RFC-0008 | #5 |
| #66 | train: codingjepa.training.module | training | p0 | m | RFC-0008 | #6 |
| #67 | train: codingjepa.training.manager | training | p0 | l | RFC-0008 | #6 |
| #68 | train: codingjepa.training.dataloader | training | p0 | m | RFC-0008 | #6 |
| #69 | train: codingjepa.training.optimizer | training | p0 | s | RFC-0008 | #6 |
| #70 | train: callbacks.RankDiagnostic | training | p0 | s | RFC-0008, spec/04 | #6 |
| #71 | train: callbacks.LossMonotonicity | training | p0 | s | RFC-0008, spec/04 | #6 |
| #72 | train: callbacks.Checkpoint | training | p0 | s | RFC-0008 | #6 |
| #73 | train: WandB integration | training | p1 | s | RFC-0008, spec/06 | #6 |
| #74 | train: codingjepa.training.preflight | training | p0 | s | RFC-0008 | #6 |
| #75 | train: Stage A pretrain run | training | p0 | l | RFC-0008 | #6 |
| #79 | baseline: B2 — MLM-encoder | baselines | p0 | l | RFC-0005 | #7 |

## Milestone: v0.3 — fine-tune + retrieval + safety + inference

| # | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|-------|------|----------|--------|------------|----------|
| #22 | ci: perf workflow with regression gate | infra | p1 | m | RFC-0009, spec/08 | #3 |
| #76 | train: Stage B intent fine-tune run | training | p0 | l | RFC-0008 | #6 |
| #77 | docs: PHASE-2.md and PHASE-3.md notes | training | p1 | s | spec/IMPLEMENTATION-PLAN | #6 |
| #82 | inference: codingjepa.inference.embed | inference | p0 | s | RFC-0009 | #8 |
| #83 | inference: codingjepa.inference.index | inference | p0 | m | RFC-0009, spec/03 | #8 |
| #84 | inference: codingjepa.inference.retrieve | inference | p0 | m | RFC-0009 | #8 |
| #85 | inference: codingjepa.inference.rerank | inference | p0 | m | RFC-0009, RFC-0007 | #8 |
| #86 | inference: refusal logic | inference | p0 | s | spec/04, RFC-0007 | #8 |
| #87 | inference: codingjepa.inference.confidence | inference | p1 | s | RFC-0007, RFC-0009 | #8 |
| #88 | inference: tests/perf/test_latency.py | inference | p1 | m | RFC-0009, spec/08 | #8 |
| #89 | inference: tests/inference/test_round_trip.py | inference | p0 | s | RFC-0009 | #8 |
| #90 | inference: index_id contract enforcement | inference | p0 | s | RFC-0009, spec/04 | #8 |
| #91 | safety: side-effect-introduction checker | safety | p0 | m | RFC-0007 | #9 |
| #92 | safety: side-effect-elimination checker | safety | p0 | m | RFC-0007 | #9 |
| #93 | safety: exception-contract-change checker | safety | p0 | m | RFC-0007 | #9 |
| #94 | safety: public-api-change checker | safety | p0 | m | RFC-0007 | #9 |
| #95 | safety: async/sync-boundary-change checker | safety | p0 | s | RFC-0007 | #9 |
| #96 | safety: filter chain | safety | p0 | s | RFC-0007, spec/04 | #9 |
| #98 | safety: property test for the filter chain | safety | p0 | m | RFC-0007 | #9 |

## Milestone: v0.4 — demo

| # | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|-------|------|----------|--------|------------|----------|
| #100 | demo: codingjepa.demo.cli | demo | p1 | m | RFC-0006, spec/02 | #10 |
| #101 | demo: codingjepa.demo.web FastAPI app | demo | p1 | l | RFC-0006, RFC-0009, spec/02, spec/04 | #10 |
| #102 | demo: codingjepa.demo.diff renderer | demo | p1 | m | RFC-0006 | #10 |
| #103 | demo: HTMX templates + form | demo | p2 | s | RFC-0006 | #10 |
| #104 | demo: codingjepa.demo.messages | demo | p1 | s | RFC-0006, spec/04 | #10 |
| #105 | demo: deterministic example | demo | p1 | s | RFC-0006 | #10 |
| #106 | demo: hidden-step ban enforcement | demo | p1 | s | RFC-0006 | #10 |

## Milestone: v0.5 — eval + memo

| # | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|-------|------|----------|--------|------------|----------|
| #21 | ci: eval-smoke workflow | infra | p1 | s | RFC-0013, RFC-0010 | #3 |
| #23 | ci: nightly slow tests + ML diagnostics | infra | p2 | m | spec/07 | #3 |
| #107 | eval: codingjepa.eval.harness | eval | p0 | m | RFC-0010 | #11 |
| #108 | eval: CJ-RET-100 / CJ-RET-1k | eval | p0 | m | RFC-0010, RFC-0005 | #11 |
| #109 | eval: CJ-INTENT | eval | p0 | s | RFC-0010, RFC-0005 | #11 |
| #110 | eval: CJ-EXEC | eval | p0 | l | RFC-0010, RFC-0005 | #11 |
| #111 | eval: codingjepa.eval.sandbox | eval | p0 | l | RFC-0013, spec/06 | #11 |
| #112 | eval: CJ-ROB-FMT / RENAME / DOC | eval | p0 | m | RFC-0010, RFC-0005 | #11 |
| #113 | eval: CJ-OOD | eval | p1 | m | RFC-0010, RFC-0014 | #11 |
| #114 | eval: CJ-PROBE-NAME / DEFECT / CLONE | eval | p1 | l | RFC-0010 | #11 |
| #115 | eval: CJ-HUMAN | eval | p1 | m | RFC-0010, RFC-0005 | #11 |
| #116 | eval: codingjepa.eval.pools | eval | p0 | s | RFC-0010 | #11 |
| #117 | eval: codingjepa.eval.stats | eval | p0 | s | RFC-0005, RFC-0010 | #11 |
| #118 | eval: results.json schema + validator | eval | p0 | s | RFC-0010, spec/03 | #11 |
| #119 | eval: codingjepa.eval.memo | eval | p1 | m | RFC-0010 | #11 |
| #120 | eval: diff gallery + confusions HTML | eval | p2 | m | RFC-0010 | #11 |
| #122 | eval: hash-check enforcer | eval | p0 | s | RFC-0010, spec/04 | #11 |
| #123 | eval: 10-example fixture for eval-smoke | eval | p1 | m | RFC-0010 | #11 |

## Milestone: v1.0 — paper, release, packaging

| # | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|-------|------|----------|--------|------------|----------|
| #24 | infra: Dockerfile.eval | infra | p1 | m | RFC-0013 | #3 |
| #30 | infra: LICENSES/ + NOTICE generator | infra | p1 | s | RFC-0014 | #3 |
| #121 | eval: figures generator (PDFs) | eval | p1 | m | RFC-0010, RFC-0011 | #11 |
| #124 | release: paper/main.tex skeleton | release | p1 | s | RFC-0011 | #12 |
| #125 | release: MODEL_CARD.md template | release | p0 | s | RFC-0013, spec/03 | #12 |
| #126 | release: HF Hub upload (model + tokenizer) | release | p1 | m | RFC-0013 | #12 |
| #127 | release: HF Hub upload (pairs corpus) | release | p1 | m | RFC-0014 | #12 |
| #129 | release: PHASE-4.md / PHASE-5.md notes | release | p2 | s | spec/IMPLEMENTATION-PLAN | #12 |
| #130 | release: v1.0.0 release runbook | release | p1 | s | spec/09 | #12 |

## Cross-cutting dependencies

These are the high-impact dependencies that affect more than one subsystem.

- **#13 (pyproject) blocks everything.** Nothing else can install.
- **#16 (package skeleton) blocks every code-touching issue.** Stub modules let the rest stub against real import paths.
- **#25 (errors) blocks every issue that raises typed exceptions.** It is consumed by inference, training, eval, sandbox, schema validation.
- **#26 (observability) blocks every issue that emits structured logs** — training callbacks, inference, eval, demo.
- **#31 (JSONSchemas) blocks every issue that writes a persisted artifact.**
- **#40 (`codingjepa.intents.acceptance`) blocks every labeler (#41–#48), the inference rerank (#85), the eval scoring (in #110 and elsewhere), and the safety property tests.** Single source of truth contract is enforced by `tests/test_acceptance_singleton.py`.
- **#52 (audit) gates the data pipeline.** No training launches if any of `compile_ok_rate`, `duplication_rate`, `secret_scanner_hits` fail.
- **#53 (manifest) gates training and inference.** Both refuse to run without `data/manifest.lock.json`.
- **#74 (training preflight) gates Stage A.** Cheap-baselines-first (#81), tiny-slice (#65), tokenizer artifact (#37), manifest (#53), all verified before launch.
- **#75 (Stage A pretrain run) blocks #76 (Stage B), the FAISS index build (#83), and inference (#82–#86).**
- **#107 (eval harness) + #116 (pools) + #117 (stats) + #118 (results.schema) block every CJ-* benchmark (#108–#115).**
- **#122 (hash-check enforcer) gates `make eval`.** Refuses on drift.
- **#125 (MODEL_CARD) gates the v1.0 release** and the hash-check enforcer.

## Open questions surfaced by the spec

These remain `OPEN QUESTION`s in the corpus; they do not block v1 critical path but are tracked for visibility.

- **OQ-A.** Whether to ship a thin re-implementation of `stable_pretraining` rather than carrying it as a hard dep (RFC-0013 §D3). Decision lives in #67. **Owner: training subsystem.**
- **OQ-B.** Whether B4 (CodeT5+ embeddings) is included in v1 results. Currently optional (RFC-0005 §D5). **Owner: baselines subsystem (#7).**
- **OQ-C.** Whether `Devign` Python subset is available; if not, a synthetic mutation-defect set replaces it (RFC-0010 §E6 / #114). **Owner: eval subsystem (#11).**
- **OQ-D.** Whether `BigCloneBench` Python subset is available; if not, the clone-detection probe is reported as N/A (RFC-0010 §E6 / #114). **Owner: eval subsystem (#11).**
- **OQ-E.** Whether `nsjail` or `firejail` is the sandbox backend (#111). Either is acceptable per RFC-0013 §D7. **Owner: eval subsystem (#11).**
- **OQ-F.** Whether the demo UI vendors HTMX or pulls it from a CDN (#103 currently vendors per the supply-chain rule in spec/06). **Owner: demo subsystem (#10).**

## Notes for maintainers

- Labels (`type:*`, `area:*`, `priority:*`, `effort:*`, `spec:*`) were applied at issue-create time and auto-created on first use.
- Milestones (`v0.1` through `v1.0`) are recorded in each issue's `Metadata` block; create them in GitHub and re-assign the issues using this table.
- Sub-issue links (parent/child via the GitHub `sub_issue_write` API) were not applied in the bootstrap; the `Tracking` column links the dashboard issue.
- Re-deriving this tracker after an RFC amendment: change the affected RFC, re-derive the affected issues, append a row noting the amendment date.
