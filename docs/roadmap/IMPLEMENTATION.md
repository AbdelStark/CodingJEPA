# Implementation Tracker — updated 2026-05-15

Generated from the spec corpus. Every implementable unit of work in the spec is filed below.
Each issue is independently shippable; cross-issue dependencies are noted inline.

---

## Status legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Merged to `main`; GitHub issue closed |
| 🔲 | Open — not yet started |

---

## Completed (merged to `main` by 2026-05-15)

All infrastructure, data pipeline, model, training infrastructure, baselines,
safety, and inference pipeline issues have landed. GitHub issues are closed.

### Infrastructure (v0.1)
| # | Title | PR |
|---|-------|----|
| #13 | infra: initialize pyproject.toml | #132 |
| #14 | infra: generate uv.lock | #134 |
| #15 | infra: top-level Makefile | #135 |
| #16 | infra: scaffold codingjepa/ package skeleton | #133 |
| #18 | ci: lint workflow | #139 |
| #19 | ci: unit workflow | #140 |
| #20 | ci: safety workflow | #141 |
| #25 | infra: codingjepa.errors module | #136 |
| #26 | infra: codingjepa.observability | #137 |
| #27 | infra: CHANGELOG.md skeleton | #147 |
| #28 | docs: CONTRIBUTING.md + PR / issue templates | #148 |
| #29 | security: SECURITY.md | #146 |
| #31 | infra: data/schemas/ JSONSchemas | #138 |
| #32 | tests: cross-artifact invariants | #142 |
| #33 | infra: pre-commit hooks | #153 |
| #40 | intents: codingjepa.intents.acceptance | #143 |
| #97 | safety: stable refusal copy table (R001–R007) | #144 |
| #99 | safety: secret pattern table | #145 |

### Data pipeline (v0.1)
| # | Title | PR |
|---|-------|----|
| #34 | data: codingjepa.data.mirror | #156 |
| #35 | data: codingjepa.data.chunker | #158 |
| #36 | data: codingjepa.data.normalize | #157 |
| #37 | data: SentencePiece BPE tokenizer | #162 |
| #38 | data: tokenizer coverage audit | #162 |
| #39 | data: PyDriller commit walker | #159 |
| #41 | data: labeler — extract-helper | #161 |
| #42 | data: labeler — inline-helper | #161 |
| #43 | data: labeler — comprehension-rewrite | #161 |
| #44 | data: labeler — dataclass-migration | #161 |
| #45 | data: labeler — exception-handling-cleanup | #161 |
| #46 | data: labeler — loop-to-vectorized | #161 |
| #47 | data: labeler — argument-defaulting | #161 |
| #48 | data: labeler — none-typing-modernization | #161 |
| #49 | data: codingjepa.data.dedup | #163 |
| #50 | data: codingjepa.data.splits + leakage | #164 |
| #51 | data: codingjepa.data.secrets_scan | #164 |
| #52 | data: codingjepa.data.audit | #164 |
| #53 | data: codingjepa.data.manifest | #165 |
| #54 | data: codingjepa.data.sequences | #165 |
| #56 | data: codingjepa.data.cli wiring | #166 |
| #57 | data: per-intent quotas | #165 |

### Eval primitives (v0.1)
| # | Title | PR |
|---|-------|----|
| #116 | eval: codingjepa.eval.pools | #154 |
| #117 | eval: codingjepa.eval.stats | #155 |
| #118 | eval: results.json schema + validator | #149 |
| #122 | eval: hash-check enforcer | #150 |

### Model stack (v0.2)
| # | Title | PR |
|---|-------|----|
| #58 | model: codingjepa.modules.encoder | #167 |
| #59 | model: codingjepa.modules.projector | #167 |
| #60 | model: codingjepa.modules.ar_predictor | #167 |
| #61 | model: codingjepa.modules.pred_proj | #167 |
| #62 | model: codingjepa.modules.intent_embedder | #167 |
| #63 | model: codingjepa.modules.sigreg | #167 |
| #64 | model: codingjepa.model.CodingJEPA | #167 |
| #65 | model: tiny-slice training pass | #167 |

### Training infrastructure (v0.2)
| # | Title | PR |
|---|-------|----|
| #66 | train: codingjepa.training.module | #168 |
| #67 | train: codingjepa.training.manager | #168 |
| #68 | train: codingjepa.training.dataloader | #168 |
| #69 | train: codingjepa.training.optimizer | #168 |
| #70 | train: callbacks.RankDiagnostic | #168 |
| #71 | train: callbacks.LossMonotonicity | #168 |
| #72 | train: callbacks.Checkpoint | #168 |
| #73 | train: WandB integration | #168 |
| #74 | train: codingjepa.training.preflight | #168 |

### Baselines (v0.2)
| # | Title | PR |
|---|-------|----|
| #78 | baseline: B1 — BM25 over BPE tokens | #169 |
| #79 | baseline: B2 — MLM-encoder | #169 |
| #80 | baseline: B3 — frozen CodeBERT | #169 |
| #81 | baseline: cheap-baseline-first gate | #169 |

### Inference pipeline (v0.3)
| # | Title | PR |
|---|-------|----|
| #82 | inference: codingjepa.inference.embed | #171 |
| #83 | inference: codingjepa.inference.index | #171 |
| #84 | inference: codingjepa.inference.retrieve | #171 |
| #85 | inference: codingjepa.inference.rerank | #171 |
| #86 | inference: refusal logic | #171 |
| #87 | inference: codingjepa.inference.confidence | #171 |
| #88 | inference: tests/perf/test_latency.py | #171 |
| #89 | inference: tests/inference/test_round_trip.py | #171 |
| #90 | inference: index_id contract enforcement | #171 |

### Safety checkers (v0.3)
| # | Title | PR |
|---|-------|----|
| #91 | safety: side-effect-introduction checker | #170 |
| #92 | safety: side-effect-elimination checker | #170 |
| #93 | safety: exception-contract-change checker | #170 |
| #94 | safety: public-api-change checker | #170 |
| #95 | safety: async/sync-boundary-change checker | #170 |
| #96 | safety: filter chain | #170 |
| #98 | safety: property test for the filter chain | #170 |

### Demo (v0.4)
| # | Title | PR |
|---|-------|----|
| #100 | demo: codingjepa.demo.cli | #172 |
| #101 | demo: codingjepa.demo.web FastAPI app | #172 |
| #102 | demo: codingjepa.demo.diff renderer | #172 |
| #103 | demo: HTMX templates + form | #172 |
| #104 | demo: codingjepa.demo.messages | #172 |
| #105 | demo: deterministic example | #172 |
| #106 | demo: hidden-step ban enforcement | #172 |

### Release scaffolding
| # | Title | PR |
|---|-------|----|
| #125 | release: MODEL_CARD.md template | #151 |
| #130 | release: v1.0.0 release runbook | #152 |

---

### Eval harness (v0.5)
| # | Title | PR |
|---|-------|----|
| #107 | eval: codingjepa.eval.harness | #198 |
| #108 | eval: CJ-RET-100 / CJ-RET-1k | #198 |
| #109 | eval: CJ-INTENT | #198 |
| #110 | eval: CJ-EXEC (stub) | #198 |
| #111 | eval: codingjepa.eval.sandbox | #198 |
| #112 | eval: CJ-ROB-FMT / RENAME / DOC | #198 |
| #113 | eval: CJ-OOD | #198 |
| #114 | eval: CJ-PROBE-NAME / DEFECT / CLONE | #198 |
| #115 | eval: CJ-HUMAN (stub) | #198 |
| #119 | eval: codingjepa.eval.memo | #198 |
| #120 | eval: diff gallery + confusions HTML | #198 |
| #121 | eval: figures generator (PDFs) | #198 |
| #123 | eval: 10-example fixture for eval-smoke | #198 |

---

**Current test count: 898 passing, 0 failing.**

**Next priority:** Actual training runs (#75, #76), then release (#124–#129).

---

## Tracking issues (subsystem dashboards)

| # | Subsystem | Status | RFCs |
|---|---|---|---|
| #2 | [meta] tracker: spec corpus + implementation issue set bootstrap | open | — |
| #3 | [Tracking] Infrastructure & reproducibility | open (partial) | RFC-0013, spec/02–/09 |
| #4 | [Tracking] Data pipeline | open (#55 closed → #189) | RFC-0002, RFC-0004, RFC-0012, RFC-0014 |
| #5 | [Tracking] Model stack | ✅ closed | RFC-0003 |
| #6 | [Tracking] Training pipeline | open (#75, #76, #77 remain) | RFC-0008 |
| #7 | [Tracking] Baselines | ✅ closed | RFC-0005 |
| #8 | [Tracking] Inference pipeline | ✅ closed | RFC-0009, RFC-0007 |
| #9 | [Tracking] Safety rails | ✅ closed | RFC-0007 |
| #10 | [Tracking] Demo & developer workflow | ✅ closed | RFC-0006, RFC-0009 |
| #11 | [Tracking] Evaluation harness | ✅ closed | RFC-0010, RFC-0005 |
| #12 | [Tracking] Paper, release, packaging | open | RFC-0011, RFC-0013, RFC-0014, spec/09 |
| #173 | [Tracking] Dataset: human-python-refactors — curation, quality, HF publication | open | RFC-0002, RFC-0014, RFC-0015 |

## Milestone: v0.1 — data pipeline frozen + baselines + infrastructure ready

| # | Status | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|--------|-------|------|----------|--------|------------|----------|
| #13 | ✅ | infra: initialize pyproject.toml | infra | p0 | s | RFC-0013 | #3 |
| #14 | ✅ | infra: generate uv.lock; require uv sync --frozen in CI | infra | p0 | s | RFC-0013, spec/06 | #3 |
| #15 | ✅ | infra: top-level Makefile | infra | p0 | s | RFC-0006, spec/02 | #3 |
| #16 | ✅ | infra: scaffold codingjepa/ package skeleton | infra | p0 | m | spec/01, spec/02 | #3 |
| #18 | ✅ | ci: lint workflow | infra | p0 | s | RFC-0013, spec/07 | #3 |
| #19 | ✅ | ci: unit workflow | infra | p0 | s | RFC-0013, spec/07 | #3 |
| #20 | ✅ | ci: safety workflow | infra | p0 | s | RFC-0013, RFC-0007, spec/06 | #3 |
| #25 | ✅ | infra: codingjepa.errors module | infra | p0 | s | spec/04 | #3 |
| #26 | ✅ | infra: codingjepa.observability | infra | p0 | m | spec/05, spec/06 | #3 |
| #27 | ✅ | infra: CHANGELOG.md skeleton | infra | p1 | s | spec/09 | #3 |
| #28 | ✅ | docs: CONTRIBUTING.md + PR / issue templates | infra | p1 | s | spec/09, spec/07 | #3 |
| #29 | ✅ | security: SECURITY.md | infra | p1 | s | spec/06 | #3 |
| #31 | ✅ | infra: data/schemas/ JSONSchemas | infra | p0 | m | spec/03 | #3 |
| #32 | ✅ | tests: cross-artifact invariants | infra | p0 | m | spec/03 | #3 |
| #33 | ✅ | infra: pre-commit hooks | infra | p2 | s | spec/07 | #3 |
| #34 | ✅ | data: codingjepa.data.mirror | data | p0 | m | RFC-0002, RFC-0014 | #4 |
| #35 | ✅ | data: codingjepa.data.chunker | data | p0 | l | RFC-0012, spec/03 | #4 |
| #36 | ✅ | data: codingjepa.data.normalize | data | p0 | m | RFC-0012 | #4 |
| #37 | ✅ | data: SentencePiece BPE tokenizer training | data | p0 | m | RFC-0012 | #4 |
| #38 | ✅ | data: tokenizer coverage audit | data | p0 | s | RFC-0012 | #4 |
| #39 | ✅ | data: PyDriller commit walker for refactor pairs | data | p0 | m | RFC-0002 | #4 |
| #40 | ✅ | intents: codingjepa.intents.acceptance | data | p0 | l | RFC-0004, spec/01 | #4 |
| #41 | ✅ | data: labeler — extract-helper | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #42 | ✅ | data: labeler — inline-helper | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #43 | ✅ | data: labeler — comprehension-rewrite | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #44 | ✅ | data: labeler — dataclass-migration | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #45 | ✅ | data: labeler — exception-handling-cleanup | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #46 | ✅ | data: labeler — loop-to-vectorized | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #47 | ✅ | data: labeler — argument-defaulting | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #48 | ✅ | data: labeler — none-typing-modernization | data | p1 | m | RFC-0002, RFC-0004 | #4 |
| #49 | ✅ | data: codingjepa.data.dedup | data | p0 | m | RFC-0014 | #4 |
| #50 | ✅ | data: codingjepa.data.splits + cross-split leakage | data | p0 | m | RFC-0014 | #4 |
| #51 | ✅ | data: codingjepa.data.secrets_scan | data | p0 | m | RFC-0014, spec/06 | #4 |
| #52 | ✅ | data: codingjepa.data.audit | data | p0 | m | RFC-0002, RFC-0014 | #4 |
| #53 | ✅ | data: codingjepa.data.manifest | data | p0 | s | RFC-0014, spec/03 | #4 |
| #54 | ✅ | data: codingjepa.data.sequences | data | p0 | s | RFC-0002, RFC-0012 | #4 |
| #55 | 🔲 | data: gold subset annotation tooling + 200 pairs | data | p1 | l | RFC-0002 | #4 |
| #56 | ✅ | data: codingjepa.data.cli wiring | data | p0 | s | spec/02 | #4 |
| #57 | ✅ | data: per-intent quotas | data | p0 | s | RFC-0002 | #4 |
| #78 | ✅ | baseline: B1 — BM25 over BPE tokens | baselines | p0 | m | RFC-0005 | #7 |
| #79 | ✅ | baseline: B2 — MLM-encoder | baselines | p0 | l | RFC-0005 | #7 |
| #80 | ✅ | baseline: B3 — frozen CodeBERT | baselines | p0 | m | RFC-0005, spec/06 | #7 |
| #81 | ✅ | baseline: cheap-baseline-first gate | baselines | p0 | s | RFC-0005 | #7 |
| #97 | ✅ | safety: stable refusal copy table | safety | p0 | s | RFC-0007, spec/04 | #9 |
| #99 | ✅ | safety: secret pattern table | safety | p1 | s | spec/05, spec/06 | #9 |
| #128 | 🔲 | release: PHASE-1.md note (data freeze) | release | p1 | s | spec/IMPLEMENTATION-PLAN | #12 |

## Milestone: v0.2 — model stack + Stage A pretrain

| # | Status | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|--------|-------|------|----------|--------|------------|----------|
| #17 | 🔲 | infra: Hydra config tree skeleton | infra | p1 | m | RFC-0008 | #3 |
| #58 | ✅ | model: codingjepa.modules.encoder | model | p0 | m | RFC-0003 | #5 |
| #59 | ✅ | model: codingjepa.modules.projector | model | p0 | s | RFC-0003 | #5 |
| #60 | ✅ | model: codingjepa.modules.ar_predictor | model | p0 | m | RFC-0003 | #5 |
| #61 | ✅ | model: codingjepa.modules.pred_proj | model | p0 | s | RFC-0003 | #5 |
| #62 | ✅ | model: codingjepa.modules.intent_embedder | model | p0 | s | RFC-0003 | #5 |
| #63 | ✅ | model: codingjepa.modules.sigreg | model | p0 | m | RFC-0003 | #5 |
| #64 | ✅ | model: codingjepa.model.CodingJEPA | model | p0 | m | RFC-0003, RFC-0008, spec/02 | #5 |
| #65 | ✅ | model: tiny-slice training pass | model | p0 | m | RFC-0003, RFC-0008 | #5 |
| #66 | ✅ | train: codingjepa.training.module | training | p0 | m | RFC-0008 | #6 |
| #67 | ✅ | train: codingjepa.training.manager | training | p0 | l | RFC-0008 | #6 |
| #68 | ✅ | train: codingjepa.training.dataloader | training | p0 | m | RFC-0008 | #6 |
| #69 | ✅ | train: codingjepa.training.optimizer | training | p0 | s | RFC-0008 | #6 |
| #70 | ✅ | train: callbacks.RankDiagnostic | training | p0 | s | RFC-0008, spec/04 | #6 |
| #71 | ✅ | train: callbacks.LossMonotonicity | training | p0 | s | RFC-0008, spec/04 | #6 |
| #72 | ✅ | train: callbacks.Checkpoint | training | p0 | s | RFC-0008 | #6 |
| #73 | ✅ | train: WandB integration | training | p1 | s | RFC-0008, spec/06 | #6 |
| #74 | ✅ | train: codingjepa.training.preflight | training | p0 | s | RFC-0008 | #6 |
| #75 | 🔲 | train: Stage A pretrain run (200k steps, B=64) | training | p0 | l | RFC-0008 | #6 |
| #79 | ✅ | baseline: B2 — MLM-encoder | baselines | p0 | l | RFC-0005 | #7 |

## Milestone: v0.3 — fine-tune + retrieval + safety + inference

| # | Status | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|--------|-------|------|----------|--------|------------|----------|
| #22 | 🔲 | ci: perf workflow with regression gate | infra | p1 | m | RFC-0009, spec/08 | #3 |
| #76 | 🔲 | train: Stage B intent fine-tune run | training | p0 | l | RFC-0008 | #6 |
| #77 | 🔲 | docs: PHASE-2.md and PHASE-3.md notes | training | p1 | s | spec/IMPLEMENTATION-PLAN | #6 |
| #82 | ✅ | inference: codingjepa.inference.embed | inference | p0 | s | RFC-0009 | #8 |
| #83 | ✅ | inference: codingjepa.inference.index | inference | p0 | m | RFC-0009, spec/03 | #8 |
| #84 | ✅ | inference: codingjepa.inference.retrieve | inference | p0 | m | RFC-0009 | #8 |
| #85 | ✅ | inference: codingjepa.inference.rerank | inference | p0 | m | RFC-0009, RFC-0007 | #8 |
| #86 | ✅ | inference: refusal logic | inference | p0 | s | spec/04, RFC-0007 | #8 |
| #87 | ✅ | inference: codingjepa.inference.confidence | inference | p1 | s | RFC-0007, RFC-0009 | #8 |
| #88 | ✅ | inference: tests/perf/test_latency.py | inference | p1 | m | RFC-0009, spec/08 | #8 |
| #89 | ✅ | inference: tests/inference/test_round_trip.py | inference | p0 | s | RFC-0009 | #8 |
| #90 | ✅ | inference: index_id contract enforcement | inference | p0 | s | RFC-0009, spec/04 | #8 |
| #91 | ✅ | safety: side-effect-introduction checker | safety | p0 | m | RFC-0007 | #9 |
| #92 | ✅ | safety: side-effect-elimination checker | safety | p0 | m | RFC-0007 | #9 |
| #93 | ✅ | safety: exception-contract-change checker | safety | p0 | m | RFC-0007 | #9 |
| #94 | ✅ | safety: public-api-change checker | safety | p0 | m | RFC-0007 | #9 |
| #95 | ✅ | safety: async/sync-boundary-change checker | safety | p0 | s | RFC-0007 | #9 |
| #96 | ✅ | safety: filter chain | safety | p0 | s | RFC-0007, spec/04 | #9 |
| #98 | ✅ | safety: property test for the filter chain | safety | p0 | m | RFC-0007 | #9 |

## Milestone: v0.4 — demo

| # | Status | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|--------|-------|------|----------|--------|------------|----------|
| #100 | ✅ | demo: codingjepa.demo.cli | demo | p1 | m | RFC-0006, spec/02 | #10 |
| #101 | ✅ | demo: codingjepa.demo.web FastAPI app | demo | p1 | l | RFC-0006, RFC-0009, spec/02, spec/04 | #10 |
| #102 | ✅ | demo: codingjepa.demo.diff renderer | demo | p1 | m | RFC-0006 | #10 |
| #103 | ✅ | demo: HTMX templates + form | demo | p2 | s | RFC-0006 | #10 |
| #104 | ✅ | demo: codingjepa.demo.messages | demo | p1 | s | RFC-0006, spec/04 | #10 |
| #105 | ✅ | demo: deterministic example | demo | p1 | s | RFC-0006 | #10 |
| #106 | ✅ | demo: hidden-step ban enforcement | demo | p1 | s | RFC-0006 | #10 |

## Milestone: v0.5 — eval + memo

| # | Status | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|--------|-------|------|----------|--------|------------|----------|
| #21 | 🔲 | ci: eval-smoke workflow | infra | p1 | s | RFC-0013, RFC-0010 | #3 |
| #23 | 🔲 | ci: nightly slow tests + ML diagnostics | infra | p2 | m | spec/07 | #3 |
| #107 | ✅ | eval: codingjepa.eval.harness | eval | p0 | m | RFC-0010 | #11 |
| #108 | ✅ | eval: CJ-RET-100 / CJ-RET-1k | eval | p0 | m | RFC-0010, RFC-0005 | #11 |
| #109 | ✅ | eval: CJ-INTENT | eval | p0 | s | RFC-0010, RFC-0005 | #11 |
| #110 | ✅ | eval: CJ-EXEC (stub) | eval | p0 | l | RFC-0010, RFC-0005 | #11 |
| #111 | ✅ | eval: codingjepa.eval.sandbox | eval | p0 | l | RFC-0013, spec/06 | #11 |
| #112 | ✅ | eval: CJ-ROB-FMT / RENAME / DOC | eval | p0 | m | RFC-0010, RFC-0005 | #11 |
| #113 | ✅ | eval: CJ-OOD | eval | p1 | m | RFC-0010, RFC-0014 | #11 |
| #114 | ✅ | eval: CJ-PROBE-NAME / DEFECT / CLONE | eval | p1 | l | RFC-0010 | #11 |
| #115 | ✅ | eval: CJ-HUMAN (stub) | eval | p1 | m | RFC-0010, RFC-0005 | #11 |
| #116 | ✅ | eval: codingjepa.eval.pools | eval | p0 | s | RFC-0010 | #11 |
| #117 | ✅ | eval: codingjepa.eval.stats | eval | p0 | s | RFC-0005, RFC-0010 | #11 |
| #118 | ✅ | eval: results.json schema + validator | eval | p0 | s | RFC-0010, spec/03 | #11 |
| #119 | ✅ | eval: codingjepa.eval.memo | eval | p1 | m | RFC-0010 | #11 |
| #120 | ✅ | eval: diff gallery + confusions HTML | eval | p2 | m | RFC-0010 | #11 |
| #121 | ✅ | eval: figures generator (PDFs) | eval | p1 | m | RFC-0010, RFC-0011 | #11 |
| #122 | ✅ | eval: hash-check enforcer | eval | p0 | s | RFC-0010, spec/04 | #11 |
| #123 | ✅ | eval: 10-example fixture for eval-smoke | eval | p1 | m | RFC-0010 | #11 |

## Milestone: v1.0 — paper, release, packaging

| # | Status | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|--------|-------|------|----------|--------|------------|----------|
| #24 | 🔲 | infra: Dockerfile.eval | infra | p1 | m | RFC-0013 | #3 |
| #30 | 🔲 | infra: LICENSES/ + NOTICE generator | infra | p1 | s | RFC-0014 | #3, #173 |
| #124 | 🔲 | release: paper/main.tex skeleton | release | p1 | s | RFC-0011 | #12 |
| #125 | ✅ | release: MODEL_CARD.md template | release | p0 | s | RFC-0013, spec/03 | #12 |
| #126 | 🔲 | release: HF Hub upload (model + tokenizer) | release | p1 | m | RFC-0013 | #12 |
| #127 | 🔲 | release: HF Hub upload (pairs corpus) — superseded by #173 stream | release | p1 | m | RFC-0014, RFC-0015 | #173 |
| #128 | 🔲 | release: PHASE-1.md note (data freeze) | release | p1 | s | spec/IMPLEMENTATION-PLAN | #12 |
| #129 | 🔲 | release: PHASE-4.md / PHASE-5.md notes | release | p2 | s | spec/IMPLEMENTATION-PLAN | #12 |
| #130 | ✅ | release: v1.0.0 release runbook | release | p1 | s | spec/09 | #12 |

## Milestone: dataset-v1.0 — human-python-refactors corpus and HF publication

| # | Status | Title | Area | Priority | Effort | RFC / Spec | Tracking |
|---|--------|-------|------|----------|--------|------------|----------|
| #174 | 🔲 | data: wire --cutoff 2023-12-31 into pairs.py and data CLI | data | p0 | s | RFC-0002, RFC-0015 | #173 |
| #175 | 🔲 | data: add commit_cutoff_utc to manifest JSON schema | data | p0 | s | RFC-0002, RFC-0015 | #173 |
| #176 | 🔲 | tools: hf_convert.py — DatasetDict builder + HF push script | release | p0 | m | RFC-0015 | #173 |
| #177 | 🔲 | tools: secret-scan assertion helpers | data | p0 | s | RFC-0014, RFC-0015 | #173 |
| #178 | 🔲 | data: Step 1 — mirror all 10 repos at pinned SHAs | data | p0 | m | RFC-0002, RFC-0015 | #173 |
| #179 | 🔲 | data: Step 2 — normalize all repo source files | data | p0 | m | RFC-0012, RFC-0014 | #173 |
| #180 | 🔲 | data: Step 3 — chunk all repos → data/chunks/*.parquet | data | p0 | m | RFC-0012, RFC-0015 | #173 |
| #181 | 🔲 | data: Step 4 — extract refactor pairs (--cutoff 2023-12-31) | data | p0 | l | RFC-0002, RFC-0015 | #173 |
| #182 | 🔲 | data: Step 5 — apply 8 heuristic labelers + enforce 12k/intent quota | data | p0 | m | RFC-0002, RFC-0015 | #173 |
| #183 | 🔲 | data: Step 6 — secret scan over pairs corpus, zero-hit gate | data | p0 | s | RFC-0014, RFC-0015 | #173 |
| #184 | 🔲 | data: Step 7 — deduplication (exact + MinHash LSH near-dedup) | data | p0 | m | RFC-0014, RFC-0015 | #173 |
| #185 | 🔲 | data: Step 8 — assign train/val/test splits by-repository | data | p0 | s | RFC-0014, RFC-0015 | #173 |
| #186 | 🔲 | data: Step 9 — full corpus audit, all 4 gates must pass | data | p0 | m | RFC-0002, RFC-0014 | #173 |
| #187 | 🔲 | data: Step 10 — write content-addressed manifest.lock.json and commit | data | p0 | s | RFC-0014, RFC-0015 | #173 |
| #188 | 🔲 | data: corpus QA report — statistics, intent distribution, pair counts | data | p1 | m | RFC-0015 | #173 |
| #189 | 🔲 | data: gold subset annotation — 200 pairs, Cohen's κ ≥ 0.7 (closes #55) | data | p1 | l | RFC-0002, RFC-0015 | #173 |
| #190 | 🔲 | release: write data/DATASET_CARD.md from RFC-0015 template | release | p1 | s | RFC-0015 | #173 |
| #191 | 🔲 | release: dry-run hf_convert.py — validate HF schema on corpus sample | release | p1 | s | RFC-0015 | #173 |
| #192 | 🔲 | release: upload pairs config to CodingJEPA/human-python-refactors | release | p1 | m | RFC-0015 | #173 |
| #193 | 🔲 | release: upload chunks config to CodingJEPA/human-python-refactors | release | p1 | m | RFC-0015 | #173 |
| #194 | 🔲 | release: upload NOTICE + LICENSES/ to HF dataset repo | release | p1 | s | RFC-0014, RFC-0015 | #173 |
| #195 | 🔲 | release: publish dataset card (README.md) to HF | release | p1 | s | RFC-0015 | #173 |
| #196 | 🔲 | release: verify public HF access — load_dataset() roundtrip smoke test | release | p0 | s | RFC-0015 | #173 |
| #197 | 🔲 | release: create GitHub tag data-v1.0.0 at the manifest commit | release | p1 | s | RFC-0015 | #173 |

## Cross-cutting dependencies

✅ = already resolved on `main`.

- **✅ #13 (pyproject) blocks everything.**
- **✅ #16 (package skeleton) blocks every code-touching issue.**
- **✅ #25 (errors) blocks every issue that raises typed exceptions.**
- **✅ #26 (observability) blocks every issue that emits structured logs.**
- **✅ #31 (JSONSchemas) blocks every issue that writes a persisted artifact.**
- **✅ #40 (`codingjepa.intents.acceptance`) blocks labelers, rerank, eval scoring, safety property tests.**
- **✅ #52 (audit) gates the data pipeline.**
- **✅ #53 (manifest) gates training and inference.**
- **✅ #74 (training preflight) gates Stage A** — checks baselines-first, tiny-slice, tokenizer, manifest.
- **#75 (Stage A pretrain run) blocks #76 (Stage B), the FAISS index build, and live inference.**
- **✅ #116 + #117 + #118 + #122 unblock every CJ-* benchmark (#108–#115)** — still need #107 (eval harness).
- **✅ #125 (MODEL_CARD) gates the v1.0 release.**

## Open questions surfaced by the spec

- **OQ-A.** `stable_pretraining` dep vs. thin re-implementation (RFC-0013 §D3). Owner: training (#6).
- **OQ-B.** Whether B4 (CodeT5+ embeddings) is included in v1 results. Currently optional. Owner: #7.
- **OQ-C.** Whether `Devign` Python subset is available. Owner: eval (#11).
- **OQ-D.** Whether `BigCloneBench` Python subset is available. Owner: eval (#11).
- **OQ-E.** `nsjail` vs. `firejail` sandbox backend (#111). Owner: eval (#11).
- **OQ-F.** HTMX vendored vs. CDN (#103). Owner: demo (#10).

## Notes for maintainers

- Labels (`type:*`, `area:*`, `priority:*`, `effort:*`, `spec:*`) were applied at issue-create time.
- Sub-issue links (parent/child) were not applied in the bootstrap; the `Tracking` column links the dashboard issue.
- Re-deriving this tracker after an RFC amendment: change the affected RFC, re-derive the affected issues, append a row noting the amendment date.
