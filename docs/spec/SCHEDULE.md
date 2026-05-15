# CodingJEPA — schedule

## Milestone plan (10 weeks)

| Week | Milestone | Output artifact | Validation |
|---|---|---|---|
| 0 | Spec stack locked | RFCs 0001–0014 `Status: Locked` | All RFCs reviewed |
| 1–2 | Data + baselines | `data/manifest.lock.json`, `data/pairs/v1.parquet`, baseline metrics JSON | Baselines run end-to-end on gold subset |
| 3–5 | Pretraining | `runs/pretrain-v1/checkpoint.ckpt`, rank metrics | Embedding rank ≥ 0.9 × dim; SIGReg converged |
| 6–7 | Fine-tune + retrieval | `runs/finetune-v1/checkpoint.ckpt`, FAISS index, retrieval metrics | PRD §8 thresholds met |
| 8 | Demo surface | CLI + web UI, `examples/` | `make demo` works from clean clone |
| 9 | Eval + memo | `results/RESULTS-MEMO.md`, ablation tables | `make eval` reproduces all numbers |
| 10 | Paper + release | Paper PDF, `v1.0` tag, public checkpoints | Numbers traceable to released artifacts |

## Operating rule

Each milestone closes with:
- one explicit output artifact committed to the repo,
- one validation step (a script that returns non-zero on regression),
- one written note (`docs/notes/PHASE-N.md`) about what changed in the project understanding.

## Budget

- **Compute:** ≤ 7 days × 1×H100 for pretraining + ≤ 24h for fine-tune + ≤ 24h for all eval runs.
- **Storage:** ≤ 200 GB for raw + parsed + checkpoints + indices.
- **Wall-clock for one researcher:** ~10 weeks, full-time.

## Risk reserves

- 1 week reserved between Phase 2 and Phase 3 for unscheduled debugging.
- 1 week reserved at end of Phase 5 for paper revisions.
- If retrieval@10 fails to meet PRD threshold at end of Phase 3, halt; re-open RFC-0003 and RFC-0008 rather than proceeding.
