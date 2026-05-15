# Phase 5 — Evaluation Harness Notes

> Status: **complete** (2026-05-15, PR #198).

## What landed

Phase 5 (eval harness) is fully implemented per RFC-0010. All 13 issues (#107–#123) are closed.

| Component | Module | Notes |
|-----------|--------|-------|
| Benchmark base class | `codingjepa.eval.harness` | `Benchmark` ABC + `run_suite` orchestrator |
| CJ-RET-100 / CJ-RET-1k | `codingjepa.eval.benchmarks.ret` | FAISS IndexFlatIP, R@1/R@5/R@10/MRR |
| CJ-INTENT | `codingjepa.eval.benchmarks.intent` | Conditioned vs. unconditional R@5 |
| CJ-EXEC | `codingjepa.eval.benchmarks.exec` | Stub; returns `no_executable_pairs` until sandboxed data present |
| Sandbox | `codingjepa.eval.sandbox` | nsjail/firejail/plain subprocess |
| CJ-ROB-* | `codingjepa.eval.benchmarks.robustness` | FMT/RENAME/DOC: rank_change_pct + cosine_drift |
| CJ-OOD | `codingjepa.eval.benchmarks.ood` | R@10 on 200-pair pool |
| CJ-PROBE-* | `codingjepa.eval.benchmarks.probes` | NAME/DEFECT/CLONE linear probes (sklearn-optional) |
| CJ-HUMAN | `codingjepa.eval.benchmarks.human` | Stub; returns `no_human_annotations` until annotation file present |
| RESULTS-MEMO generator | `codingjepa.eval.memo` | All 11 RFC-0010 §D6 sections |
| Diff gallery | `codingjepa.eval.diff_gallery` | HTML per-pair diff pages + index.html |
| Confusions | `codingjepa.eval.confusions` | Worst-50 error pages per intent |
| Figures | `codingjepa.eval.figures` | matplotlib PDFs (graceful no-op if matplotlib absent) |
| Smoke fixture | `tests/eval/test_harness.py` | 52 tests on 10-example fixture |

## What's pending

- **CJ-EXEC**: The execution-preservation benchmark returns a stub result until real sandboxed pairs are provided (`exec_pairs.jsonl` in the data directory). This requires the actual data pipeline to run end-to-end.
- **CJ-HUMAN**: Returns a stub until a human-annotation file is present. Annotation tooling is in #189.
- **Real numbers**: All benchmarks produce valid JSON output but metrics are computed on synthetic data. Meaningful numbers require a trained checkpoint and the real eval pools.
- **`make eval` CLI wiring**: The `codingjepa eval` CLI subcommand is not yet wired; `make eval` currently calls `python -m codingjepa eval`. This is tracked for the release milestone.

## Statistical reporting (RFC-0010 §D7)

When reporting results:
- Run ≥ 3 seeds per number.
- Report 95% CI via 1000-resample bootstrap (`codingjepa.eval.stats.bootstrap_ci`).
- Report paired-bootstrap p-values for "CodingJEPA beats baseline X" claims; p < 0.05 required.

## Known limitations

- sklearn is optional: `CJ-PROBE-*` benchmarks fall back to a stub result if sklearn is not installed.
- matplotlib is optional: figures are not generated if matplotlib is not installed.
- The sandbox runner defaults to `plain` subprocess (no nsjail/firejail) on machines without those tools. This is safe for CI but not for untrusted code.
