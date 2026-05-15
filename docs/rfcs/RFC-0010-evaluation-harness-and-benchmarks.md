# RFC-0010 — Evaluation harness and benchmarks

## Status
Locked (2026-05-15)

## Problem

Specify the mechanics of the evaluation harness: the data, the metrics computed, the benchmark suite, the report format, and the "single command reproduces every number" rule.

This RFC is the operational companion to RFC-0005 (which specifies *what* counts as a valid measurement).

## Decisions locked

### D1 — Single-command rule
`make eval` regenerates `results/results.json` and `results/RESULTS-MEMO.md` from:
- the released checkpoint hash (recorded in `MODEL_CARD.md`);
- the data manifest hash (in `data/manifest.lock.json`);
- the index hash (in `indices/v1.meta.json`).

If any of those hashes drifts, `make eval` refuses to run and prints a clear diff.

### D2 — Evaluation datasets

**E1 — Held-out refactor pairs.**
- Source: the 2 test repos (`psf/black`, `python/cpython:Lib/`).
- ~12,000 pairs across 8 intents (heuristic-labeled).
- Used for Retrieval@k, MRR, intent-conditioned hit rate, per-intent breakdown.

**E2 — Gold subset.**
- 200 hand-curated pairs, stratified 25 per intent.
- Used for human-review rubric (RFC-0005 §D8) and the diff gallery.

**E3 — Executable subset.**
- ~500 pairs whose module has runnable pytest coverage.
- Used for execution-preservation pass-rate.

**E4 — Robustness probe set.**
- 1,000 chunks (stratified, intent-agnostic).
- Each chunk is perturbed by 3 perturbers (formatting, alpha-rename, docstring); 4,000 evaluations.

**E5 — OOD probe.**
- 200 hand-curated pairs from cpython `Lib/` refactors during the 3.11 → 3.13 dev cycle.
- Held out from all training. Used only at final evaluation.

**E6 — Code understanding (linear probe).**
- Function-name prediction from body (test set carved from the val repos).
- Defect detection (Devign Python subset, if available; otherwise a synthetic mutation-defect set).
- Clone detection (BigCloneBench Python subset, if available).

### D3 — The benchmark suite

The full suite, run in this order by `make eval`:

| ID | Name | Dataset | Metric |
|---|---|---|---|
| CJ-RET-100 | Retrieval@k, pool=100 | E1 | R@1, R@5, R@10, MRR |
| CJ-RET-1k | Retrieval@k, pool=1000 | E1 | R@1, R@5, R@10, MRR |
| CJ-INTENT | Intent-conditioned hit rate | E1 | R@5 with vs. without intent |
| CJ-EXEC | Execution preservation | E3 | Pass rate |
| CJ-ROB-FMT | Formatting invariance | E4 | rank-change %, cosine drift |
| CJ-ROB-RENAME | Alpha-rename invariance | E4 | rank-change %, cosine drift |
| CJ-ROB-DOC | Docstring invariance | E4 | rank-change %, cosine drift |
| CJ-OOD | OOD retrieval | E5 | R@10 |
| CJ-PROBE-NAME | Function-name probe | E6 | Top-1 accuracy |
| CJ-PROBE-DEFECT | Defect detection probe | E6 | F1 |
| CJ-PROBE-CLONE | Clone-detection probe | E6 | F1 |
| CJ-HUMAN | Human review | E2 | Mean Likert per axis, κ |

### D4 — Baselines on every benchmark
- B1 BM25, B2 MLM-encoder, B3 CodeBERT-frozen run on every CJ-* benchmark, with identical pool, tokenization, and split.
- Results table in `results/RESULTS-MEMO.md` is wide-format: rows = benchmarks, columns = system × metric.

### D5 — Ablations
The ablation table covers the matrix from RFC-0005 §D6. Each cell is mean ± std over ≥ 3 seeds.

### D6 — Report format

`results/RESULTS-MEMO.md` contains, in order:
1. **One-paragraph TL;DR.**
2. **Setup table** (model size, training compute, data size, tokenizer).
3. **Main results table** (CJ-RET-100 + CJ-RET-1k + CJ-EXEC).
4. **Per-intent breakdown** (8 rows × Retrieval@10).
5. **Robustness probes** (CJ-ROB-*).
6. **OOD probe** (CJ-OOD).
7. **Code understanding** (CJ-PROBE-*).
8. **Human review** (CJ-HUMAN).
9. **Ablations** (matrix from RFC-0005 §D6).
10. **Failure modes** (worst-50 retrievals per intent; embedded HTML).
11. **Limitations** (corpus contamination risks, OOD scope, statistical caveats).

The memo is at most 8 pages when rendered. Numbers beyond that go in `results/results.json`.

### D7 — Statistical reporting
- ≥ 3 seeds per number.
- 95% CI via 1000-resample bootstrap.
- Paired-bootstrap p-values for "CodingJEPA beats baseline X" claims; p < 0.05 required.

### D8 — Harness implementation
- `codingjepa.eval.harness` orchestrates each benchmark.
- Each benchmark is a `Benchmark` subclass with `prepare`, `run`, `score` methods.
- Outputs a per-benchmark JSON; aggregator stitches them into `results/results.json`.
- Determinism: every benchmark sets its own seed derived from a global seed + benchmark name (so changing one benchmark doesn't shuffle the others).

### D9 — Pool construction (deterministic)
For each retrieval benchmark we record the exact pool composition (chunk hashes) in `eval/pools/<benchmark>.lock.json`. The pool is content-addressed; an accidental change re-versions the benchmark.

### D10 — Cost of one eval run
- E1 (12k pairs × encode + retrieve over 100/1k pool): ~30 min on H100.
- E3 (500 sandboxed pytest runs): ~2h.
- E4 (4k embeddings): ~10 min.
- E6 (linear probes): ~30 min.
- Total: ≤ 4h wall-clock per system.

### D11 — Avoiding contamination
- The test repos (`psf/black`, `python/cpython:Lib/`) never appear in training corpus.
- MinHash LSH dedup is applied across all splits before any training run (RFC-0014).
- Public pretrained baselines (CodeBERT) are noted to be trained on overlapping data; we document this explicitly in the memo's limitations section.

### D12 — Outputs
- `results/results.json` (all numbers).
- `results/RESULTS-MEMO.md` (the human-readable memo).
- `results/diffs/` (HTML diff gallery for the gold subset).
- `results/confusions/` (HTML pages for worst-50 errors per intent).
- `results/figures/` (PDFs for the paper).

## Deferred items
- Cross-corpus generalization (train on this set, eval on a held-out repo from outside the 10).
- Calibration curves and Brier scores on the confidence signal.
- Inference latency as a benchmark (currently in `tests/perf/`).

## Acceptance condition

Locked when:
- `make eval` runs end-to-end on a stub checkpoint with no errors and emits all expected outputs;
- the `results.json` schema is committed and `pytest` validates a sample against it;
- each benchmark has at least one unit test on a 10-example fixture.
