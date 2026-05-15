# Phase 1 — Data Pipeline Notes

> Status: **frozen** (2026-05-15). These notes document decisions and findings from the data pipeline build-out.

## Summary

The Phase 1 data pipeline is fully implemented and frozen at the `v0.1` tag. This note captures the key findings, statistics, and assumptions that are relevant for reproducibility and for interpreting the evaluation results.

## Corpus summary

| Metric | Value |
|--------|-------|
| Source repositories | 10 (see `codingjepa.data.mirror.REPO_REGISTRY`) |
| Commit cutoff | 2023-12-31 23:59:59 UTC (RFC-0002 §D11) |
| Intents covered | 8 (extract-helper, inline-helper, comprehension-rewrite, dataclass-migration, exception-handling-cleanup, loop-to-vectorized, argument-defaulting, none-typing-modernization) |
| Pairs per intent (train cap) | 12,000 |
| BPE vocabulary | 32,000 tokens + 15 special tokens |

## Audit gates

All four hard gates (RFC-0014 §D10) must pass before the manifest is written:

| Gate | Threshold | Status |
|------|-----------|--------|
| Compile rate | ≥ 0.95 | Passes (all chunks pass `compile()`) |
| Dedup rate | < 0.30 | Passes (MinHash LSH near-dedup Jaccard ≥ 0.85, 128 functions, 32 bands) |
| License | ∈ allowed set | Passes (all 10 repos are MIT/Apache-2.0/PSF) |
| Secrets | == 0 | Passes (full regex scan; see `codingjepa.safety.secret_patterns`) |

## Leakage audit

Cross-split leakage is detected by MinHash Jaccard similarity between the train/val/test splits. Assignment is by repository (per RFC-0014 §D7): the two held-out test repos (`psf/black`, `python/cpython:Lib/`) are never in training.

## Deduplication

- Exact dedup: SHA-256 chunk IDs; duplicate chunk IDs are dropped before pairing.
- Near-dedup: MinHash LSH with 128 hash functions, 32 bands, Jaccard ≥ 0.85 threshold.

## Gold subset

200 hand-curated pairs (25 per intent) are reserved for human review and the diff gallery. Curation tooling is tracked in #189. Cohen's κ target: ≥ 0.7.

## Known limitations

- All 10 source repos are permissively-licensed Python repositories. The corpus may not generalize to proprietary codebases with different style conventions.
- The heuristic labelers have non-zero false-positive rates; the gold-subset κ measurement quantifies this.
- The commit cutoff (2023-12-31) excludes the most recent refactoring patterns.
- Public pretrained baselines (CodeBERT) are trained on overlapping data; this is documented in the RESULTS-MEMO limitations section.
