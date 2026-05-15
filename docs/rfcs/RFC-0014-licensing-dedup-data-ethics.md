# RFC-0014 — Licensing, deduplication, data ethics

## Status
Locked (2026-05-15)

## Problem

Pin the legal and ethical contract for the corpus, the deduplication procedure, and the split construction. Without this RFC, redistributing checkpoints and datasets is unsafe.

## Decisions locked

### D1 — License inventory

Each of the 10 source repos is one of `{PSF, BSD-3-Clause, MIT, Apache-2.0}`. The mapping is recorded in `data/manifest.lock.json` with SPDX identifiers.

### D2 — Effective license of derived corpus

The derived `chunks` and `pairs` corpora are governed by the **most restrictive of the constituent licenses**, conservatively interpreted as **BSD-3-Clause** (because it requires attribution but is otherwise permissive). A `NOTICE` file in the corpus release enumerates per-repo licenses with original copyright headers preserved.

### D3 — Attribution

- The corpus release includes a `LICENSES/` directory with the full text of every applicable license.
- Every Parquet row carries `repo`, `commit_sha`, and `file_path` provenance; no chunk is detached from its origin.

### D4 — Redistribution

- We **redistribute** the derived `chunks` and `pairs` corpora on Hugging Face Datasets under the v1 manifest.
- We **redistribute** model checkpoints derived from this corpus under Apache-2.0 (the model weights themselves are our work; the training data attribution is preserved via `MODEL_CARD.md`).
- We do **not** redistribute raw source repository tarballs; the manifest records commit SHAs, and the mining scripts re-fetch from the upstream repos.

### D5 — Personal data and secrets

- We scan the chunk corpus for high-confidence secrets (regexes for AWS keys, GitHub PATs, etc.) with `trufflehog` or `detect-secrets`; any chunk with a hit is dropped from the corpus.
- Author identifiers from commits are **not** retained in any released artifact. Provenance carries `commit_sha`, not the committer.
- Email addresses and any PII discovered in comments or docstrings are dropped at the docstring-sentinel step (RFC-0012 §D5 collapses docstrings; for comments, we run an additional email/regex scrub before training).

### D6 — Deduplication procedure

1. **Exact dedup.** SHA-256 of normalized chunk source. Identical chunks across different files are kept once and the duplicates are recorded with provenance.
2. **Near dedup.** MinHash LSH over 5-gram BPE shingles, Jaccard threshold 0.85. Near-duplicate clusters are reduced to one representative per cluster.
3. **Cross-split anti-leakage.** Before splitting, the near-dup graph is computed across all repos. Any cluster that spans split boundaries pins all of its members to a **single split**, chosen by a stable hash of the cluster id; the alternative is to drop the cluster entirely. We *drop* clusters that would otherwise contaminate test.
4. **Boilerplate detection.** Chunks that match common copy-paste templates (`__init__.py` boilerplate, `setup.py` boilerplate, single-line lambda exports, `__all__ = […]`-only chunks) are filtered.
5. **Per-author dedup is not performed** (we don't track authors).

### D7 — Splits

- **By-repository splits**:
  - Train (6 repos): `pandas-dev/pandas`, `scikit-learn/scikit-learn`, `huggingface/transformers`, `pytest-dev/pytest`, `fastapi/fastapi`, `sqlalchemy/sqlalchemy`.
  - Val (2 repos): `django/django`, `python/mypy`.
  - Test (2 repos): `psf/black`, `python/cpython:Lib/`.
- The split mapping is recorded in `data/splits/v1.lock.json` and content-addressed.
- **No partial-repo leakage:** a file from repo `R` cannot appear in two splits.

### D8 — OOD probe

200 hand-curated pairs from `python/cpython:Lib/` refactors in the 3.11 → 3.13 development cycle. The OOD probe is held out from *all* corpus construction (it is constructed at eval time from a separate commit range than the test split's chunk corpus).

### D9 — Ethics statement

- The artifact is dual-use: a learned refactor tool can mask subtle behavior changes. RFC-0007 specifies the safety rails (refusal, no auto-apply, explicit confidence). The paper's limitations section makes this explicit.
- We do not train on private code, scraped commercial sources, or unlicensed data.
- We do not claim CodingJEPA is safe to apply to production code without human review. Every demo path makes this clear.

### D10 — Audit trail

The data audit (per RFC-0002 §D2) records, per repo:
- commit hash;
- license + SPDX;
- `.py` file count;
- chunk count;
- dropped-chunk rate;
- duplication rate;
- secret-scanner hits (must be 0 after the scrub).

The audit is `data/audit/<repo>.json` and committed to the repository.

### D11 — Right-to-removal

- We accept GitHub-issue requests to remove specific chunks tied to identifiable code. We re-version the corpus to `v1.1` and re-release if any such request is honored.
- We do not commit to a long SLA on these requests; the artifact is research-grade.

## Deferred items
- Multi-language license analysis (when we extend beyond Python).
- A formal data-card following the Hugging Face dataset card schema (we will produce one for the release; the RFC commits us to it but does not yet contain it).

## Acceptance condition

Locked when:
- the audit JSON exists for all 10 repos with all fields populated and 0 secret-scanner hits;
- the LSH dedup pipeline emits a deduplication report (`data/audit/dedup.json`);
- the cross-split leakage report (`data/audit/cross_split_leakage.json`) shows 0 cross-split clusters.
