# CodingJEPA — candidate Python repositories (v1)

This document lists the **10 source repositories** used to mine training and evaluation data for CodingJEPA v1. The list is curated for diversity, quality, refactor-history density, and license compatibility.

## Selection criteria

1. **Idiomatic, reviewed Python.** Every PR goes through code review; refactors are deliberate.
2. **Long commit history with refactor density.** ≥ 10k commits or ≥ 5 years of active development.
3. **High test coverage.** Enables execution-preservation checks in the eval harness.
4. **Permissive license** (BSD/MIT/Apache/PSF). Reproducible redistribution of derived corpora.
5. **Spans a domain.** Together the 10 cover language internals, web, data, ML, type analysis, async, ORM, formatting, testing — broad enough to stress generality.
6. **Stable directory layout.** Reduces noise in chunk-extraction heuristics.
7. **Not contaminated by frontier code-model training to an unknown degree** beyond what we already accept by using public OSS.

## The 10 repositories (v1)

| # | Repo | Domain | License | Approx. size | Why it's in the list |
|---|---|---|---|---|---|
| 1 | `python/cpython` (`Lib/` only) | Language stdlib | PSF | 200k+ commits | Reference Python idioms; high-quality reviewed refactors; gold OOD probe source. |
| 2 | `django/django` | Web framework | BSD-3 | 35k+ commits | Mature OO Python, large test suite, deliberate refactor history. |
| 3 | `pandas-dev/pandas` | Data analysis | BSD-3 | 35k+ commits | Heavy performance refactors, vectorization, dataclass migrations. |
| 4 | `scikit-learn/scikit-learn` | Machine learning | BSD-3 | 30k+ commits | Clean estimator API; rich refactor history around API stability. |
| 5 | `huggingface/transformers` | Deep learning | Apache-2.0 | 17k+ commits | Modern Python; frequent refactors mirroring API evolution; large per-module surface. |
| 6 | `pytest-dev/pytest` | Testing framework | MIT | 12k+ commits | Idiomatic Python, plugin architecture, mature refactor norms. |
| 7 | `python/mypy` | Type checker | MIT | 20k+ commits | Semantics-heavy code; refactors are unusually careful about behavior preservation. |
| 8 | `psf/black` | Formatter | MIT | 4k+ commits | Lots of small focused refactors; stable test corpus; ideal for formatting-invariance probes. |
| 9 | `fastapi/fastapi` | Async web framework | MIT | 4k+ commits | Type-hint-heavy, modern Python; clear refactor commits. |
| 10 | `sqlalchemy/sqlalchemy` | ORM | MIT | 22k+ commits | Complex API design with documented refactor history; exercises descriptors, metaclasses, dynamic attrs. |

### Pinning

Each repository is pinned to a specific commit hash in `data/manifest.lock.json` (created in Phase 1). Pins are chosen at the start of Phase 1 and frozen for v1; any update bumps the corpus version to `v2`.

### Path filtering

For each repo we filter:
- only `*.py` (no `.pyi`, `.pyx`, `.pxd`).
- exclude vendored code (`*/vendor/*`, `*/third_party/*`, `*/_vendor/*`).
- exclude generated files (`*_pb2.py`, anything containing `# DO NOT EDIT`).
- exclude tests for **training corpus** but include tests for the **execution-preservation eval subset** (see RFC-0010).
- `python/cpython` is restricted to `Lib/` (stdlib only); we do not include the interpreter C bindings or the test suite for the standard library beyond what's needed for the OOD probe.

### Estimated training corpus size (post-dedup)

- ~1.2 GB of source code.
- ~600k chunks at 512 BPE tokens median.
- ~80k labeled refactor pairs across the 8 intents (see RFC-0004).

### Splits (RFC-0014)

- **Train (6 repos):** pandas, scikit-learn, transformers, pytest, fastapi, sqlalchemy.
- **Val (2 repos):** django, mypy.
- **Test (2 repos):** black, cpython (`Lib/` only).
- **OOD probe (200 hand-curated pairs):** drawn from cpython refactors during 3.11 → 3.13 cycle.

The split is **by-repository** to prevent intra-repo leakage; any pair whose `before` or `after` near-duplicates a chunk in another split is dropped.

## Excluded candidates (and why)

- `tensorflow/tensorflow`, `pytorch/pytorch` — mostly C++/CUDA; the Python surface is large but very heterogeneous in style.
- `ansible/ansible` — strong candidate; deferred to v2 corpus expansion (heavy DSL surface).
- `apache/airflow` — strong candidate; deferred to v2 (provider-package noise inflates chunk count).
- `tornadoweb/tornado` — small and clean but overlaps stylistically with `fastapi`; we kept `fastapi` for type-hint richness.
- `pallets/flask` — overlaps with `django`; we kept `django` for size and refactor history.
- `sympy/sympy` — strong candidate; deferred (symbolic-math idioms are out of distribution for general refactor intents).
- `scipy/scipy` — much of the value is in Fortran/C; the Python wrappers are stylistically uniform.

## License compliance

All 10 repos are under PSF / BSD-3 / MIT / Apache-2.0. The derived corpus is redistributable under the most restrictive of those (BSD-3-Clause attribution) plus a NOTICE file enumerating per-repo licenses and commit hashes. See RFC-0014 §3.

## Quality audit checklist (per repo, Phase 1)

For each repo we record, in `data/audit/<repo>.json`:
- commit hash pinned;
- number of `.py` files in scope;
- median chunk size in BPE tokens;
- candidate refactor-pair count per intent;
- duplication rate (MinHash LSH);
- proportion of chunks that successfully `compile()`-check;
- license string and SPDX identifier.

Any repo that fails the audit (compile rate < 95%, or duplication rate > 30%) is dropped and an alternate is promoted from the deferred list.
