# RFC-0015 — Dataset release and Hugging Face publication

## Status
Draft (2026-05-15)

## Problem

Pin the dataset identity (name, schema, card), specify the full end-to-end recipe for generating the corpus from scratch, and define the Hugging Face upload procedure so any contributor can reproduce or update the release.

---

## Decisions locked

### D1 — Dataset name

**`human-python-refactors`**

Hugging Face slug: `CodingJEPA/human-python-refactors`

Rationale:
- `human` — primary differentiator: the corpus covers only commits authored before
  large-scale LLM code generation reached meaningful adoption (see RFC-0002 §D11,
  cutoff 2023-12-31). The word is a machine-searchable signal for downstream
  researchers who need provably human-authored training data.
- `python` — language scope; single-language datasets are more usable than polyglot
  ones for language-specific probing.
- `refactors` — content type; intent-labeled refactoring pairs distinguish this from
  general code corpora (The Stack, StarCoder).

Short alias used in papers and tooling: **HPR**.

### D2 — Schema

Two top-level HF splits: `pairs` and `chunks`.

**`pairs`** — refactor pair records (primary artifact):

| Field | Type | Description |
|---|---|---|
| `id` | `string` | SHA-256 of `(repo, commit_sha_before, commit_sha_after, node_qualname)` |
| `repo` | `string` | `owner/name` slug |
| `commit_before` | `string` | 40-char hex SHA of the "before" commit |
| `commit_after` | `string` | 40-char hex SHA of the "after" commit |
| `file_path` | `string` | Repo-relative file path |
| `node_qualname` | `string` | Dot-separated qualified name, e.g. `MyClass.my_method` |
| `chunk_before` | `string` | Normalized Python source of the node before the change |
| `chunk_after` | `string` | Normalized Python source of the node after the change |
| `intent_label` | `string` | One of 8 classes or `"NONE"` (RFC-0004) |
| `labeler_confidence` | `float32` | Heuristic labeler confidence ∈ [0, 1] |
| `split` | `string` | `"train"` / `"validation"` / `"test"` |
| `labeler_version` | `string` | semver of the labeler that produced this row |
| `corpus_version` | `string` | corpus release tag, e.g. `"v1.0.0"` |

**`chunks`** — pretraining chunk sequences (secondary artifact):

| Field | Type | Description |
|---|---|---|
| `id` | `string` | SHA-256 of normalized chunk source |
| `repo` | `string` | `owner/name` slug |
| `commit_sha` | `string` | HEAD commit of the pinned repo snapshot |
| `file_path` | `string` | Repo-relative file path |
| `node_qualname` | `string` | Qualified name or `"<interstitial>"` |
| `source` | `string` | Normalized Python source |
| `bpe_token_count` | `int32` | Token count under the 32k SentencePiece vocab |
| `split` | `string` | `"train"` / `"validation"` / `"test"` |
| `corpus_version` | `string` | corpus release tag |

### D3 — Splits

Matches RFC-0014 §D7 exactly (by-repository):

| Split | Repos |
|---|---|
| train | pandas-dev/pandas, scikit-learn/scikit-learn, huggingface/transformers, pytest-dev/pytest, fastapi/fastapi, sqlalchemy/sqlalchemy |
| validation | django/django, python/mypy |
| test | psf/black, python/cpython (Lib/ only) |

---

## End-to-end generation recipe

### Prerequisites

```bash
# Python ≥ 3.12, git, uv
pip install uv
uv sync --all-extras          # installs pydriller, libcst, datasketch, faiss-cpu, datasets, huggingface_hub, …

# Optional secret scanner
pip install detect-secrets    # or: brew install trufflehog
```

Environment variables:

```bash
export HF_TOKEN="hf_..."          # Hugging Face write token
export HF_REPO="CodingJEPA/human-python-refactors"
```

### Step 1 — Mirror repositories

Clone (or update) the 10 repos pinned at their locked SHAs.  The manifest
records the pinned SHA for each repo; clones are shallow-disabled so PyDriller
can walk the full history up to the 2023-12-31 cutoff.

```bash
uv run codingjepa data mirror \
  --output-dir data/repos \
  --manifest   data/manifest.lock.json
```

What this does:
- For each entry in `REPO_REGISTRY` (`codingjepa/data/mirror.py`), runs
  `git clone --no-single-branch` into `data/repos/<owner>/<name>`.
- Checks out the pinned SHA.
- Writes `data/manifest.lock.json` (content-addressed, includes
  `"commit_cutoff_utc": "2023-12-31T23:59:59Z"`).

Expected time: ~30 min on a 1 Gbit connection (repos total ~8 GB).

### Step 2 — Normalize source files

```bash
uv run codingjepa data normalize \
  --repos-dir data/repos \
  --output    data/normalized
```

Applies (per RFC-0012 §D5):
- `black` formatting (stable, `--line-length 88`).
- `isort` import sorting.
- Docstring replacement with `<DOC>` sentinel.
- Email/PII regex scrub (RFC-0014 §D5).
- `compile()` gate — files that fail are skipped and logged.

### Step 3 — Chunk

```bash
uv run codingjepa data chunk \
  --normalized-dir data/normalized \
  --output         data/chunks
```

Emits `data/chunks/<repo_slug>.parquet` with schema matching §D2 `chunks`
(minus `split`, added in Step 7).

### Step 4 — Extract refactor pairs (with 2023 cutoff)

```bash
uv run codingjepa data pairs \
  --repos-dir      data/repos \
  --normalized-dir data/normalized \
  --cutoff         2023-12-31 \
  --output         data/pairs_raw
```

The `--cutoff` flag is passed to the PyDriller loop (RFC-0002 §D11).  The
flag must be wired into `codingjepa/data/pairs.py` `COMMIT_CUTOFF_DATE`
constant (or accepted as a CLI argument) and compared against
`commit.author_date`.

Emits `data/pairs_raw/<repo_slug>.parquet`.

### Step 5 — Heuristic labeling

```bash
uv run codingjepa data label \
  --pairs-dir data/pairs_raw \
  --output    data/pairs_labeled
```

Runs the 8 heuristic labelers (RFC-0002 §D6) over each candidate pair.
Candidates that match no labeler are tagged `intent_label="NONE"` and routed
to the `chunks` pretraining pool only.

Applies the per-intent quota cap of 12,000 train pairs (RFC-0002 §D8).
Overflow rows are written to `data/pairs_labeled/overflow.parquet`.

### Step 6 — Secret scan

```bash
# Option A: detect-secrets
detect-secrets scan --all-files data/pairs_labeled/ > .secrets.baseline
python tools/assert_no_secrets.py .secrets.baseline

# Option B: trufflehog
trufflehog filesystem data/pairs_labeled/ --json | python tools/assert_trufflehog_clean.py
```

**Gate: zero hits required.** Any row with a hit is dropped and the ID is
appended to `data/audit/secrets_dropped.txt`.

### Step 7 — Deduplication

```bash
uv run codingjepa data dedup \
  --chunks-dir data/chunks \
  --pairs-dir  data/pairs_labeled \
  --output     data/deduped
```

Applies RFC-0014 §D6:
1. Exact dedup (SHA-256).
2. Near dedup (MinHash LSH, Jaccard ≥ 0.85, 128 hash functions, 32 bands).
3. Cross-split anti-leakage (drops clusters that span split boundaries).
4. Boilerplate filter.

Emits `data/deduped/dedup_report.json` (counts per step) and the cleaned
Parquet files.

### Step 8 — Assign splits

```bash
uv run codingjepa data splits \
  --deduped-dir data/deduped \
  --output      data/splits
```

Tags every row with `split ∈ {train, validation, test}` based on the
by-repository mapping (RFC-0014 §D7).  Writes the cross-split leakage
report to `data/audit/cross_split_leakage.json` (must be 0 cross-split
clusters).

### Step 9 — Audit

```bash
uv run codingjepa data audit \
  --repos-dir   data/repos \
  --splits-dir  data/splits \
  --output      data/audit
```

Produces `data/audit/<repo>.json` for all 10 repos and gates on:
- compile rate ≥ 0.95
- dedup rate < 0.30
- license ∈ {PSF, BSD-3-Clause, MIT, Apache-2.0}
- secrets == 0

**All gates must pass before upload.**

### Step 10 — Write manifest

```bash
uv run codingjepa data manifest \
  --splits-dir data/splits \
  --audit-dir  data/audit \
  --output     data/manifest.lock.json
```

Content-addresses the final corpus and writes the canonical
`data/manifest.lock.json` (includes `corpus_version`, `commit_cutoff_utc`,
per-repo SHAs, row counts, file hashes).

Commit this file:

```bash
git add data/manifest.lock.json data/audit/
git commit -m "data: v1.0.0 corpus manifest and audit"
```

### Step 11 — Convert to Hugging Face Datasets format

```python
# tools/hf_convert.py
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value

SPLITS_DIR = Path("data/splits")

def load_split(kind: str, split: str) -> Dataset:
    parquet_path = SPLITS_DIR / kind / f"{split}.parquet"
    return Dataset.from_parquet(str(parquet_path))

pairs = DatasetDict({
    "train":      load_split("pairs", "train"),
    "validation": load_split("pairs", "validation"),
    "test":       load_split("pairs", "test"),
})

chunks = DatasetDict({
    "train":      load_split("chunks", "train"),
    "validation": load_split("chunks", "validation"),
    "test":       load_split("chunks", "test"),
})
```

### Step 12 — Upload to Hugging Face Hub

```python
import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO  = os.environ.get("HF_REPO", "CodingJEPA/human-python-refactors")

# Create the repo if it doesn't exist
api = HfApi()
api.create_repo(repo_id=HF_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True)

# Push pairs config
pairs.push_to_hub(
    HF_REPO,
    config_name="pairs",
    token=HF_TOKEN,
    commit_message="data: v1.0.0 refactor pairs",
)

# Push chunks config
chunks.push_to_hub(
    HF_REPO,
    config_name="chunks",
    token=HF_TOKEN,
    commit_message="data: v1.0.0 pretraining chunks",
)

# Upload NOTICE and LICENSES/
api.upload_folder(
    folder_path="data/LICENSES",
    path_in_repo="LICENSES",
    repo_id=HF_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
api.upload_file(
    path_or_fileobj="data/NOTICE",
    path_in_repo="NOTICE",
    repo_id=HF_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
```

Run:

```bash
uv run python tools/hf_convert.py
```

### Step 13 — Publish dataset card

The dataset card must be committed to the HF repo as `README.md`.  Template:

```markdown
---
license: bsd-3-clause
language:
  - code
pretty_name: Human Python Refactors
tags:
  - code
  - python
  - refactoring
  - human-authored
  - pre-llm
configs:
  - config_name: pairs
    data_files:
      - split: train
        path: pairs/train-*.parquet
      - split: validation
        path: pairs/validation-*.parquet
      - split: test
        path: pairs/test-*.parquet
  - config_name: chunks
    data_files:
      - split: train
        path: chunks/train-*.parquet
      - split: validation
        path: chunks/validation-*.parquet
      - split: test
        path: chunks/test-*.parquet
---

# Human Python Refactors (HPR)

A corpus of intent-labeled Python refactoring pairs extracted from 10 curated
open-source repositories. All commits are dated **on or before 2023-12-31**,
ensuring the dataset contains exclusively human-authored code from the
pre-large-scale-LLM era.

## Usage

```python
from datasets import load_dataset

pairs  = load_dataset("CodingJEPA/human-python-refactors", "pairs")
chunks = load_dataset("CodingJEPA/human-python-refactors", "chunks")
```

## Source repositories

| Repo | License | Commits used |
|---|---|---|
| python/cpython (Lib/) | PSF | up to 2023-12-31 |
| django/django | BSD-3-Clause | up to 2023-12-31 |
| pandas-dev/pandas | BSD-3-Clause | up to 2023-12-31 |
| scikit-learn/scikit-learn | BSD-3-Clause | up to 2023-12-31 |
| huggingface/transformers | Apache-2.0 | up to 2023-12-31 |
| pytest-dev/pytest | MIT | up to 2023-12-31 |
| python/mypy | MIT | up to 2023-12-31 |
| psf/black | MIT | up to 2023-12-31 |
| fastapi/fastapi | MIT | up to 2023-12-31 |
| sqlalchemy/sqlalchemy | MIT | up to 2023-12-31 |

## Intent taxonomy

`extract-helper`, `inline-helper`, `comprehension-rewrite`, `dataclass-migration`,
`exception-handling-cleanup`, `loop-to-vectorized`, `argument-defaulting`,
`none-typing-modernization`, `NONE` (unlabeled, pretraining-only).

## Citation

```bibtex
@dataset{human-python-refactors,
  author    = {CodingJEPA contributors},
  title     = {Human Python Refactors},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/CodingJEPA/human-python-refactors},
}
```
```

Upload the card:

```bash
api.upload_file(
    path_or_fileobj="data/DATASET_CARD.md",
    path_in_repo="README.md",
    repo_id=HF_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
```

---

## Full pipeline one-liner (after prerequisites)

```bash
uv run codingjepa data all \
  --cutoff 2023-12-31 \
  --output data/ && \
uv run python tools/hf_convert.py
```

The `codingjepa data all` subcommand runs Steps 1–10 in sequence with early
exit on any audit gate failure.

---

## Release checklist

- [ ] `data/manifest.lock.json` committed and content-addressed
- [ ] `data/audit/<repo>.json` exists for all 10 repos (compile ≥ 0.95, secrets == 0)
- [ ] `data/audit/dedup_report.json` shows cross-split leakage == 0
- [ ] Secret scan: 0 hits
- [ ] `NOTICE` and `LICENSES/` directory present in repo and uploaded to HF
- [ ] `pairs` and `chunks` configs visible on HF Hub
- [ ] Dataset card published (includes citation block and source repo table)
- [ ] `--commit_cutoff_utc: 2023-12-31T23:59:59Z` visible in HF card metadata
- [ ] HF dataset repo set to public
- [ ] GitHub release tag `data-v1.0.0` created pointing to the manifest commit

---

## Acceptance condition

This RFC is locked when:
- the HF dataset `CodingJEPA/human-python-refactors` is publicly accessible;
- `load_dataset("CodingJEPA/human-python-refactors", "pairs")` returns the expected splits without error;
- the release checklist is fully checked.

## Deferred items

- JSONL export alongside Parquet (for tooling that doesn't support Parquet natively) — v2.
- Versioned re-releases when right-to-removal requests are honored (RFC-0014 §D11) — tracked as `v1.1.0`, etc.
- Multilingual extension beyond Python — v2 corpus.
