# 03 — Data model

This document specifies the schemas, on-disk formats, invariants, and versioning policy for every persistent artifact CodingJEPA produces. It is the authority for `data/schemas/`. Anything not specified here is private and may change.

## Schema versioning policy

- Every schema file in `data/schemas/` carries a `schema_version` field of the form `vMAJOR[.MINOR]`.
- A backwards-compatible addition (new optional column / field) bumps the minor version.
- A breaking change (renamed field, changed type, removed required field) bumps the major version and re-versions the artifact (e.g., `pairs/v1.parquet` → `pairs/v2.parquet`).
- Old major versions are retained read-only; downstream code must check `schema_version` and refuse to load incompatible majors.
- All schemas are validated by `tests/test_schemas.py` against committed fixture rows.

## Identifiers and hashes

| Identifier | Definition |
|---|---|
| `chunk_id` | `sha256(repo + commit_sha + file_path + node_qualname + source_normalized)[:16]` |
| `pair_id`  | `sha256(chunk_id_before + chunk_id_after + intent)[:16]` |
| `manifest_hash` | `sha256` of the canonicalized JSON of `data/manifest.lock.json` |
| `tokenizer_hash` | `sha256` of `tokenizer/v1/tokenizer.model` |
| `checkpoint_hash` | `sha256` of the safetensors weight blob |
| `index_id` | `f"{checkpoint_hash[:8]}-{manifest_hash[:8]}"` (RFC-0009 §D3) |

All hashes are hex-encoded lowercase. Truncations are stable and never re-keyed.

## Parquet schemas

### `data/parsed/<repo>/<file_path>.chunks.parquet`

One row per chunk. RFC-0012 §D11.

| Column | Type | Required | Notes |
|---|---|---|---|
| `chunk_id` | `string` | yes | See identifiers above. |
| `repo` | `string` | yes | e.g., `python/cpython`. |
| `commit_sha` | `string` | yes | Pinned commit. |
| `file_path` | `string` | yes | Path within the repo. |
| `node_qualname` | `string` | yes | Dotted name, e.g., `Class.method`. |
| `chunk_kind` | `enum<func, async_func, class, interstitial>` | yes | Dropped: `class` chunks exceeding 512 BPE tokens (RFC-0012 §D2). |
| `start_line` | `int32` | yes | 1-indexed, inclusive. |
| `end_line` | `int32` | yes | 1-indexed, inclusive. |
| `source_raw` | `string` | yes | Pre-normalization source. |
| `source_normalized` | `string` | yes | Post-normalization (RFC-0012 §D5). |
| `token_ids` | `list<int32>` | yes | BPE token ids; max length 512. |
| `compiled_ok` | `bool` | yes | `True` iff `compile(source_normalized, '<chunk>', 'exec')` succeeded under Python 3.12. |
| `dropped_reason` | `string` | no | Set iff this row's `chunk_id` was excluded from training; one of `{over_cap, parse_failed, secret_hit, near_dup, boilerplate}`. |
| `secrets_scanner_version` | `string` | yes | Version of `trufflehog`/`detect-secrets` used. |
| `chunker_version` | `string` | yes | Version of `codingjepa.data.chunker`. |
| `tokenizer_hash` | `string` | yes | Bytes hash of `tokenizer.model`. |

**Invariants:**

- `start_line ≤ end_line`.
- `compiled_ok = True` for any row not marked `dropped_reason`.
- `len(token_ids) ≤ 512`.
- `chunk_id` is unique within a single `chunks.parquet` file; cross-file duplicates are recorded in `data/audit/dedup.json`, not by re-emitting rows.

### `data/sequences/v1.parquet`

One row per training sliding window. RFC-0002 §D9.

| Column | Type | Required | Notes |
|---|---|---|---|
| `sequence_id` | `string` | yes | `sha256` over the ordered chunk_ids. |
| `repo` | `string` | yes | All chunks in the window must share `repo`. |
| `chunk_ids` | `list<string>` | yes | Length `S = H + n_preds + 1` (default 5). |
| `intent` | `string` | yes | Always `"NONE"` for v1 sequences. |
| `split` | `enum<train, val>` | yes | `test` repos do not produce pretraining sequences. |
| `manifest_hash` | `string` | yes | Manifest under which this row was generated. |

**Invariants:** all `chunk_ids` exist in `data/parsed/<repo>/...chunks.parquet` for the same `manifest_hash` and the same `repo`.

### `data/pairs/v1.parquet`

One row per labeled refactor pair. RFC-0002 §D6, §D10.

| Column | Type | Required | Notes |
|---|---|---|---|
| `pair_id` | `string` | yes | See identifiers. |
| `repo` | `string` | yes | |
| `commit_sha_before` | `string` | yes | |
| `commit_sha_after` | `string` | yes | Adjacent commit on the same file. |
| `file_path` | `string` | yes | |
| `node_qualname` | `string` | yes | Must be preserved across `before` and `after`. |
| `chunk_id_before` | `string` | yes | Resolves into `chunks.parquet`. |
| `chunk_id_after` | `string` | yes | |
| `intent` | `enum` | yes | One of the 8 intents or `"NONE"`. |
| `confidence` | `float32` | yes | Labeler confidence ∈ [0, 1]. |
| `labeler_version` | `string` | yes | Per-labeler semver. |
| `split` | `enum<train, val, test, ood>` | yes | Set after dedup + cross-split leakage resolution. |
| `gold_reviewed` | `bool` | yes | True for the 200 `data/gold/v1.jsonl` rows. |
| `gold_label` | `enum` | no | Set iff `gold_reviewed`. The reviewers' adjudicated intent. |
| `manifest_hash` | `string` | yes | |

**Invariants:**

- `(chunk_id_before, chunk_id_after, intent)` is unique.
- For `intent != "NONE"`: `acceptance_check(intent, before_cst, after_cst)` returns `True` (verified at load time in tests).
- Per-intent train cap: `count by (intent, split=train) ≤ 12,000` (RFC-0002 §D8).
- Surplus pairs go to `data/pairs/v1.overflow.parquet` with the same schema.

## JSON schemas

### `data/manifest.lock.json`

```json
{
  "schema_version": "v1",
  "manifest_hash": "<computed at write>",
  "generated_at": "ISO-8601 UTC",
  "chunker_version": "x.y.z",
  "tokenizer_hash": "<sha256>",
  "secrets_scanner_version": "x.y.z",
  "splits_path": "data/splits/v1.lock.json",
  "repos": [
    {
      "name": "python/cpython",
      "url": "https://github.com/python/cpython.git",
      "commit_sha": "<40-hex>",
      "license_spdx": "PSF-2.0",
      "subset_paths": ["Lib/"],
      "split": "test",
      "py_files_in_scope": 1234,
      "chunks_emitted": 5678,
      "chunks_dropped": 91,
      "audit_path": "data/audit/python-cpython.json"
    }
  ]
}
```

**Invariants:** the `manifest_hash` is computed by removing the `manifest_hash` field, canonicalizing the remaining JSON (sorted keys, no insignificant whitespace), and `sha256`-ing the bytes. `tests/test_manifest.py` validates this.

### `data/splits/v1.lock.json`

```json
{
  "schema_version": "v1",
  "split_hash": "<sha256 over canonicalized content>",
  "by_repo": {
    "pandas-dev/pandas": "train",
    "django/django": "val",
    "psf/black": "test"
  }
}
```

**Invariant:** `split_hash` is referenced from `data/manifest.lock.json` and from every `pairs.parquet` row.

### `data/audit/<repo>.json` (RFC-0014 §D10)

```json
{
  "schema_version": "v1",
  "repo": "...",
  "commit_sha": "...",
  "license_spdx": "...",
  "py_files_in_scope": 0,
  "chunk_count": 0,
  "median_chunk_token_len": 0,
  "drop_rate_over_cap": 0.0,
  "drop_rate_parse_fail": 0.0,
  "duplication_rate": 0.0,
  "secret_scanner_hits": 0,
  "compile_ok_rate": 0.0,
  "per_intent_pair_count": {"extract-helper": 0, "...": 0}
}
```

**Audit gates** (RFC-0002 §D2): `compile_ok_rate ≥ 0.95`, `duplication_rate < 0.30`, `secret_scanner_hits == 0`. Failure → drop the repo and promote a deferred candidate from `docs/data/CANDIDATE_REPOS.md`.

### `data/audit/dedup.json`

Records the MinHash-LSH near-dup clusters that were collapsed to a single representative.

```json
{
  "schema_version": "v1",
  "shingle_n": 5,
  "jaccard_threshold": 0.85,
  "clusters": [
    {"representative_chunk_id": "...", "members": ["...", "..."], "kept_in_split": "train"}
  ],
  "dropped_clusters": [
    {"reason": "would_contaminate_test", "members": ["..."]}
  ]
}
```

### `data/audit/cross_split_leakage.json`

Reports cross-split near-dup clusters and the resolution. **Invariant: zero entries with split crossings remain after the dedup pass.**

### `eval/pools/<benchmark>.lock.json`

```json
{
  "schema_version": "v1",
  "benchmark": "CJ-RET-100",
  "pool_size": 100,
  "chunk_ids": ["..."],
  "pool_hash": "<sha256 over sorted chunk_ids>"
}
```

### `indices/<index_id>.meta.parquet`

Sidecar to the FAISS index. RFC-0009 §D3.

| Column | Type | Notes |
|---|---|---|
| `chunk_id` | `string` | Resolves to `chunks.parquet`. |
| `position` | `int32` | Row index in the FAISS vector store. |
| `intent_acceptance_<intent>` | `bool` | One column per intent; precomputed for the rerank filter. |
| `provenance` | `struct<repo, commit_sha, file_path, node_qualname>` | |

**Invariant:** `len(parquet) == faiss.ntotal`.

### `data/gold/v1.jsonl`

One JSON object per gold pair. 200 rows, stratified 25/intent.

```json
{
  "schema_version": "v1",
  "pair_id": "...",
  "intent": "extract-helper",
  "before": "...",
  "after": "...",
  "annotators": ["a", "b"],
  "ratings_per_annotator": {"a": {"semantic": 5, "stylistic": 4, "intent": 5},
                             "b": {"semantic": 4, "stylistic": 5, "intent": 5}},
  "kappa_final": 0.78
}
```

**Invariant:** Cohen's κ ≥ 0.7 (RFC-0002 §D7). If lower, the gold subset is re-annotated, never silently accepted.

### `results/results.json` (RFC-0010 §D8, §D12)

The schema is committed at `data/schemas/results.schema.json`. Key invariants:

- Every benchmark in `RFC-0010 §D3` appears as a top-level key.
- Every reported number carries `mean`, `std`, `n_seeds`, `ci95_low`, `ci95_high`.
- Every "beats baseline" claim carries a `paired_bootstrap_p`.
- The file's `metadata` block includes `manifest_hash`, `checkpoint_hash`, `index_id`, `git_sha`.

### `MODEL_CARD.md` (Hugging Face Hub convention)

The model card is markdown with a YAML front-matter block:

```yaml
---
schema_version: v1
license: apache-2.0
checkpoint_hash: <sha256>
manifest_hash: <sha256>
tokenizer_hash: <sha256>
index_id: <id>
training_compute_h100_hours: <int>
seeds_reported: 3
---
```

## Cross-artifact invariants (CI-checked)

- Every `chunk_id` referenced in `pairs.parquet` exists in `chunks.parquet` for the same `manifest_hash`.
- Every `chunk_id` in any `eval/pools/*.lock.json` exists in `chunks.parquet` for the same `manifest_hash`.
- `data/manifest.lock.json#manifest_hash` matches the canonicalized hash of the file's other content.
- No `chunk_id` appears in two splits of `pairs.parquet`.
- The 8 intents have at least 100 test pairs each, or are flagged as "supplementary only" in `results.json` (RFC-0014 anti-leakage rule §D7 + RFC-0002 §D7 §D8 + RFC-0010 acceptance).

These invariants are tested in `tests/test_invariants.py`.

## Out of scope of this document

- The Python class `dataclass` definitions used at runtime — those live in `codingjepa.eval.schema` and `codingjepa.inference` and are documented in `docs/spec/02-public-api.md`.
- Hydra config schemas — see `config/` and RFC-0008 §D11.
- Demo log format (`.runs/demo-log.jsonl`) — gitignored; documented in RFC-0007 §D5.
