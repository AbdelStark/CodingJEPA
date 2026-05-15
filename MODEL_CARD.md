---
schema_version: v1
license: apache-2.0
checkpoint_hash: 0000000000000000000000000000000000000000000000000000000000000000
manifest_hash: 0000000000000000000000000000000000000000000000000000000000000000
tokenizer_hash: 0000000000000000000000000000000000000000000000000000000000000000
index_id: 00000000-00000000
training_compute_h100_hours: 0
seeds_reported: 1
---

# CodingJEPA v1.0 — model card

> **Template.** The front-matter hashes above are placeholders. Run
> `python tools/model_card_update.py <checkpoint> <manifest>` to populate
> them from real artifacts before publishing. `make eval` refuses to run
> while any hash is the all-zero placeholder
> (`ManifestHashMismatch` / `CheckpointHashMismatch` / `IndexHashMismatch`,
> exit code 4 per spec/04 §Exit codes).

## Model summary

CodingJEPA is a Joint-Embedding Predictive Architecture trained on Python
code chunks. It produces an embedding per ~512-token code chunk; a
single intent index conditions retrieval-based refactoring recommendations
(see `docs/rfcs/RFC-0004` for the 8-intent vocabulary).

## Intended use

- **In scope.** Narrow refactoring intents over Python source at chunk
  granularity; retrieval-based candidate selection followed by an
  acceptance + safety + threshold filter.
- **Out of scope.** General-purpose code generation, multi-language
  support, line-level edits beyond chunks, replacement for compilers /
  linters / IDE tools.

## Training data

10 curated permissive-license Python repositories, pinned to specific
commits in `data/manifest.lock.json`. See `docs/data/CANDIDATE_REPOS.md`
for the source list and `docs/rfcs/RFC-0014` for licensing and dedup
rules.

## Evaluation

- Benchmarks: CJ-RET-100, CJ-RET-1k, CJ-INTENT, CJ-EXEC, CJ-ROB-*,
  CJ-OOD, CJ-PROBE-*, CJ-HUMAN (RFC-0010 §D3).
- Acceptance bar (RFC-0001 §D6): Retrieval@10 ≥ 1.5× CodeBERT;
  intent-conditioned hit rate ≥ 2× unconditional latent baseline;
  execution-preservation pass rate ≥ 70% on the 500-pair subset;
  formatting-invariance rank change < 5%.

## Limitations

- Single-language (Python).
- Single-GPU training; not frontier-scale.
- Refactor recommendations are filtered through acceptance + safety;
  any refusal surface is in `docs/spec/04-error-model.md` §Refusal
  taxonomy.

## How to verify the release

```bash
make eval CHECKPOINT=... MANIFEST=...
```

The eval harness reads the front-matter above and refuses to run unless
the runtime artifacts match. Use `tools/model_card_update.py` to keep
the model card in sync with a checkpoint produced by `make pretrain` +
`make finetune`.

## Reproducibility

- Python 3.12, CUDA 12.4, torch 2.4+ (`pyproject.toml`).
- `uv sync --frozen` against `uv.lock` (RFC-0013 §D3).
- Seeds reported above; ≥ 3 seeds per RFC-0005 §D7 for any quoted
  metric.

## License

Released under Apache-2.0 (RFC-0014 §D4). The labeled pairs corpus is
released under the most restrictive constituent license (BSD-3-Clause,
RFC-0014 §D2).
