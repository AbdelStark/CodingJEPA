# RFC-0012 — Code chunking and tokenization

## Status
Locked (2026-05-15)

## Problem

Specify exactly how a Python file becomes a sequence of chunks and how a chunk becomes a sequence of BPE token IDs.

## Decisions locked

### D1 — Chunk definition
A **chunk** is the AST representation of one of:
- a `FunctionDef` or `AsyncFunctionDef`;
- a `ClassDef` body (one chunk per class; nested methods are *not* their own chunks, they live inside the class chunk in v1);
- a contiguous run of module-level statements between top-level definitions (an "interstitial" chunk).

The chunker is implemented with `libcst` to preserve formatting metadata that we will then deliberately normalize.

### D2 — Chunk size cap
- **Hard cap:** 512 BPE tokens (after normalization & tokenization).
- Chunks exceeding the cap are **dropped from training**; they do not contribute to either pretraining sequences or labeled pairs.
- For inference, oversized inputs trigger a refusal (RFC-0007 §D2).

We choose to drop rather than truncate because truncation would change semantics and corrupt the JEPA target signal. Larger chunks are a v2 problem.

### D3 — Chunk granularity (locked decision)
**Function/method-level granularity** is the unit:
- Each `FunctionDef` is one chunk; **nested defs become their own chunks** at training time but are also included verbatim within their enclosing chunk for the contextual encoder pass. (Subtle: training uses both perspectives; the eval and demo use only the function-level chunk.)
- Each `ClassDef` is one chunk *containing* its methods inline. Class-level chunks frequently exceed the 512-token cap and are then dropped per D2.

Rationale: functions are the most common refactor target in the corpus and they preserve semantic boundaries cleanly. Statement-level chunks are too small to give the predictor useful context; module-level chunks are too large and inflate the dropped-rate.

### D4 — Sequence construction
- **Within a file:** chunks are emitted in file order.
- **Sliding window:** `(H + n_preds + 1)`-sized windows for pretraining; default `H=3`, `n_preds=1`, so windows of 5.
- **Stride:** 1 (windows overlap maximally).
- **Cross-file sequences are not built in v1.** Each file's window stream is independent. (Cross-file via the import graph is a v2 extension noted in `RESEARCH.md`.)

### D5 — Normalization (applied to *chunk source text*, before tokenization)
- Run `black` with project default settings — formatting nuisance is removed before training.
- Run `isort` on imports inside the chunk if present.
- Replace docstrings with a fixed token sentinel `<DOC>` (length-preserving is not attempted; we collapse multi-line docstrings to the sentinel). Rationale: docstrings are high-cardinality, low-signal nuisance for retrieval; we treat them as a flag, not content.
- Strip all `# type: ignore` and editor pragmas.
- Strip all blank lines except those between top-level statements (which `black` enforces).
- Strip trailing whitespace; normalize line endings to `\n`.
- **Comments are kept** (they often carry semantic intent), but normalized to ASCII and stripped of trailing whitespace.

After normalization the chunk source is verified to `compile()` under Python 3.12; if it doesn't, the chunk is dropped.

### D6 — Identifier handling
- We do **not** alpha-rename or anonymize identifiers in training (e.g., we do not replace local names with `var_1, var_2, …`).
- Rationale: identifier semantics matter for code understanding; removing them removes signal. The robustness probes (RFC-0005 §D3) measure whether the model is *overfitting* to identifier surface form, which is the right way to check this.

### D7 — Tokenizer

- **Model:** SentencePiece BPE, vocab 32k.
- **Training corpus:** the 6 train repos' normalized chunks, ≈ 800 MB of text. The val and test repos are excluded from tokenizer training.
- **Special tokens:**
  - `[PAD]` (id 0), `[UNK]` (1), `[CLS]` (2), `[SEP]` (3), `[CHUNK]` (4), `<DOC>` (5).
  - Intent tokens: `[I_0]` … `[I_7]`, `[I_NONE]` (ids 6–14). These exist so we can also feed intent as a prefix token if the predictor-only conditioning underperforms in ablations.
- **Whitespace handling:** SentencePiece treats spaces as part of tokens; we use the standard `▁` prefix. Indentation is preserved.
- **Byte fallback:** enabled, so any UTF-8 input is at worst byte-segmented.
- **Coverage requirement:** ≥ 99.9% of training chunks tokenize without `[UNK]`. Audited before locking.

### D8 — Tokenization at inference
- Identical to training tokenizer.
- The committed tokenizer artifact is `tokenizer/v1/tokenizer.model` (binary SentencePiece file) plus `tokenizer/v1/special_tokens.json`.
- The tokenizer is version-pinned by content hash in `data/manifest.lock.json`.

### D9 — Encoder input format
For one chunk:
```
[CLS] <token_1> <token_2> … <token_n> [SEP] [PAD]…
```
- Max length 512.
- Per-position attention mask (1 for non-pad, 0 for pad).
- No segment IDs; the chunk is the segment.

For one sequence of `S` chunks (training-time): each chunk is encoded independently, producing `(S, D)`. We do **not** concatenate chunks into a single 512×S sequence in v1.

### D10 — Determinism
- Tokenization is bit-exact across runs (SentencePiece guarantees this).
- The chunker's order is fixed by `libcst` traversal; we sort within ambiguous orderings by `node.start_position`.

### D11 — Storage formats
- `data/parsed/<repo>/<file_path>.chunks.parquet` — one row per chunk, columns: `chunk_id`, `chunk_qualname`, `chunk_kind`, `start_line`, `end_line`, `source_normalized`, `token_ids`, `repo`, `commit_sha`.
- `data/sequences/v1.parquet` — sliding-window training sequences for Stage A.
- `data/pairs/v1.parquet` — labeled refactor pairs for Stage B.
- All Parquet schemas committed to `data/schemas/`.

## Deferred items
- Statement-level chunking (for v2 fine-grained probes).
- Multi-chunk sequences fed to one encoder call (concatenation with chunk separators).
- Identifier anonymization as an *augmentation*, not a normalization (v2 ablation).
- AST-aware tokenization (e.g., tokens per node-type with `parso`).

## Acceptance condition

Locked when:
- the chunker, normalizer, and tokenizer round-trip a held-out file to byte-exact tokens across two runs;
- the coverage audit reports ≥ 99.9% `[UNK]`-free coverage on the train corpus;
- the dropped-chunk rate (over-cap chunks) is reported and ≤ 15% on the train corpus.
