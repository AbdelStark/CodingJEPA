# RFC-0002 — Dataset mining and intent labeling

## Status
Locked (2026-05-15)

## Problem

Specify how raw Python repositories become labeled refactor pairs and chunk sequences fit for training.

## Decisions locked

### D1 — Source repos
The 10 repositories listed in `docs/data/CANDIDATE_REPOS.md`, pinned at commit hashes recorded in `data/manifest.lock.json`. The list is closed for v1.

### D2 — Repository quality criteria (applied per repo audit)
- ≥ 95% of `.py` files in scope `compile()` successfully on Python 3.12.
- MinHash LSH duplication rate < 30% over file-level content.
- License is one of {PSF, BSD-3, MIT, Apache-2.0}.
- ≥ 5 years of commit history.

Any repo failing the audit is replaced from the deferred list in `docs/data/CANDIDATE_REPOS.md`.

### D3 — Commit walk and pair extraction
Use **PyDriller** to walk commit history. For each modified `.py` file in a commit:
1. Parse pre- and post-commit versions with `libcst`.
2. Identify edited top-level nodes (`FunctionDef`, `AsyncFunctionDef`, `ClassDef`).
3. For each node that exists in both versions and whose content differs, emit a candidate `(chunk_before, chunk_after, commit_sha, repo, file_path, node_qualname)`.

### D4 — Commit filtering (drops candidates)
A candidate is dropped if any of:
- merge commit, revert, or auto-generated (`Co-authored-by: dependabot[bot]`, `pre-commit-ci[bot]`, etc.).
- `chunk_before` or `chunk_after` exceeds the 512-BPE-token chunk cap from RFC-0012 by > 2× (truncation would change semantics).
- diff is purely whitespace or comments (`difflib` on a normalized version returns empty).
- diff touches only a `docstring` literal.
- the commit message contains any of the strings `wip`, `temp`, `revert`, `formatting only`, `lint only`, `style only`, `typo`.
- `chunk_before` and `chunk_after` differ in name or signature in a way that breaks public API resolution (we want refactors, not API changes); enforced by checking that the node qualname is preserved.

### D5 — Intent taxonomy (8 classes)
Locked in RFC-0004. The labelers in this RFC produce one of:
- `extract-helper`
- `inline-helper`
- `comprehension-rewrite`
- `dataclass-migration`
- `exception-handling-cleanup`
- `loop-to-vectorized`
- `argument-defaulting`
- `none-typing-modernization`

A candidate that does not match any heuristic is labeled `NONE` and goes only into the unconditional pretraining pool (not the intent-conditioned fine-tune pool).

### D6 — Heuristic labelers
Each intent has a labeler that takes `(before_cst, after_cst)` and returns `(matched: bool, confidence: float ∈ [0,1])`. The labelers are *conservative*: they prefer false negatives over false positives. Examples:

- `extract-helper`: a new top-level `FunctionDef` appears in `after` with name resolvable from the parent scope; ≥ 2 statements move from `before`'s node body to the new function; a call to the new function appears at the move site.
- `comprehension-rewrite`: a `for`-loop with a single `append`/`add`/`update` in `before` becomes a `ListComp`/`SetComp`/`DictComp` in `after`.
- `dataclass-migration`: a class with `__init__` setting `self.x = x` for `n` params in `before` becomes a `@dataclass` decorated class with `n` annotated attributes in `after`.
- `exception-handling-cleanup`: bare `except:` → `except Exception:` (or narrower), or `try/except/pass` → `try/except` with logging or re-raise.
- `loop-to-vectorized`: a Python for-loop over a numpy/pandas object becomes a vectorized operation on the same object.
- `argument-defaulting`: a function gains a default argument that previously had to be passed; call sites updated.
- `none-typing-modernization`: `Optional[X]` → `X | None`, `Union[A,B]` → `A | B`, or `from typing import List` → builtin generic.
- `inline-helper`: inverse of `extract-helper`.

Each labeler is unit-tested on 10 hand-picked positive + 10 negative cases.

### D7 — Gold subset
A 200-pair **gold subset** is sampled (stratified by intent, 25/intent) and manually reviewed by ≥ 2 reviewers. Inter-rater agreement (Cohen's κ) must be ≥ 0.7 before the heuristic labelers are trusted at scale.

The gold subset lives in `data/gold/v1.jsonl` and is committed to the repository.

### D8 — Per-intent quotas
Heuristic-labeled training pairs are capped per intent to prevent the corpus from being dominated by easy classes (e.g., `none-typing-modernization` is mechanical and high-recall). Cap: 12,000 train pairs per intent. Surplus pairs are kept in `data/pairs/v1.overflow.parquet` for v2.

### D9 — Pretraining sequence construction
For unconditional pretraining, sequences are built from each file by:
1. Parsing the file at HEAD of the pinned commit.
2. Emitting top-level chunks in file order.
3. Sliding a window of `S = H + n_preds + 1` chunks (default `H=3`, `n_preds=1`, so `S=5`).
4. Treating the last chunk as the prediction target and the preceding `H` as the context window.

Within a sequence, intent embedding is fixed to `[I_NONE]`.

### D10 — Provenance
Every pair carries its provenance: `(repo, commit_sha_before, commit_sha_after, file_path, node_qualname, intent_label, labeler_version)`. Provenance is the join key for downstream auditing.

## Deferred items

- Cross-file refactor detection (move-function-across-modules) — v2.
- Synthetic data augmentation via automated refactor tools (Rope, bowler) — v2.
- Multi-commit aggregation (squashing a refactor that spans 3 commits) — v2.

## Acceptance condition

Locked when:
- the 8 labelers each pass their unit tests;
- the gold subset reaches Cohen's κ ≥ 0.7;
- `data/manifest.lock.json` is committed and content-addressed.
