# 07 — Testing strategy

This document specifies the test pyramid, the property and ML-specific tests, the CI gates, and the rules under which a change is considered to have evidence sufficient to merge.

## Goals

- The smallest test that can fail tells the operator exactly what broke.
- Tests are deterministic; flakes are bugs and are quarantined within 24 hours.
- ML-specific failures (collapse, leakage, label drift, p-hacking) have dedicated tests, not just spot checks.
- The full `make eval` is not a test. Tests run in seconds-to-minutes; eval runs in hours.

## Pyramid

| Layer | Scope | Runtime | Marker | When |
|---|---|---|---|---|
| Unit | One function / class | < 100 ms each | `not slow` | Every PR. |
| Integration | One subsystem (e.g., chunker pipeline end-to-end on 1 file) | < 5 s each | `not slow` | Every PR. |
| Property | A documented invariant under random inputs | < 30 s each | `not slow` | Every PR. |
| Smoke | `make eval` on a 10-example fixture | < 90 s | `eval-smoke` | Every PR. |
| Slow | Real corpus slice; perf tests | minutes | `slow` | Nightly + on demand. |
| ML diagnostic | Collapse / leakage / drift detection | minutes | `slow` | Nightly. |

`pytest -m "not slow"` runs in CI on every PR. `pytest -m "slow"` runs nightly.

## Unit and integration tests (per subsystem)

| Module | Unit / integration coverage |
|---|---|
| `data.chunker` | Parses 50 fixture files, asserts `chunk_kind`, `start_line`, `end_line`, byte-exact `source_normalized`. |
| `data.normalize` | Black + isort idempotency. Docstring sentinel collapse. Pragma stripping. |
| `data.tokenizer` | Bit-exact tokenization across two runs. Coverage ≥ 99.9% `[UNK]`-free on a 1k-chunk fixture. |
| `data.pairs` | PyDriller fixture with 20 hand-curated commits emits the expected `(before, after)` pairs. |
| `data.dedup` | MinHash LSH on a fixture with known clusters; assert cluster ids and representative selection. |
| `data.splits` | A perturbation that would cause cross-split overlap is detected and dropped. |
| `data.audit` | Per-repo audit JSON validates against schema; gate logic correctly fails / passes on synthetic inputs. |
| `intents.acceptance` | Each of the 8 intents: 10 positive + 10 negative fixtures (RFC-0002 §D6 contract). |
| `modules.encoder` | Forward shape + parameter count + RoPE positional behavior on a sin/cos test. |
| `modules.projector` / `pred_proj` | Shape + BatchNorm1d behavior on `train()` vs. `eval()`. |
| `modules.ar_predictor` | Shape + cross-attention mask + n_preds=1 vs. n_preds=2. |
| `modules.intent_embedder` | One-hot index → embedding shape; `[I_NONE]` is index 8. |
| `modules.sigreg` | Numerical reference: known input distribution → known SIGReg loss within `1e-4`. |
| `model.CodingJEPA.forward` | Round-trip on a 4-example batch matches a numerically-evaluated reference (RFC-0008 acceptance). |
| `training.dataloader` | Deterministic seeding: two runs with same seed produce same batch order. |
| `training.callbacks` | Rank diagnostic gate fires below `0.9 × embed_dim`; loss monotonicity gate fires on synthetic divergence. |
| `inference.embed` | Round-trip embedding is L2-normalized to within `1e-6`. |
| `inference.index` | FAISS `IndexFlatIP` returns expected nearest neighbors on a 100-vector fixture. |
| `inference.retrieve` | End-to-end: chunk + intent → top-`M` is deterministic for a fixed seed. |
| `inference.rerank` | Acceptance + safety chain produces the documented refusal codes on hand-built failure cases. |
| `inference.confidence` | Softmax-at-`τ=0.1` matches a reference within `1e-6`. |
| `safety.checkers.*` | ≥ 5 positive + ≥ 5 negative cases per checker (RFC-0007 §D6). |
| `eval.benchmarks.*` | Each benchmark scores a 10-example fixture to a known number. |
| `eval.stats` | Bootstrap CI matches a reference (`scipy.stats.bootstrap`) within `1e-3`. |
| `demo.cli` | Click-style invocation; exit codes 0/1/2/3/4 each reachable from a test. |
| `demo.web` | `httpx.AsyncClient` exercises every endpoint; `200`/`400`/`413`/`503` reachable. |

## Property tests

| Property | Generator | Invariant |
|---|---|---|
| Tokenizer round-trip | Random ASCII Python source from a small grammar | `decode(encode(s))` is byte-equivalent up to whitespace normalization. |
| Chunker stability | Random reorderings of chunks within a file | `chunk_id` is invariant under file reordering of unrelated chunks. |
| Acceptance rule for `extract-helper` | Random insertion/removal of statements outside the extracted block | Acceptance verdict unchanged. |
| Safety chain | Random mutations to `chunk_after` that violate one checker | The corresponding checker fires; no candidate slips through (RFC-0007 §D6). |
| SIGReg invariance | Permutations of the batch axis | Loss invariant under permutation. |
| Bootstrap CI coverage | Synthetic samples from a known distribution | 95% CI covers the population mean ≥ 94% over 100 trials. |
| Manifest hash idempotency | Random reorderings of the JSON pre-hash | Hash invariant under canonicalization. |
| Pool determinism | Random seeds | `eval/pools/*.lock.json` content is invariant for a fixed seed and split. |

Property tests use `hypothesis` with deterministic seeds.

## ML-specific diagnostic tests

These run nightly (`slow` marker). They are not gates on PR merge but are gates on launching a real training run (RFC-0008 §D16) and on declaring v1.0.

| Diagnostic | Definition | Gate |
|---|---|---|
| Embedding rank | Effective rank `exp(H(σ))` over 10k val embeddings | ≥ `0.9 × embed_dim` (RFC-0008 §D7). |
| Pretrain loss monotonicity | Mean of `pred_loss` over 100-step windows in the first 5k steps | Strictly decreasing. |
| Cross-split leakage | MinHash LSH clusters spanning splits in a fresh manifest | Zero (RFC-0014). |
| Intent-balance | Per-intent count in fine-tune minibatches | Within ±10% of uniform after class weighting. |
| Tokenizer coverage | `[UNK]` rate on the train corpus | ≤ 0.1%. |
| Dropped-chunk rate | Over-cap chunks divided by total | ≤ 15% (RFC-0012 acceptance). |
| Index round-trip | Embed → index → retrieve same chunk → top-1 hit rate | ≥ 99.9%. |
| Determinism | `infer(s, cfg)` twice on a 100-chunk sample | All outputs bit-identical. |

Each diagnostic emits a JSON record under `runs/<run_id>/diagnostics/`, and the `train.gate.failed` log event surfaces a hard halt if a gate fails during training.

## Eval-as-test

The eval harness has its own correctness tests (since the harness is the artifact that decides whether the project ships):

- `eval/benchmarks/test_pool_determinism.py` — pools regenerated with same seed are bit-equal.
- `eval/test_results_schema.py` — `results.json` validates against `data/schemas/results.schema.json`.
- `eval/test_paired_bootstrap.py` — paired bootstrap p-values match `scipy.stats.bootstrap` (paired statistic).
- `eval/test_baseline_first_gate.py` — assert `make pretrain` refuses to run if `data/baselines/*/results.json` are missing (RFC-0005 §D9).

## Performance tests

Per RFC-0009 §D10. `tests/perf/test_latency.py` runs:

- 100 inference requests on a 512-token chunk against a 100k-vector index.
- Asserts P50 < 400 ms and P95 < 1.5 s.
- Records the actual percentiles to a per-run JSON for trending.
- CI fails if P95 regresses by > 20% vs. the last green main commit.

## Sandbox tests

Per `docs/spec/06-security.md`. Run on every PR:

- `tests/sandbox/test_no_network.py`
- `tests/sandbox/test_no_filesystem_escape.py`
- `tests/sandbox/test_timeout.py`
- `tests/sandbox/test_memory_cap.py`

## Coverage policy

- We do not enforce a coverage percentage. Lines without observable behavior do not need coverage; lines with documented contracts must have a test that fails when the contract is violated.
- Every public symbol from `docs/spec/02-public-api.md` has at least one test that exercises its documented contract.
- Every refusal code in `docs/spec/04-error-model.md` has a test that triggers it.
- Every error class in `codingjepa.errors` has a test that constructs and serializes it.

## CI gates (RFC-0013 §D6)

| Gate | Workflow | Blocks merge? |
|---|---|---|
| `lint` | `ruff` + `black --check` + `mypy --strict` on `codingjepa/` | yes |
| `unit` | `pytest -m "not slow"` on CPU | yes |
| `safety` | RFC-0007 §D6 property tests + sandbox tests | yes |
| `eval-smoke` | `make eval` on the 10-example fixture | yes |
| `perf` | `tests/perf/test_latency.py` | yes (regression > 20%) |
| `nightly-slow` | `pytest -m "slow"` + ML diagnostics on a 10k-chunk slice | no, but a red nightly opens an issue |
| `nightly-eval` | Full `make eval` on a recent checkpoint | no, posts results to a tracking issue |

## Flake policy

- A test that fails ≥ 2 times across distinct PRs without a code cause is marked `@pytest.mark.flaky` and an issue is opened with the test ID, the failure mode, and an owner with a 7-day SLA.
- Quarantine is not retirement: an unfixed flake in 30 days is escalated to a `priority:p1` issue and either fixed or removed.

## Test data

- Fixtures under `tests/fixtures/`. Real source code is committed under fair use only when essential and < 1 KB per file.
- Larger fixtures (10k-chunk slice for diagnostics) are produced on demand by `tests/fixtures/build_slice.py` from the pinned manifest, and cached in `~/.cache/codingjepa/` (gitignored).
- Random-seed test inputs are committed only as the seed; the inputs are regenerated.
