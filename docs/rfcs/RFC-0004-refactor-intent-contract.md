# RFC-0004 — Refactor intent contract

## Status
Locked (2026-05-15)

## Problem

Define the eight refactor intents, their input/output semantics, acceptance rules, and how unsupported cases are surfaced.

## Decisions locked

### D1 — Intent vocabulary (closed, 8 items + NONE)

| Index | Intent | One-line description |
|---|---|---|
| 0 | `extract-helper` | Extract a contiguous block of statements into a new helper function. |
| 1 | `inline-helper` | Inline a helper function at its (sole) call site. |
| 2 | `comprehension-rewrite` | Rewrite a `for` loop that builds a list/set/dict into a comprehension. |
| 3 | `dataclass-migration` | Convert a class with a hand-written `__init__` into a `@dataclass`. |
| 4 | `exception-handling-cleanup` | Tighten bare `except`, remove `try/except/pass`, prefer `contextlib.suppress` where appropriate. |
| 5 | `loop-to-vectorized` | Replace a Python loop over a numpy/pandas object with a vectorized op. |
| 6 | `argument-defaulting` | Add a default to a function parameter and propagate to call sites. |
| 7 | `none-typing-modernization` | `Optional[X]` → `X | None`, `Union` → `\|`, `typing.List` → `list`, etc. |
| 8 | `NONE` | No matching refactor. Used only for pretraining and as a refusal label. |

### D2 — Per-intent acceptance rules (used by the labeler in RFC-0002 *and* the inference filter in RFC-0009)

#### `extract-helper`
- `before` contains a contiguous block of ≥ 2 statements at a single nesting depth.
- `after` contains a new top-level `FunctionDef` `f` not present in `before`.
- A `Call(f, …)` replaces the extracted block in `after`.
- The set of names read/written by the extracted block matches the parameters and return values of `f`.

#### `inline-helper`
- `before` defines a function `f` with exactly one call site in scope.
- `after` removes `f` and inlines its body at the (sole) call site.
- Locals are alpha-renamed if needed; the inliner must preserve resolution.

#### `comprehension-rewrite`
- `before` contains `result = []; for x in iter: result.append(g(x))` (or the set/dict analogs).
- `after` replaces this with `result = [g(x) for x in iter]`.
- No `break`, `continue`, side effects in `g(x)`, or accumulator mutations beyond append/add/update.

#### `dataclass-migration`
- `before` is a `ClassDef` with an `__init__` whose body is `self.x = x` for each parameter (no defaulting, no validation, no derived state).
- `after` decorates the class with `@dataclass`, lists annotated fields, removes `__init__`.

#### `exception-handling-cleanup`
- One of:
  - bare `except:` → `except Exception:` or narrower;
  - `try: … except …: pass` → `try: … except …: log.exception(...)` or `raise`;
  - `try: … except …: pass` → `with contextlib.suppress(…): …`.

#### `loop-to-vectorized`
- `before` has a Python loop over a `pd.DataFrame`, `pd.Series`, or `numpy.ndarray` performing an elementwise operation.
- `after` calls a vectorized API (`.apply`, `.map`, arithmetic on the whole array, `np.where`, etc.) on the same object.
- The vectorized op is observationally equivalent for IEEE-754 floats and integer overflow semantics (best-effort heuristic, not proof).

#### `argument-defaulting`
- A function `f(a, b)` becomes `f(a, b=DEFAULT)`.
- Every call site that previously passed `b=...` is unchanged; call sites that did not pass `b` are unchanged.
- `DEFAULT` is a literal (not a mutable default; we reject `def f(x=[])` style).

#### `none-typing-modernization`
- Mechanical: `Optional[X]` → `X | None`; `Union[A, B]` → `A | B`; `typing.List[X]` → `list[X]`; etc.
- No semantic difference; must be a fixed-point under the rewriter.

### D3 — Prompt/conditioning representation
- Each intent has a dedicated learned embedding vector (index 0–7).
- `[I_NONE]` (index 8) is used during pretraining and as the "no intent matched" fallback.
- At inference, the user-selected intent index is passed to `action_encoder(intent_idx) → act_emb`, then summed onto the source chunk's embedding in the predictor input.

### D4 — Unsupported-case behavior
- If no intent's acceptance rule fires on a `(source, top-1 candidate)` pair, the demo surfaces "no acceptable candidate" rather than returning a low-confidence top-1.
- The acceptance rules are also used at **training-pair-mining** time (RFC-0002) and at **eval scoring** time (RFC-0010), guaranteeing consistency.

### D5 — Refusal rules (interlock with RFC-0007)
- The system refuses to apply an intent if:
  - the source chunk fails to `compile()` under Python 3.12;
  - the source chunk exceeds the 512-BPE-token cap (no truncated-input refactors);
  - the top-1 candidate's cosine similarity is below `τ_refuse = 0.55` after the rerank filter.

### D6 — Versioning
- The intent vocabulary is **closed** for v1. Adding or removing an intent bumps the project version to v2 and the data corpus version to `pairs/v2.parquet`.

## Deferred items
- Compound intents (`extract-helper` + `none-typing-modernization` on the same diff).
- Auto-suggesting an intent given only the source (intent-classification head).
- Intent-conditioned generation (replacing retrieval with a generative head).

## Acceptance condition

Locked when:
- the 8 labelers in `codingjepa/data/labelers/*.py` exist and pass their unit tests;
- the acceptance-rule filter `codingjepa/intents/acceptance.py` is implemented and reused by RFC-0009 and RFC-0010 (single source of truth).
