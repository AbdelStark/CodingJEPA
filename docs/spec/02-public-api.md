# 02 — Public API

This document specifies the surfaces a contributor or downstream user is allowed to depend on. Anything not listed here is private and may change without a deprecation cycle.

## Stability promise

| Surface | Stability for v1.x |
|---|---|
| `codingjepa` Python package public symbols (this doc) | Backwards-compatible additions only |
| CLI flags listed below | Backwards-compatible additions only |
| HTTP endpoints listed below | Backwards-compatible additions only |
| Parquet / JSON schemas in `data/schemas/` | Bumped via schema version; see `docs/spec/09-release-and-versioning.md` |
| Released checkpoints (HF Hub) | Frozen by content hash; v1.0.x are weight-compatible with v1.0 |
| Internal modules not listed below | None — may change at any time |

Removing or renaming a public symbol requires the deprecation procedure in `docs/spec/09-release-and-versioning.md`.

## Python package — public symbols

Importing path: `from codingjepa import …`. Anything reachable through that import is part of the contract; submodule paths starting with `_` or not re-exported through `codingjepa.__init__` are private.

### `codingjepa.model`

```python
class CodingJEPA(torch.nn.Module):
    """Top-level JEPA module mirroring LeWorldModel's `JEPA`.

    Composition is fixed: encoder + projector + ARPredictor + pred_proj +
    intent_embedder + sigreg. See RFC-0003 §D10.
    """

    encoder: codingjepa.modules.Encoder
    projector: codingjepa.modules.Projector
    predictor: codingjepa.modules.ARPredictor
    pred_proj: codingjepa.modules.PredProj
    action_encoder: codingjepa.modules.IntentEmbedder
    sigreg: codingjepa.modules.SIGReg

    def encode(self, chunk_tokens: torch.LongTensor) -> dict:
        """Encode (B, S, T) → dict with 'emb' (B, S, D) and 'act_emb' (B, S, D)."""

    def predict(self, ctx_emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Predict (B, n_preds, D) given context (B, H, D) + action (B, H, D)."""

    def embed(self, chunk_tokens: torch.LongTensor) -> torch.Tensor:
        """Project a single chunk to (B, D), L2-normalized. Used by the FAISS index."""

    def forward(self, batch: dict) -> dict:
        """Full LeWM-style `lejepa_forward`. Returns
        {'pred_loss': scalar, 'sigreg_loss': scalar, 'loss': scalar, 'emb': ..., 'pred_emb': ...}.
        """
```

`embed_dim`, `H`, `n_preds`, and parameter counts are documented in RFC-0003 and asserted in `tests/test_param_count.py`.

### `codingjepa.intents`

```python
INTENTS: tuple[str, ...] = (
    "extract-helper", "inline-helper", "comprehension-rewrite",
    "dataclass-migration", "exception-handling-cleanup",
    "loop-to-vectorized", "argument-defaulting", "none-typing-modernization",
)
INTENT_NONE: str = "NONE"

def intent_index(intent: str) -> int: ...
def intent_name(idx: int) -> str: ...

def acceptance_check(intent: str, before: libcst.Module, after: libcst.Module) -> bool:
    """Single source of truth for the per-intent acceptance rule (RFC-0004 §D2).

    Used by labelers, inference filter, eval scoring, and safety tests.
    """
```

### `codingjepa.inference`

```python
@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_path: pathlib.Path
    index_path: pathlib.Path
    intent: str
    k: int = 10
    threshold: float = 0.55
    deterministic: bool = True

@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    source: str
    cosine: float
    confidence: float
    accepted_by_intent: bool
    rejected_reason: str | None
    provenance: Provenance

@dataclass(frozen=True)
class Provenance:
    repo: str
    commit_sha: str
    file_path: str
    node_qualname: str

@dataclass(frozen=True)
class InferenceResult:
    candidates: list[Candidate]    # length ≤ k, post-filter, post-rerank
    refusal: str | None            # None when at least one candidate passed

def infer(source: str, cfg: InferenceConfig) -> InferenceResult: ...
```

### `codingjepa.eval`

```python
class Benchmark(Protocol):
    name: str

    def prepare(self, ctx: BenchmarkContext) -> None: ...
    def run(self, ctx: BenchmarkContext) -> dict: ...
    def score(self, raw: dict) -> dict: ...

def run_all(checkpoint: pathlib.Path, manifest: pathlib.Path, out: pathlib.Path) -> None: ...
```

The full `results.json` JSONSchema is committed at `data/schemas/results.schema.json` (see `docs/spec/03-data-model.md`).

### `codingjepa.modules`

`Encoder`, `Projector`, `ARPredictor`, `PredProj`, `IntentEmbedder`, `SIGReg` are public for ablation work. Their constructors are part of the contract; their internal `forward` shapes are documented in RFC-0003.

### Symbols that are NOT public

- Anything under `codingjepa.training._*`.
- Anything under `codingjepa.demo.web._*`.
- Anything not re-exported from a top-level `codingjepa.*.__init__`.
- Internal helpers in `codingjepa.data.*` other than the documented Parquet schemas.

## CLI surface

Every CLI is exposed via `python -m codingjepa <subcommand>`. The subcommands and their flags are part of the public contract.

### `python -m codingjepa refactor`

```
python -m codingjepa refactor \
    --file PATH \
    --node QUALNAME \
    --intent {extract-helper|inline-helper|...|none-typing-modernization} \
    [--k INT=10] \
    [--threshold FLOAT=0.55] \
    [--out PATH]
```

Behavior: per RFC-0006 §D2.

| Exit code | Meaning |
|---|---|
| 0 | One or more candidates passed acceptance and rerank. |
| 1 | Usage error (bad flags, missing file, unknown intent). |
| 2 | No acceptable candidate (refusal); see `--out` HTML for the reason if provided. |
| 3 | Internal error (load failure, index missing, deterministic mode violation). |

### `python -m codingjepa demo`

Launches the FastAPI + HTMX web app on `http://localhost:8080`. Flags:

```
python -m codingjepa demo \
    [--host HOST=127.0.0.1] \
    [--port PORT=8080] \
    [--checkpoint PATH] \
    [--index PATH]
```

### `python -m codingjepa data <step>`

Pipeline subcommands (one per Phase 1 step). Each is idempotent and deterministic.

```
python -m codingjepa data mirror      # mirror 10 source repos at pinned hashes
python -m codingjepa data chunk       # chunk + normalize + tokenize → parsed/
python -m codingjepa data pairs       # walk commits, label intents → pairs/
python -m codingjepa data dedup       # MinHash LSH dedup + cross-split anti-leakage
python -m codingjepa data splits      # write data/splits/v1.lock.json
python -m codingjepa data audit       # per-repo audit JSON, secret scan
python -m codingjepa data manifest    # write data/manifest.lock.json
```

### `python -m codingjepa train pretrain` / `train finetune`

```
python -m codingjepa train pretrain --config-name pretrain
python -m codingjepa train finetune --config-name finetune
```

Hydra-overrides are accepted directly: `python -m codingjepa train pretrain optimizer.lr=1e-4`.

### `python -m codingjepa eval`

```
python -m codingjepa eval [--checkpoint PATH] [--manifest PATH] [--out results/]
```

Refuses to run if the manifest, checkpoint, or index hashes do not match the values in `MODEL_CARD.md` (RFC-0010 §D1). Exit code 4 on hash mismatch.

### `python -m codingjepa index build`

```
python -m codingjepa index build --checkpoint PATH --pool {test|gold|user_pool.parquet}
```

## HTTP surface (demo)

FastAPI app at `codingjepa.demo.web.app`. v1 has no auth, no rate limiting, no multi-tenant support; intended for localhost only.

| Method | Path | Body / params | Response |
|---|---|---|---|
| `GET` | `/` | — | HTML form page |
| `POST` | `/refactor` | `{source: str, intent: str, k?: int}` | HTMX fragment with candidate list, diffs, scores, refusal banner if applicable |
| `GET` | `/healthz` | — | `200` JSON `{checkpoint_hash, index_hash, version}` |
| `GET` | `/version` | — | `200` JSON `{package_version, git_sha, model_card_hash}` |

Status codes:

- `200` for success and refusal (refusal is *displayed*, not an HTTP error).
- `400` for malformed body / unknown intent.
- `413` for source body exceeding 64 KB.
- `503` if the model fails to load at startup.

The web UI is a screenshot/video target, not a product. See RFC-0006 §D3.

## Makefile targets

```
make data           # full Phase 1 pipeline
make pretrain       # Stage A
make finetune       # Stage B
make eval           # full eval harness
make demo           # launch web UI
make clean-artifacts # clean .runs/, .artifacts/; never touches data/manifest.lock.json or checkpoints
make smoke          # CI smoke test (10-example fixture)
make lint           # ruff + black --check + mypy
make test           # pytest -m "not slow"
```

`make eval`'s output is `results/results.json` + `results/RESULTS-MEMO.md` + `results/diffs/` + `results/confusions/` + `results/figures/`.

## On-disk artifact paths

These paths are part of the contract; renaming requires a major version bump.

- `data/manifest.lock.json`
- `data/splits/v1.lock.json`
- `data/pairs/v1.parquet`
- `data/sequences/v1.parquet`
- `data/parsed/<repo>/<file_path>.chunks.parquet`
- `data/gold/v1.jsonl`
- `data/audit/<repo>.json`
- `data/audit/dedup.json`
- `data/audit/cross_split_leakage.json`
- `tokenizer/v1/tokenizer.model`
- `tokenizer/v1/special_tokens.json`
- `indices/<index_id>.faiss` and `indices/<index_id>.meta.parquet`
- `eval/pools/<benchmark>.lock.json`
- `results/results.json`
- `results/RESULTS-MEMO.md`
- `MODEL_CARD.md`

See `docs/spec/03-data-model.md` for the schemas.

## Deprecation policy

See `docs/spec/09-release-and-versioning.md`. Public symbols are removed only after one minor-version deprecation window with a `DeprecationWarning` and a documented migration path.
