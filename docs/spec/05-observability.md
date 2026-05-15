# 05 — Observability

This document specifies how the system exposes its internal state — logs, metrics, traces — so that operators and contributors can diagnose failures and reproduce results without re-running every pipeline stage.

## Principles

- **Structured first.** Every log record is JSON; human-readable rendering is a downstream concern.
- **Reproducible cardinality.** Metrics that we *report* are bounded (one value per benchmark per system per seed); metrics that we *gather* during training are time-series owned by WandB.
- **Redaction by default.** No source identifiers (committer email, `_secret`-named locals, environment variables) leave the process.
- **Determinism is observable.** When `torch.use_deterministic_algorithms(True)` is on, the determinism state is in every log record and trace.
- **No drift between train/eval/inference logs.** Same field names, same JSON shape, same identifiers.

## Logging

### Format

JSON Lines, UTF-8, one record per line. Records are emitted to stdout; production deployments capture stdout to a file or log shipper. Records are also written to per-run directories during training and eval (`runs/<run_id>/log.jsonl`).

### Schema (every record)

| Field | Type | Notes |
|---|---|---|
| `ts` | string | ISO-8601 UTC with millisecond precision. |
| `level` | enum | `debug`, `info`, `warn`, `error`. |
| `event` | string | A short stable name, e.g., `train.step`, `eval.benchmark.start`, `inference.refusal`. |
| `request_id` | string | UUIDv7. Set per HTTP request and per CLI invocation. |
| `run_id` | string | UUIDv7. Set per training run / eval run. |
| `git_sha` | string | Code git sha. |
| `manifest_hash` | string | Set when relevant. |
| `checkpoint_hash` | string | Set when relevant. |
| `index_id` | string | Set when relevant. |
| `deterministic` | bool | Whether `torch.use_deterministic_algorithms(True)` is in force. |
| `seed` | int | Set for any record produced inside a seeded scope. |
| `error_code` | string | Set iff `level == "error"`. From the closed taxonomy in `docs/spec/04-error-model.md`. |
| `payload` | object | Free-form, but every key documented in the per-event table below. |

Records that fail JSON serialization are dropped to prevent log corruption; the drop count is itself logged at `event=log.dropped`.

### Closed event taxonomy (v1)

Adding an event is allowed; renaming or removing one bumps the schema version on the log writer.

| Event | Level | Payload | Context |
|---|---|---|---|
| `data.mirror.repo` | info | `{repo, commit_sha, files_in_scope}` | `python -m codingjepa data mirror` |
| `data.chunk.file` | debug | `{file_path, chunks_emitted, dropped_reasons}` | chunker |
| `data.dedup.cluster` | debug | `{representative, members_count, kept_in_split}` | dedup pass |
| `data.audit.repo` | info | `{repo, audit_path, gates_passed}` | audit script |
| `train.step` | info | `{step, epoch, pred_loss, sigreg_loss, loss, grad_norm, lr}` | every step |
| `train.eval_probe` | info | `{step, retrieval_at_10_val}` | every 5k steps |
| `train.rank_diagnostic` | info | `{step, effective_rank, embed_dim, gate_passed}` | per epoch |
| `train.checkpoint.write` | info | `{path, hash, val_metric}` | per epoch |
| `train.gate.failed` | error | `{gate, value, threshold}` | rank gate, monotonicity gate |
| `inference.embed` | debug | `{chunk_id, latency_ms}` | per call |
| `inference.retrieve` | info | `{request_id, intent, top_m, latency_ms}` | per request |
| `inference.rerank` | info | `{request_id, top_k, candidates: [...]}` | per request |
| `inference.refusal` | info | `{request_id, refusal_code}` | per refusal |
| `inference.success` | info | `{request_id, top_1_chunk_id, top_1_cosine, top_1_confidence}` | per success |
| `eval.benchmark.start` | info | `{benchmark, n_seeds}` | per benchmark |
| `eval.benchmark.done` | info | `{benchmark, metrics, wall_clock_s}` | per benchmark |
| `eval.harness.done` | info | `{benchmarks_run, total_wall_clock_s, results_path}` | end of `make eval` |
| `eval.sandbox.run` | debug | `{pair_id, passed, exit_code, wall_clock_s, reason}` | per execution-preservation pair |
| `safety.checker.fired` | info | `{request_id, checker, candidate_chunk_id, reason}` | per filtered candidate |
| `log.dropped` | warn | `{count, last_failure}` | log writer self-report |

### Log levels

| Level | Default in | Used for |
|---|---|---|
| `error` | always | Any exception from `codingjepa.errors`; gate failures. |
| `warn` | always | Non-fatal degradations: log dropped, slow-disk, cache miss. |
| `info` | training, inference, eval | Normal lifecycle events; one record per request, per benchmark, per epoch. |
| `debug` | off by default; on with `LOG_LEVEL=debug` | Per-step / per-chunk granularity. |

### Redaction

Before serialization, every record is passed through `codingjepa.observability.redact`. Rules:

- Strip any value matching the patterns in `codingjepa/safety/secret_patterns.py` (AWS keys, GitHub PATs, JWTs, hex strings ≥ 32 chars *not* in the whitelist of known hash fields).
- Strip any value matching email regex.
- Strip any string longer than 4 KB and replace with `"<redacted: too_long>"`.
- Never log raw source code chunks at `info`. At `debug` they may appear, truncated to 256 characters with `"…"`.

## Metrics

### Training (WandB, RFC-0008 §D12)

- Per step: `pred_loss`, `sigreg_loss`, `loss`, `grad_norm`, `lr`, `tokens_per_s`.
- Per 5k steps: `rank_diagnostic.effective_rank`, `retrieval_at_10_val`, `intent_hit_rate_val_probe`.
- Per epoch: full eval harness on val split.

WandB project: `codingjepa-v1`. Offline mode is acceptable; the resolved Hydra config is logged either way.

### Inference (Prometheus-style names; emitted as `info`-level log records)

| Metric | Type | Labels | Notes |
|---|---|---|---|
| `coding_jepa_requests_total` | counter | `intent, outcome={success, refusal_code, error_code}` | One per request. |
| `coding_jepa_request_latency_seconds` | histogram | `intent` | Buckets `{0.1, 0.25, 0.4, 0.7, 1.0, 1.5, 3.0}`. |
| `coding_jepa_index_hits_total` | counter | `pool` | Top-1 retrieval hit rate at runtime. |
| `coding_jepa_safety_checker_fires_total` | counter | `checker` | One per fired checker. |
| `coding_jepa_refusals_total` | counter | `refusal_code` | One per refusal. |

These are emitted as `inference.*` log records; an optional Prometheus exporter sidecar can scrape the structured logs. v1 ships no built-in Prometheus exporter.

### Eval (results/)

The eval harness emits a single `results.json` per run. Per-metric numbers carry mean / std / n_seeds / 95% CI / paired-bootstrap p-value (RFC-0010 §D6, §D7). See `docs/spec/03-data-model.md` for the schema.

## Tracing

v1 does not ship distributed tracing (no OpenTelemetry). Per-request correlation is via `request_id` in every log record. Eval and training use `run_id` for the same purpose.

The contract is forward-compatible: adding OpenTelemetry spans in v1.x must not change `request_id` semantics.

## Per-run directories

Every training and eval run creates a directory `runs/<run_id>/` containing:

```
runs/<run_id>/
├── config.yaml                 # resolved Hydra config
├── log.jsonl                   # complete run log
├── manifest.lock.json          # symlinked or copied
├── tokenizer.model             # symlinked or copied
├── seeds.json                  # per-component seeds
├── env.json                    # python version, torch version, cuda version, driver, hostname (anonymized)
├── checkpoints/
│   ├── weights_epoch_<N>.ckpt
│   ├── object_epoch_<N>.ckpt
│   └── best.ckpt
└── diagnostics/
    ├── rank_per_epoch.parquet
    └── loss_per_step.parquet
```

`runs/` is gitignored; releasing artifacts is a separate, deliberate step (see `docs/spec/09-release-and-versioning.md`).

## Demo log

`.runs/demo-log.jsonl` (gitignored). One record per demo request. Schema:

```json
{
  "ts": "ISO-8601 UTC",
  "request_id": "<uuid7>",
  "intent": "...",
  "source_hash": "<sha256 of normalized source>",
  "outcome": "success|refusal_code|error_code",
  "top_k_chunk_ids": ["..."],
  "top_k_scores": [0.84, ...],
  "user_marked": "accepted|rejected|null"
}
```

The demo log is **not** redistributed; it stays on the operator's machine and is only used for "shadow eval" reruns by the operator (RFC-0007 §D5).

## What we explicitly do not log

- Raw committer information (name, email).
- Raw source bodies at `info` level (only at `debug`, truncated).
- Hyperparameter sweeps' full config in every per-step record (it is logged once per run, referenced by `run_id`).
- Floating-point values to more than 6 significant digits (we emit `round(x, 6)` to keep records compact).

## Test coverage

- `tests/test_log_redaction.py` — feeds known-secret strings through every event payload and asserts the redactor strips them.
- `tests/test_log_schema.py` — every event in the closed taxonomy emits records that validate against the JSONSchema in `data/schemas/log.schema.json`.
- `tests/test_request_id_propagation.py` — asserts `request_id` flows from HTTP middleware through inference into log records.
