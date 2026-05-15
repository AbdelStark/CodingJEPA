# Phase 4 — Demo Subsystem Notes

> Status: **complete** (2026-05-15, PR #172).

## What landed

Phase 4 implements the demo subsystem per RFC-0006. All 7 issues (#100–#106) are closed.

| Component | Module | Notes |
|-----------|--------|-------|
| CLI | `codingjepa.demo.cli` | `python -m codingjepa refactor --source … --intent … --k 10` |
| FastAPI web app | `codingjepa.demo.web` | `GET /`, `POST /refactor`, `GET /healthz`, `GET /version` |
| Diff renderer | `codingjepa.demo.diff` | Terminal (pygments) + self-contained HTML (monokai, no CDN at runtime) |
| HTMX templates | `codingjepa.demo.web.templates` | No build step; HTMX@1.9.12 loaded from unpkg at page load |
| Refusal messages | `codingjepa.demo.messages` | 5 keys covering demo-path refusal codes |
| Deterministic example | `examples/demo-cpython-extract-helper.py` | Runs without a trained checkpoint |
| Hidden-step ban | `codingjepa.demo.diff.render_diff_html` | Asserts `data-hidden-step` absent from diffs |

## Design decisions

- **No SPA framework**: HTMX was chosen over React/Vue to avoid a JS build step (RFC-0006 §D3).
- **Self-contained HTML**: `render_diff_html` produces a single file with inlined CSS — no external requests at render time.
- **Stub behavior**: With no trained checkpoint loaded, the CLI exits with a "no checkpoint loaded" message (exit 0). This allows the demo to be tested without a real model.

## What's not in this phase

- A live model: the demo wires up the inference pipeline but requires a trained checkpoint to serve real suggestions.
- Public deployment: deployment is out of scope for v0.4; the demo is a local dev tool.
