# RFC-0006 — Demo and developer workflow

## Status
Locked (2026-05-15)

## Problem

Define the demo surface: what it shows, how it is invoked, how diffs and rationale are rendered, and what failure looks like to the user.

## Decisions locked

### D1 — Primary operator journey
1. User runs `python -m codingjepa demo` (CLI) or opens `http://localhost:8080` (web).
2. User pastes a Python snippet (a function or class up to 512 BPE tokens).
3. User selects one of the 8 intents.
4. The system encodes, predicts the target latent, retrieves top-`k=10` candidates, filters with the intent acceptance rule (RFC-0004 §D2), and reranks.
5. The UI shows:
   - the top-`k` candidate snippets, ordered by reranked score;
   - per-candidate side-by-side diff (unified or split);
   - per-candidate cosine similarity, confidence, and provenance (source repo + commit hash);
   - the rejected candidates (those that failed acceptance), greyed out with a one-line reason.
6. User can mark a candidate accepted/rejected; the action is logged to `.runs/demo-log.jsonl` for analysis.

### D2 — CLI contract
`python -m codingjepa refactor --file path/to/source.py --node qualname --intent extract-helper [--k 10] [--threshold 0.55] [--out diff.html]`

- Default output: pretty terminal diff with cosine scores.
- `--out diff.html` writes a self-contained HTML diff card suitable for screenshots.
- Exit code 0 on success, 2 on "no acceptable candidate", 1 on usage error.

### D3 — Web UI contract
- Built with FastAPI + HTMX (no SPA, no build step).
- One route: `POST /refactor` returns an HTMX-friendly fragment with the candidates.
- One static page: `GET /` returns the form.
- The web UI is intentionally minimal; it is screenshot/video target, not a product.

### D4 — Diff and rationale rendering
- Unified diff via `difflib.unified_diff`, syntax-highlighted with `pygments`.
- Cosine score and confidence are shown numerically (e.g., `cos 0.84 · conf 0.71`).
- Provenance: small text caption with `repo @ shorthash` and `node.qualname`.
- Acceptance-rule outcome: a green tick / red cross with a one-line reason if rejected (e.g., "comprehension-rewrite: source has a `break`, not eligible").

### D5 — Screenshot/video export requirements
- `--out diff.html` produces a single self-contained HTML file (inlined CSS, no external requests). Renders in any browser, scriptable by Playwright for video capture.
- The demo includes a deterministic example in `examples/demo-cpython-extract-helper.py` that, when run, produces a reproducible diff.

### D6 — Failure surfacing
- If no candidate meets the acceptance rule for the chosen intent, the UI shows:
  > **No acceptable candidate for intent `extract-helper`.** The top-1 candidate failed acceptance because: *the proposed helper's read/write set does not match its parameter list*.
- If the source chunk exceeds the 512-BPE-token cap:
  > **Input too long.** CodingJEPA v1 only supports chunks ≤ 512 BPE tokens.
- If the source chunk fails to `compile()`:
  > **Source does not parse.** Returning verbatim. (Refusal per RFC-0007.)

These messages are deterministic and committed to a copy table in `codingjepa/demo/messages.py`.

### D7 — Developer workflow
- `make data` — runs the Phase 1 pipeline.
- `make pretrain` — launches the LeWM-derived pretraining run.
- `make finetune` — launches intent-conditioned fine-tune.
- `make eval` — runs the full eval harness, emits `results/RESULTS-MEMO.md` and `results/results.json`.
- `make demo` — launches the web UI on `localhost:8080`.
- `make clean-artifacts` — deletes `.artifacts/` and `.runs/`. Does **not** touch `data/manifest.lock.json` or checkpoints.

### D8 — Hidden-step ban
The demo MUST run against the same artifacts the eval harness uses. No private preprocessing in the demo path. If the demo needs a precomputed FAISS index, that index is the same one referenced by `results/results.json`.

## Deferred items
- IDE integration (VS Code extension).
- Multi-file refactor sessions.
- Real-time inference latency tuning beyond the v1 target (P95 < 1.5s).

## Acceptance condition

Locked when:
- `make demo` works from a fresh clone given a released checkpoint and data snapshot;
- the deterministic example in `examples/` reproduces the same top-1 diff across runs;
- the failure messages exist as a tested copy table.
