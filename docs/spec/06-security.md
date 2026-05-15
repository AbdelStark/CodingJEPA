# 06 — Security

This document specifies the threat model, trust boundaries, secret-handling rules, sandbox guarantees, and supply-chain posture. CodingJEPA is a research artifact — it is not a hardened production service — but the rules below are non-negotiable.

## Trust model

| Boundary | Trust |
|---|---|
| Source repositories at pinned commits | Trusted as inputs (BSD-3 / MIT / Apache-2.0 / PSF). The pinning is the audit. |
| Released model checkpoints (HF Hub) | Signed by content hash in `MODEL_CARD.md`. |
| User-supplied source snippets at the demo surface | **Untrusted.** Treated as data. |
| User-supplied chunks for the eval execution-preservation sandbox | **Untrusted.** Run inside the sandbox per RFC-0013 §D7. |
| Operator (the person running `make`) | Trusted. |
| Network endpoints reached during training/eval | Trusted only for explicitly listed services (HF Hub, GitHub for source mirroring, WandB). |

## Threat model

We design against the following adversaries:

| Threat | Mitigation |
|---|---|
| **T1.** A user pastes malicious Python code into the demo to exfiltrate data or execute on the host. | We never `exec()` user input. The demo path is parse + tokenize + embed only. The eval execution-preservation sandbox is separate and isolated (RFC-0013 §D7). |
| **T2.** A poisoned source repo (one of the 10) injects backdoored chunks into training. | Source repos are pinned to specific commits in `data/manifest.lock.json`. Re-pinning is a deliberate corpus-version bump. The audit (RFC-0014 §D10) records compile-rate, license, and dedup hits per repo. |
| **T3.** A near-duplicate of a test chunk leaks into training. | Cross-split MinHash LSH dedup (RFC-0014 §D6); an explicit `data/audit/cross_split_leakage.json` invariant requires zero crossings. |
| **T4.** A secret (key, token) in a source repo is embedded into the corpus. | Pre-training secret scan with `trufflehog` / `detect-secrets` (RFC-0014 §D5); any chunk with a hit is dropped. The audit records `secret_scanner_hits == 0` as a gate. |
| **T5.** A demo refactor silently changes program semantics. | Safety filter chain (RFC-0007 §D1) refuses candidates that change side-effect surface, exception contract, public API, or async/sync boundary. Confidence threshold + acceptance rule must both pass; otherwise refusal (RFC-0007 §D2). |
| **T6.** The released checkpoint is swapped for a malicious one. | Checkpoint hash in `MODEL_CARD.md`; the eval harness refuses to run on a hash mismatch (RFC-0010 §D1). |
| **T7.** A dependency is compromised between pin and install. | `uv.lock` is the source of truth. Lockfile hashes are committed. `uv sync --frozen` in CI. |
| **T8.** The training process exfiltrates source via WandB. | WandB integration logs only the structured events listed in `docs/spec/05-observability.md`. Raw source is never uploaded; the redactor (`codingjepa.observability.redact`) is applied to every record. |
| **T9.** The web demo is exposed to a hostile network. | The web UI is intended for `localhost` only. There is no auth. Operators who expose it publicly are responsible for placing it behind a reverse proxy with auth. The README and `docs/spec/06-security.md` document this. |
| **T10.** A right-to-removal request is ignored. | RFC-0014 §D11 commits us to an issue-driven removal procedure with corpus re-versioning. |

## Secrets handling

Process-side rules:

- The training pipeline reads no secrets. WandB API keys live in `~/.netrc`; no other credentials are used.
- The mirror step uses `git clone https://github.com/...` with no auth (we depend only on public repos at public commits).
- Hugging Face Hub uploads use the operator's `HF_TOKEN` from the environment, scoped to the release repos. The token is never logged.

Corpus-side rules (RFC-0014 §D5):

1. Run `trufflehog` (or `detect-secrets`) over every chunk.
2. Drop any chunk with a high-confidence hit.
3. Run an additional regex pass for emails and known-PII patterns over `comments` (docstrings are already replaced with the `<DOC>` sentinel by the normalizer; see RFC-0012 §D5).
4. The audit asserts `secret_scanner_hits == 0` per repo before training is allowed to launch. CI gate enforces this.

Demo-side rules:

- Source pasted into the demo is hashed (`sha256`) before logging; the raw source is not persisted (RFC-0007 §D5 explicitly logs only the hash plus user-marked outcome).
- The redactor (`docs/spec/05-observability.md` §Redaction) strips secret-shaped values from every log record before write.

## Sandbox (execution-preservation eval)

Per RFC-0010 §D2 / RFC-0013 §D7. Each test invocation runs in a child process with:

- 30-second wall-clock timeout (`SIGKILL` on overrun).
- No network: firewall-denylist via `nsjail` or `firejail`. The list is committed at `eval/sandbox/policy.json`. CI verifies a no-network smoke test attempts a connection and is blocked.
- Read-only filesystem except `/tmp/<run>/`.
- Memory cap 4 GB (`prlimit --as=4G`).
- CPU-only by default; the sandbox does not have a GPU device handle.
- No environment inheritance other than `PATH`, `PYTHONPATH`, `TZ`.
- No SUID binaries, no `ptrace`, no Docker socket, no `sudo`.

Outputs are captured: `{passed: bool, exit_code: int, wall_clock_s: float, stdout: str (≤ 16 KB), stderr: str (≤ 16 KB), reason: enum<ok, timeout, memory, network, segfault, nonzero_exit>}`. The aggregated pass-rate counts timeouts and any non-zero exits as failures.

Sandbox failures are *not* errors in the sense of `docs/spec/04-error-model.md`; they are observations. Errors arise only if the sandbox itself fails to launch (e.g., `nsjail` missing).

## Demo network exposure

The web UI binds to `127.0.0.1:8080` by default. Binding to `0.0.0.0` requires the explicit `--host 0.0.0.0` flag and prints a warning at startup. There is no auth. Operators who require remote access must place a reverse proxy in front; this is out of scope for v1.

## Supply chain

| Surface | Posture |
|---|---|
| Python deps | `pyproject.toml` + `uv.lock`; `uv sync --frozen` in CI; lockfile is the source of truth. |
| Container image | `Dockerfile.eval` pinned to a specific Python and CUDA base; built image is published with a digest pin. |
| Pretrained baseline (CodeBERT) | `microsoft/codebert-base` from HF Hub at a pinned commit hash recorded in `data/baselines/codebert.lock.json`. |
| Source corpus | 10 repos at pinned commits in `data/manifest.lock.json`; manifest is content-addressed. |
| Model release | Signed by HF Hub commit hash + content hash in `MODEL_CARD.md`. |
| Tokenizer | Pinned by `tokenizer_hash` in `data/manifest.lock.json`. |
| CI runners | GitHub-hosted only; no self-hosted runners on private branches. |

We do not use sigstore / Sigsum / in-toto in v1. Adding any of those is a v1.x amendment.

## Redistribution and licensing

- The derived `chunks` and `pairs` corpora are redistributed under the most restrictive constituent license (BSD-3-Clause), with attribution preserved per chunk (RFC-0014 §D2).
- The model weights are released under Apache-2.0 (RFC-0014 §D4).
- `LICENSES/` directory in any released artifact carries the full text of every applicable license.
- We do not redistribute raw source repository tarballs; the manifest records commit SHAs and the mirror script re-fetches from upstream.

## Vulnerability reporting

Until a `SECURITY.md` is committed (see issue tracker), the reporting path is:

- Open a private GitHub Security Advisory on the repository.
- Or email the maintainer at the address listed in the README.

A v1.x amendment will replace this section with a formal `SECURITY.md` and a 90-day coordinated-disclosure window.

## Tests / CI gates

- `tests/test_secret_scan.py` — runs the secret-scanner on the audit fixtures and asserts zero hits.
- `tests/test_redactor.py` — fuzz-feeds known secret patterns through the log redactor.
- `tests/sandbox/test_no_network.py` — runs a child that attempts a TCP connect inside the sandbox; asserts the connection is blocked.
- `tests/sandbox/test_no_filesystem_escape.py` — child attempts `open("/etc/shadow", "rb")`; asserts deny.
- `tests/sandbox/test_timeout.py` — child sleeps 60s; asserts kill at 30s.
- `tests/test_safety_chain.py` — property test, RFC-0007 §D6.
- CI `safety` workflow runs all of the above on every PR.
