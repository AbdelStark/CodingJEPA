# Security policy

CodingJEPA is a single-GPU research artifact, not a hardened production service.
The rules below are however non-negotiable. See `docs/spec/06-security.md` for
the full threat model.

## Reporting a vulnerability

We prefer **private GitHub Security Advisories**:

1. Go to https://github.com/AbdelStark/CodingJEPA/security/advisories/new
2. Fill in the advisory form with reproduction steps and the affected version.

If you cannot use Security Advisories, email the address listed in `README.md`
under the maintainer block. Encrypt with PGP if you can; we do not currently
publish a project-wide PGP key, so reach out to the maintainer for theirs.

Please do **not** open a public issue for a vulnerability.

## Coordinated disclosure window

We aim for **90 days from initial report to public disclosure**:

- **Day 0:** acknowledgement of the report within 7 days.
- **Day 0–60:** investigation and fix.
- **Day 60–90:** coordinated release (advisory + patched release).
- **Day 90:** public disclosure, even if a fix is not ready, with mitigation
  guidance.

If you require an extension beyond 90 days for legitimate reasons (e.g.,
upstream dep coordination), open the request in the advisory thread.

## Supported versions

| Version | Supported | Window |
|---|---|---|
| `v1.0.x` | yes | until `v1.0.0` release date **+ 6 months** |
| `< v1.0` | no | pre-release; no SLA |

Bug-fix releases (`v1.0.x`) carry security patches. Recipe or data changes
bump to `v1.1` and require an RFC amendment (`docs/spec/09-release-and-versioning.md`).

## Signing

- Git tags for releases (`v1.0.0`, …) are signed with the maintainer's GPG key.
  The key fingerprint is published in the `v1.0.0` release notes.
- Released checkpoints carry their `checkpoint_hash` in `MODEL_CARD.md` and on
  Hugging Face Hub; the eval harness (RFC-0010 §D1) refuses to run on a hash
  mismatch.
- The container image (`Dockerfile.eval`, issue #24) is published with a digest
  pin in `MODEL_CARD.md`.

## Out of scope

- **Bug-bounty program.** None in v1.
- **Demo deployment.** The web UI binds to `127.0.0.1:8080` and has no auth.
  Exposing it on a hostile network is the operator's responsibility (see
  `docs/spec/06-security.md` §Demo network exposure).
- **Self-hosted CI runners.** We use GitHub-hosted runners only.

## See also

- `docs/spec/06-security.md` — full threat model, secrets handling, sandbox.
- `docs/spec/04-error-model.md` — refusal taxonomy.
- `docs/spec/09-release-and-versioning.md` — release cadence.
