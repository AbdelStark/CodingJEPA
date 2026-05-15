# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
per `docs/spec/09-release-and-versioning.md`.

## [Unreleased]

### Added
- Public package skeleton at `codingjepa/` matching `docs/spec/01-architecture.md`.
- `codingjepa.errors` closed exception taxonomy (`docs/spec/04-error-model.md`).
- `codingjepa.observability` structured-log writer, redactor, and UUIDv7
  `request_id` propagation (`docs/spec/05-observability.md`).
- `codingjepa.intents.acceptance` single-source-of-truth acceptance check
  for the 8 RFC-0004 §D2 intents.
- `codingjepa.safety.messages` closed refusal copy table R001–R007
  (RFC-0007 §D7).
- `codingjepa.safety.secret_patterns` full redactor/scanner pattern set
  from spec/05 §Redaction table.
- `data/schemas/` JSONSchemas for `manifest`, `splits`, `audit`, `dedup`,
  `cross_split_leakage`, `log`, `results`, `pool`, `gold`, `model_card`.
- Top-level `Makefile`, `pyproject.toml`, `uv.lock`.
- GitHub Actions workflows: `lint`, `unit`, `safety`, `changelog`.
- `codingjepa.eval.pools` deterministic, content-addressed eval pool
  construction; pools keyed by SHA-256 of sorted chunk IDs
  (`eval/pools/*.lock.json`).
- `codingjepa.eval.stats` bootstrap CI and paired-bootstrap p-value
  implementation (RFC-0005 §D6).
- `codingjepa.eval.schema.validate` — results.json schema validator with
  `jsonschema` enforcement (RFC-0010 §D5, spec/03).
- Hash-check enforcer in `codingjepa.eval.harness` — `make eval` refuses
  with exit code 4 on manifest / checkpoint / index hash drift
  (RFC-0010 §D1, spec/04).
- `MODEL_CARD.md` template with SHA-256 placeholders and
  `tools/model_card_update.py` updater script (RFC-0013 §D2).
- `docs/notes/RELEASE-RUNBOOK.md` v1.0.0 release checklist.
- Pre-commit hooks: `ruff`, `black`, `mypy`, `prettier` on `.yml`/`.yaml`/`.md`.
- `CONTRIBUTING.md`, GitHub PR template, and issue templates.
- `tests/test_invariants.py` — cross-artifact invariant suite verifying
  chunk_id ↔ pairs ↔ pools ↔ manifest consistency.

### Changed
- _none_

### Deprecated
- _none_

### Removed
- _none_

### Fixed
- _none_

### Security
- `SECURITY.md` with disclosure path and 90-day coordinated-disclosure window.

### Reproducibility
- `uv.lock` pins 129 packages on Python 3.12.
- `data/schemas/manifest.schema.json` requires sha256-shaped
  `manifest_hash` + `tokenizer_hash`; `tests/test_invariants.py` verifies
  the canonicalization rule.
- `eval/pools/*.lock.json` records the SHA-256 content address of each eval
  pool so `make eval` can detect drift before running benchmarks.

<!--
When adding an entry, place it under the matching heading above. Pick the
most accurate of:
  Added         — new public symbol, CLI flag, JSON schema, etc.
  Changed       — backwards-compatible change to existing public surface.
  Deprecated    — public surface scheduled for removal in a later release.
  Removed       — public surface removed (major-version-only).
  Fixed         — bug fixes that change observable behavior.
  Security      — fixes for vulnerabilities; references SECURITY.md.
  Reproducibility — data/schema/seed/lockfile/checkpoint changes; required
                    when public artifacts change per spec/09 §Changelog
                    discipline.

The `.github/workflows/changelog.yml` gate fails any PR touching
codingjepa/, data/schemas/, or docs/spec/02-public-api.md without an
[Unreleased] entry. Stub additions ("Add CHANGELOG entry") satisfy the
mechanical check but not the spec — write the actual entry.
-->

[Unreleased]: https://github.com/AbdelStark/CodingJEPA/compare/HEAD...HEAD
