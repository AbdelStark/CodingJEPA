# Contributing

CodingJEPA is a small research artifact with a spec-first culture. The
**spec corpus** (`SPEC.md`, `docs/spec/`, `docs/rfcs/`) locks design
decisions before code expands. Changes that affect the public surface
follow the procedures below.

## Dev install

```bash
uv sync --frozen           # install runtime deps from uv.lock
uv sync --frozen --extra dev  # plus ruff / black / mypy
make lint                  # ruff + black --check + mypy --strict
make test                  # pytest -m "not slow"
```

The lockfile (`uv.lock`) is the **source of truth** per RFC-0013 §D3.
`uv sync --frozen` is required; never `uv lock` and `uv sync` in the
same change without intent — they regenerate the lock.

Set up the pre-commit hooks (`.pre-commit-config.yaml`):

```bash
uv pip install pre-commit
pre-commit install         # registers the git hook
pre-commit run --all-files # run once across the repo
```

Hooks: `ruff` (with `--fix`), `ruff-format`, `black`, `mypy --strict`
(scoped to `codingjepa/`), `prettier` on `.yml`/`.yaml`/`.md`, plus
the standard `check-added-large-files` (1 MB cap), `end-of-file-fixer`,
`trailing-whitespace`, `check-yaml`/`check-json`/`check-toml`, and
`check-merge-conflict`. CI mirrors these via `.github/workflows/lint.yml`;
running `pre-commit` locally is the fast feedback loop.

## Branching

- `main` is the default and is **PR-only**. No direct pushes.
- Feature branches: `<area>/<descriptive-name>` — `infra/...`,
  `data/...`, `ci/...`, `safety/...`, `intents/...`, etc. No
  tool/vendor branding in branch names (`docs/goal.md` §2).
- Backports during the v1.0.x support window land on `release/v1.0.x`.
- One issue = one branch = one PR (`docs/goal.md` §3). Bundling
  issues is rejected.

## RFC lifecycle

| State | Meaning |
|---|---|
| **Draft** | Author's initial proposal. May change in any direction. |
| **Proposed** | Open for review. Comments via PR review. |
| **Locked** | Decisions are final. Implementation issues file from this state. |
| **Superseded** | Replaced by a later RFC; the document is retained read-only. |

RFCs live under `docs/rfcs/`. Use the existing RFCs as templates. RFC
amendments are explicit: bump the locked date, add a "Changelog" block
at the bottom of the RFC, and reference the amendment in `CHANGELOG.md`
under `Reproducibility`.

## Issues and labels

Open issues use the templates in `.github/ISSUE_TEMPLATE/`. The label
convention from `docs/roadmap/IMPLEMENTATION.md` and `docs/goal.md` §2.2:

| Label group | Examples |
|---|---|
| `type:` | `type:feature`, `type:bug`, `type:test`, `type:ci`, `type:security`, `type:docs` |
| `area:` | `area:infra`, `area:data`, `area:model`, `area:training`, `area:inference`, `area:eval`, `area:safety`, `area:demo`, `area:release`, `area:baselines` |
| `priority:` | `priority:p0`, `priority:p1`, `priority:p2` |
| `effort:` | `effort:s`, `effort:m`, `effort:l` |
| `spec:` | `spec:rfc-0001` … `spec:rfc-0014`, `spec:spec-01` … `spec:spec-10` |

Milestones (`v0.1` … `v1.0`) track the implementation tracker at
`docs/roadmap/IMPLEMENTATION.md`.

## Pull-request requirements

Every PR uses `.github/PULL_REQUEST_TEMPLATE.md`. The mandatory
sections (see template):

- **Problem** — what is broken / missing.
- **Solution** — what you changed.
- **Validation** — commands run + their outputs.
- **Caveats** — known gaps; out-of-scope notes.
- **`Closes #<n>`** — link the issue.

CI gates that must pass:

- `lint` — ruff + black --check + mypy --strict.
- `unit` — `pytest -m "not slow"` parallel via xdist.
- `safety` — RFC-0007 property + RFC-0010 sandbox tests on every PR.
- `changelog` — fails when `codingjepa/`, `data/schemas/`, or
  `docs/spec/02-public-api.md` change without a new `[Unreleased]`
  bullet in `CHANGELOG.md` (see `#27`).

## Changelog discipline

Every PR that touches the public surface — `codingjepa/`,
`data/schemas/`, or `docs/spec/02-public-api.md` — adds an entry under
the `[Unreleased]` heading of `CHANGELOG.md`. The seven sections from
`docs/spec/09-release-and-versioning.md` §Changelog discipline are:

```
Added            new public symbol / CLI flag / schema / etc.
Changed          backwards-compatible change to existing public surface
Deprecated       public surface scheduled for removal
Removed          public surface removed (major-version-only)
Fixed            bug fixes that change observable behavior
Security         vulnerability fixes; reference SECURITY.md
Reproducibility  data / seed / lockfile / checkpoint changes
```

Stub entries that summarise the PR ("CHANGELOG entry") satisfy the
mechanical CI check but **not** the spec — write the actual entry.

## Deprecation procedure

`docs/spec/09-release-and-versioning.md` §Deprecation:

1. Emit a `DeprecationWarning` from the deprecated path. The message
   names the replacement and the removal version.
2. Add a `Deprecated` entry to `CHANGELOG.md`.
3. Document the migration in `docs/spec/02-public-api.md`.
4. Wait one minor-version release before removal.
5. On removal, add a `Removed` entry and bump the major version.

## Test pyramid

See `docs/spec/07-testing-strategy.md`. Local commands:

```bash
make test           # pytest -m "not slow"     (unit + property + smoke)
pytest -m slow      # slow integration                  (CI nightly only)
make smoke          # eval-smoke fixture                (10-example)
make lint           # ruff + black --check + mypy --strict
```

ML-specific gates (rank diagnostic, loss monotonicity, intent-conditioned
hit rate, retrieval@10) run during training; they are not CI gates.

## Reporting security issues

See `SECURITY.md` for the disclosure path and the 90-day window. Do
**not** open a public issue for a vulnerability.

## See also

- `SPEC.md` — top-level entry point and document map.
- `docs/goal.md` — autonomous-issue-resolution playbook (used by the
  bootstrap loop).
- `docs/roadmap/IMPLEMENTATION.md` — the master implementation tracker.
- `docs/spec/09-release-and-versioning.md` — semver, deprecation, support.
- `docs/spec/07-testing-strategy.md` — the test pyramid.
