# 09 — Release and versioning

This document specifies the versioning scheme, the deprecation policy, the changelog discipline, the release procedure, and the support windows.

## Semantic versioning

The project follows SemVer for the public surface defined in `docs/spec/02-public-api.md`:

- **MAJOR** version increments on backwards-incompatible changes to public symbols, CLI flags, HTTP endpoints, or persisted artifact schemas.
- **MINOR** version increments on backwards-compatible additions.
- **PATCH** version increments on backwards-compatible fixes (no contract changes).

The first stable release is `v1.0.0`. Pre-release qualifiers (`-rc.1`, `-rc.2`) are used during release candidates.

## What is versioned

| Surface | Versioning |
|---|---|
| Python package | `pyproject.toml` `version` field; SemVer above. |
| CLI flags | Same as the package; flag removals require a deprecation cycle. |
| HTTP endpoints | Same as the package. |
| Persisted artifact schemas (`data/schemas/`) | Per-schema `schema_version` (`vMAJOR[.MINOR]`). Major bumps re-version the artifact directory (`pairs/v1.parquet` → `pairs/v2.parquet`). Old majors stay read-only. |
| Data manifest | `manifest.lock.json` carries `schema_version`; the *manifest content* is content-addressed by `manifest_hash`. |
| Tokenizer | Pinned by content hash in the manifest. New tokenizer = new manifest = new `index_id` (RFC-0009 §D3). |
| Checkpoints | Pinned by content hash. New checkpoint = new `index_id`. |
| FAISS index | `index_id = f"{checkpoint_hash[:8]}-{manifest_hash[:8]}"`. New either side = new index. |
| Eval pools | Each pool is content-addressed (`pool_hash`); change = new benchmark version. |
| RFCs | Each RFC's status moves Draft → Proposed → Locked → Superseded by RFC-NNNN. The RFC number is permanent; supersession is a new RFC, not a rewrite. |

## Deprecation policy

A public symbol may be removed only after the following procedure:

1. **Announce** in the changelog of release `MAJOR.MINOR`: mark the symbol with `@deprecated(removed_in="MAJOR.NEXT_MINOR+1")` and emit `DeprecationWarning` at runtime.
2. **Document** in `docs/spec/02-public-api.md`: a "Deprecated" section listing the symbol, the date of announcement, the migration path, and the targeted removal version.
3. **Wait** at least one minor release.
4. **Remove** in the targeted minor release; add an entry to the breaking-change section of the changelog with the migration path.

Within a major version, no public symbol is removed without going through this procedure.

For artifact schemas, breaking changes always bump the artifact major version (e.g., `v1` → `v2`); the prior major remains readable for one minor cycle to allow a migration script to run.

## Changelog discipline

`CHANGELOG.md` follows Keep-a-Changelog with strict sections per release:

- `### Added` — new public surface.
- `### Changed` — backwards-compatible behavior changes (flagged with rationale).
- `### Deprecated` — newly deprecated symbols and the targeted removal version.
- `### Removed` — removed symbols (always associated with a major bump).
- `### Fixed` — bug fixes; cite the issue.
- `### Security` — vulnerability fixes; cite the advisory.
- `### Reproducibility` — changes to manifest, tokenizer, checkpoint, index, or eval pools that change reproduced numbers.

Every PR that touches the public surface, an artifact schema, or a metric must update `CHANGELOG.md` under the `[Unreleased]` heading. CI fails on PRs that touch those surfaces without a changelog entry.

The `Reproducibility` section is load-bearing: users reproducing numbers from a previous release rely on it to know whether the upgrade changes their numbers.

## Release procedure (v1.0.0 and v1.x.y)

Per RFC-0011 §D7, RFC-0013 §D8.

1. **Pre-flight checks.** All CI gates (`lint`, `unit`, `safety`, `eval-smoke`, `perf`) green on `main`. Nightly diagnostics (rank, leakage, intent balance) green for at least 3 consecutive nights.
2. **Eval reproduction.** `make eval` from a fresh clone produces a `results.json` whose numbers match the committed `results/results.json` within bootstrap CI overlap.
3. **Versions and changelogs.** Bump `pyproject.toml`. Move `[Unreleased]` content under the new version heading in `CHANGELOG.md`. Add the release date.
4. **Tag.** `git tag v1.0.0 -s -m "v1.0.0"`. Signed tags; never tag from a fork.
5. **Build.** `uv build` produces wheels; `docker build -f Dockerfile.eval` produces the eval image; both are reproducible from the tag.
6. **Publish.** Upload wheels to PyPI; push the eval image to GHCR with a digest pin (recorded in `MODEL_CARD.md`); upload model weights, tokenizer, and `MODEL_CARD.md` to HF Hub at `<org>/coding-jepa-vMAJOR`.
7. **Release notes.** GitHub Release page links to the changelog section, the HF Hub release, the GHCR digest, and the relevant tracking issues.
8. **Audit trail.** `MODEL_CARD.md` carries `checkpoint_hash`, `manifest_hash`, `tokenizer_hash`, `index_id`, `git_sha`, `training_compute_h100_hours`, `seeds_reported`. The reproducibility statement in the paper points at the release tag.

## Eval image (`Dockerfile.eval`) and digest pin procedure

The reproducible eval image is built from `Dockerfile.eval` at the repository root. It
pins:

- `nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04` as the base (RFC-0013 §D2).
- `uv` `0.8.17` (matches `.github/workflows/*.yml`).
- Python 3.12 (RFC-0013 §D3) from the `deadsnakes` PPA.
- The package's runtime dependencies via `uv sync --frozen --no-dev` against `uv.lock`.

### Local build

```sh
make eval-docker
# or
docker build -f Dockerfile.eval -t codingjepa-eval:test .
```

`.github/workflows/docker.yml` runs the same `docker build` on every push and PR that
touches `Dockerfile.eval`, `pyproject.toml`, or `uv.lock`. The workflow does **not**
push — pushing is a release-time action only.

### Digest pin procedure (release time)

For releases, the base image must be re-pinned by **digest**, not tag, so that a
rebuild from the release tag produces a byte-identical image:

1. Resolve the current digest of the floating tag once, on release-prep day:

   ```sh
   docker pull nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04
   docker inspect --format='{{index .RepoDigests 0}}' \
       nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04
   # → nvidia/cuda@sha256:<64-hex>
   ```

2. Edit `Dockerfile.eval` and replace the `FROM` line with the digest form:

   ```dockerfile
   FROM nvidia/cuda@sha256:<64-hex>
   ```

   Keep a comment above the line recording the human-readable tag the digest
   resolved from so reviewers can sanity-check the bump.

3. Rebuild locally and record the resulting eval-image digest in `MODEL_CARD.md`
   alongside `checkpoint_hash`, `manifest_hash`, and `index_id`. The release
   procedure (above) gates on that digest being committed before the tag is cut.

4. Push to GHCR with that digest as the immutable label
   (`ghcr.io/<org>/coding-jepa-eval@sha256:<digest>`); the floating
   `:v1.0.0` tag is convenience-only.

Between releases, the floating tag is acceptable on `main` so that security
updates to the base layer flow through without a manual bump. Re-pin to a digest
only at release time.

## Support windows

| Track | Support |
|---|---|
| `v1.0.x` | Bug fixes and security fixes for 6 months from `v1.0.0`. |
| `v1.x.y` (latest minor) | Active development; bug fixes go here first. |
| Pre-`v1` (none) | N/A; we do not commit to pre-release stability. |

The demo UI is **not** under support (RFC-0013 §D9). Bug reports are accepted but not SLA'd. The eval harness, model checkpoints, and Python package are the durable artifacts.

## Reproducibility guarantees

A given `(release_tag, manifest_hash, checkpoint_hash, index_id)` reproduces every number in that release's `results.json` to within bootstrap CI overlap. Specifically:

- Releases `v1.0.x` share the same `manifest_hash` and `tokenizer_hash`.
- A `v1.1.0` release that bumps the manifest (e.g., re-pinned commits) carries a fresh `manifest_hash` and a fresh `MODEL_CARD.md`.
- The `Reproducibility` changelog section says explicitly which numbers a given release changes.

If a release inadvertently breaks reproducibility, it is yanked and re-issued.

## Branching

- `main` is the trunk; protected; PR-only; CI gates required.
- Release branches `release/v1.0.x` are cut at tag time only when patch backports are needed.
- Long-lived feature branches are discouraged; spec changes land via RFC amendment first, code changes via short PRs after.

## RFC lifecycle

Per `docs/rfcs/README.md`:

- RFCs move Draft → Proposed → Locked → Superseded.
- A Locked RFC is amended via a dated entry; it is not silently rewritten.
- A superseded RFC stays in the repo with a pointer to the replacement.
- The implementation issue tracker (`docs/roadmap/IMPLEMENTATION.md`) labels each issue with `spec:rfc-NNNN`; superseding an RFC requires the tracker to be re-derived for the affected issues.

## Right-to-removal (RFC-0014 §D11)

- Removal requests are accepted via GitHub issue.
- An honored removal triggers a corpus re-version (`v1.0.x` → `v1.1.0` if data changes) and an `MODEL_CARD.md` amendment noting the change.
- We do not commit to a long SLA; this is a research artifact.

## What we do NOT promise

- Long-term support beyond 6 months on the v1 line.
- Stability of internal modules (anything not in `docs/spec/02-public-api.md`).
- Stability of demo UI markup, CSS, or HTMX templates.
- Wheel availability for arbitrary Python versions; we publish for Python 3.12 only (RFC-0013 §D3).
- Backwards-compatible support for arbitrary CUDA versions; we pin CUDA 12.4 (RFC-0013 §D2).
