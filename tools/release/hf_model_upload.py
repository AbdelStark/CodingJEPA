#!/usr/bin/env python3
"""Upload CodingJEPA model artifacts to the HuggingFace Hub (#126).

Uploads the four model components (encoder, projector, predictor, pred_proj),
the tokenizer artifacts, ``MODEL_CARD.md`` (renamed to ``README.md`` on Hub),
and the ``LICENSES/`` directory to ``CodingJEPA/coding-jepa-v1``.

Design notes
------------
- ``HF_TOKEN`` is read from the operator environment; the token value is
  **never** logged or printed (see ``docs/spec/06-security.md`` §Secrets handling
  and RFC-0013 §D8).
- Re-uploads are idempotent: each file's SHA-256 is compared against the
  remote object's current hash; matching files are skipped.
- ``--dry-run`` lists the full upload plan without contacting the network and
  exits 0 even if some required files are missing (so operators can use it as
  a pre-flight check on partial checkpoints).
- A real upload exits 1 if any of the four ``*.safetensors`` files is missing.

Usage
-----
    python tools/release/hf_model_upload.py --checkpoint runs/v1.0 --dry-run
    HF_TOKEN=hf_xxx python tools/release/hf_model_upload.py \\
        --checkpoint runs/v1.0 --repo-id CodingJEPA/coding-jepa-v1
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import pathlib
import sys

DEFAULT_REPO_ID = "CodingJEPA/coding-jepa-v1"
REQUIRED_SAFETENSORS = ("encoder", "projector", "predictor", "pred_proj")


@dataclasses.dataclass(frozen=True)
class UploadEntry:
    """One planned upload: local ``source`` → remote ``target`` path on Hub."""

    source: pathlib.Path
    target: str
    required: bool = False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Local checkpoint directory containing *.safetensors files",
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF Hub repository ID (default: {DEFAULT_REPO_ID})",
    )
    p.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Project root containing MODEL_CARD.md, tokenizer/, LICENSES/ "
        "(default: current working directory)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List every local file and its target HF path; do not upload",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the HF repo as private",
    )
    p.add_argument(
        "--commit-message",
        default="Upload CodingJEPA v1.0 weights, tokenizer, and model card",
        help="Commit message for the HF Hub upload",
    )
    return p


def file_sha256(path: pathlib.Path, *, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of ``path`` (streamed; safe on large weights)."""

    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def build_upload_plan(
    *, checkpoint: pathlib.Path, repo_root: pathlib.Path
) -> list[UploadEntry]:
    """Enumerate every (local source, HF target) pair this script will push.

    The plan deliberately enumerates *expected* sources too: missing required
    files are flagged but kept in the plan so ``--dry-run`` can act as a
    pre-flight check.
    """

    plan: list[UploadEntry] = []

    # 1. Required safetensors at the checkpoint root → repo root.
    for name in REQUIRED_SAFETENSORS:
        src = checkpoint / f"{name}.safetensors"
        plan.append(UploadEntry(source=src, target=f"{name}.safetensors", required=True))

    # 2. Tokenizer directory → tokenizer/.
    tok_dir = repo_root / "tokenizer"
    if tok_dir.is_dir():
        for f in sorted(tok_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(tok_dir).as_posix()
                plan.append(UploadEntry(source=f, target=f"tokenizer/{rel}"))

    # 3. MODEL_CARD.md → README.md (HF Hub convention).
    model_card = repo_root / "MODEL_CARD.md"
    if model_card.exists():
        plan.append(UploadEntry(source=model_card, target="README.md"))

    # 4. LICENSES/ → LICENSES/.
    licenses_dir = repo_root / "LICENSES"
    if licenses_dir.is_dir():
        for f in sorted(licenses_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(licenses_dir).as_posix()
                plan.append(UploadEntry(source=f, target=f"LICENSES/{rel}"))

    return plan


def _remote_sha256(api: object, *, repo_id: str, target: str) -> str | None:
    """Fetch the SHA-256 of the file at ``target`` on Hub, or ``None`` if missing.

    Uses ``HfApi.get_paths_info`` so we never download the blob. Returns ``None``
    on any error (treated as "not present" — re-upload).
    """

    try:
        infos = api.get_paths_info(repo_id=repo_id, paths=[target])  # type: ignore[attr-defined]
    except Exception:
        return None
    for info in infos:
        # ``RepoFile`` carries an LFS pointer when present.
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            sha = getattr(lfs, "sha256", None)
            if sha:
                return str(sha)
        # Small (non-LFS) files expose a blob_id which is a git OID, not sha256.
        # We can't compare directly — fall through so the caller re-uploads.
    return None


def _print_plan_dry_run(plan: list[UploadEntry], *, repo_id: str) -> None:
    print(f"Dry-run upload plan for HF repo: {repo_id}")
    print("-" * 72)
    for entry in plan:
        if not entry.source.exists():
            marker = "[DRY-RUN MISSING]" if entry.required else "[DRY-RUN SKIP]"
            print(f"{marker} {entry.source} -> {entry.target}")
            continue
        sha = file_sha256(entry.source)
        print(f"[DRY-RUN] {entry.source} -> {entry.target} (sha256: {sha[:12]}...)")


def _validate_required(plan: list[UploadEntry]) -> list[UploadEntry]:
    """Return the subset of plan entries whose required source is missing."""

    return [e for e in plan if e.required and not e.source.exists()]


def _do_upload(
    plan: list[UploadEntry],
    *,
    repo_id: str,
    token: str | None,
    private: bool,
    commit_message: str,
) -> int:
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run: pip install huggingface-hub", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    uploaded = 0
    skipped = 0
    for entry in plan:
        if not entry.source.exists():
            print(f"[SKIP-MISSING] {entry.source} -> {entry.target}")
            continue

        local_sha = file_sha256(entry.source)
        remote_sha = _remote_sha256(api, repo_id=repo_id, target=entry.target)
        if remote_sha is not None and remote_sha == local_sha:
            print(f"[SKIP-MATCH] {entry.source} -> {entry.target} (sha: {local_sha[:12]}...)")
            skipped += 1
            continue

        api.upload_file(
            path_or_fileobj=str(entry.source),
            path_in_repo=entry.target,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"[UPLOADED] {entry.source} -> {entry.target} (sha: {local_sha[:12]}...)")
        uploaded += 1

    print("-" * 72)
    print(f"Done. uploaded={uploaded} skipped={skipped} repo=https://huggingface.co/{repo_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    checkpoint = args.checkpoint.resolve()
    repo_root = args.repo_root.resolve()
    if not checkpoint.exists():
        print(f"Checkpoint directory does not exist: {checkpoint}", file=sys.stderr)
        return 1

    plan = build_upload_plan(checkpoint=checkpoint, repo_root=repo_root)

    if args.dry_run:
        _print_plan_dry_run(plan, repo_id=args.repo_id)
        missing = _validate_required(plan)
        if missing:
            print(
                f"Note: {len(missing)} required file(s) missing — would fail a real upload.",
                file=sys.stderr,
            )
        return 0

    missing = _validate_required(plan)
    if missing:
        print(
            "FAIL: missing required artifacts:\n  - "
            + "\n  - ".join(str(e.source) for e in missing),
            file=sys.stderr,
        )
        return 1

    # NOTE: we deliberately read HF_TOKEN here but never echo it.
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "FAIL: HF_TOKEN is not set in the environment. "
            "Export it (without logging it) before running.",
            file=sys.stderr,
        )
        return 1

    return _do_upload(
        plan,
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    sys.exit(main())
