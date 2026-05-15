"""CLI wiring for the Phase 1 data pipeline.

Subcommands (per docs/spec/02-public-api.md §`codingjepa data <step>`):

* ``codingjepa data mirror``   — clone/update the 10 source repos.
* ``codingjepa data chunk``    — chunk all Python files (RFC-0012 §D1-D3).
* ``codingjepa data pairs``    — walk commits and extract refactor pairs (RFC-0002 §D3-D4).
* ``codingjepa data dedup``    — exact + near-duplicate deduplication.
* ``codingjepa data splits``   — assign train/val/test splits.
* ``codingjepa data audit``    — per-repo audit + gate check.
* ``codingjepa data manifest`` — write the content-addressed manifest.
* ``codingjepa data all``      — run the full pipeline end-to-end.

Every step is idempotent and writes its output under ``data/``. Modules whose
implementations are still placeholders return ``0`` (success, no-op) so the
Makefile's ``make data`` target can run end-to-end during early development.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = [
    "add_data_subparser",
    "cmd_all",
    "cmd_audit",
    "cmd_chunk",
    "cmd_dedup",
    "cmd_manifest",
    "cmd_mirror",
    "cmd_pairs",
    "cmd_splits",
]


# --------------------------------------------------------------------------- #
# Subcommand handlers                                                         #
# --------------------------------------------------------------------------- #


def cmd_mirror(args: argparse.Namespace) -> int:
    """Run the ``data/mirror`` step.

    Clones each repo in :data:`codingjepa.data.mirror.REPO_REGISTRY` at its
    pinned commit and writes ``data/manifest.lock.json``.
    """

    from codingjepa.data import mirror

    output_dir = Path(getattr(args, "output_dir", "data/repos"))
    manifest_path = Path(getattr(args, "manifest", "data/manifest.lock.json"))
    mirror.mirror(output_dir=output_dir, manifest_path=manifest_path)
    print(f"mirror: ok ({manifest_path})")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    """Run the ``data/chunk`` step (RFC-0012 §D1-D3, §D11).

    For each repo listed in the manifest, walks the in-scope Python files and
    emits chunks to ``data/parsed/<repo>/<file_path>.chunks.parquet``. The
    Parquet writer is gated on :mod:`pyarrow`; if it is absent the step is a
    no-op so the rest of the pipeline can still run during early development.
    """

    # The full chunk-and-store wiring is intentionally not implemented yet:
    # the chunker exists (RFC-0012 §D1-D3) but the parquet writer and the
    # per-file output directory layout are scheduled for a follow-up. We keep
    # the CLI surface stable so consumers and CI can rely on the contract.
    print("chunk: not yet implemented (placeholder)")
    return 0


def cmd_pairs(args: argparse.Namespace) -> int:
    """Run the ``data/pairs`` step (RFC-0002 §D3-D4)."""

    print("pairs: not yet implemented (placeholder)")
    return 0


def cmd_dedup(args: argparse.Namespace) -> int:
    """Run the ``data/dedup`` step."""

    print("dedup: not yet implemented (placeholder)")
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    """Run the ``data/splits`` step."""

    print("splits: not yet implemented (placeholder)")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Run the ``data/audit`` step."""

    print("audit: not yet implemented (placeholder)")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    """Run the ``data/manifest`` step.

    Builds the v1 corpus manifest from :data:`codingjepa.data.mirror.REPO_REGISTRY`
    and writes it to ``data/manifest.lock.json``. Schema:
    ``data/schemas/manifest.schema.json``.
    """

    from codingjepa.data import manifest as manifest_mod
    from codingjepa.data import mirror

    manifest_path = Path(getattr(args, "manifest", "data/manifest.lock.json"))
    repos: list[dict[str, Any]] = []
    for entry in mirror.REPO_REGISTRY:
        owner, name = entry["name"].split("/", 1)
        repos.append(
            {
                "name": entry["name"],
                "url": entry["url"],
                "commit_sha": entry["commit_sha"],
                "license_spdx": entry["license_spdx"],
                "subset_paths": list(entry["subset_paths"]),
                "split": entry["split"],
                "py_files_in_scope": 0,
                "chunks_emitted": 0,
                "chunks_dropped": 0,
                "audit_path": f"data/audit/{owner}__{name}.json",
            }
        )

    manifest_mod.write_manifest(repos, output_path=manifest_path)
    print(f"manifest: ok ({manifest_path})")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Run the full Phase 1 pipeline (``make data``).

    Order matches the Makefile: mirror → chunk → pairs → dedup → splits →
    audit → manifest. The first non-zero return code short-circuits the
    pipeline.
    """

    # NOTE: handlers are looked up by name on this module so tests can
    # monkeypatch a single ``cmd_*`` and have it take effect here even though
    # ``cmd_all`` was imported separately.
    import codingjepa.data.cli as _self

    steps = ("mirror", "chunk", "pairs", "dedup", "splits", "audit", "manifest")
    for step in steps:
        handler = getattr(_self, f"cmd_{step}")
        rc: int = int(handler(args))
        if rc != 0:
            return rc
    return 0


# --------------------------------------------------------------------------- #
# argparse wiring                                                             #
# --------------------------------------------------------------------------- #


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Args shared by every Phase 1 subcommand."""

    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.lock.json"),
        help="Path to data/manifest.lock.json (default: %(default)s).",
    )


def add_data_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``data`` subcommand group to a top-level CLI parser.

    The parent CLI is responsible for dispatching ``args.func(args)`` after
    parsing.
    """

    data_parser = subparsers.add_parser(
        "data",
        help="Phase 1 data pipeline subcommands.",
        description=(
            "Run a step of the Phase 1 data pipeline. See "
            "docs/spec/02-public-api.md §`codingjepa data <step>`."
        ),
    )
    sub = data_parser.add_subparsers(dest="data_step", metavar="STEP")

    descriptions = {
        "mirror": "Clone/update the 10 source repos at pinned commits.",
        "chunk": "Chunk + normalize + tokenize Python files.",
        "pairs": "Walk commits and extract refactor pairs.",
        "dedup": "Exact + near-duplicate deduplication.",
        "splits": "Write data/splits/v1.lock.json.",
        "audit": "Per-repo audit + gate check.",
        "manifest": "Write data/manifest.lock.json.",
        "all": "Run the full pipeline end-to-end.",
    }

    for step, help_text in descriptions.items():
        p = sub.add_parser(step, help=help_text, description=help_text)
        _add_common_args(p)
        if step == "mirror":
            p.add_argument(
                "--output-dir",
                dest="output_dir",
                type=Path,
                default=Path("data/repos"),
                help="Directory to clone repos into (default: %(default)s).",
            )
        p.set_defaults(func=_dispatch_for(step), data_step=step)

    # Make ``data`` (no sub) print help and return non-zero.
    data_parser.set_defaults(func=lambda _a: _data_no_step(data_parser))


def _data_no_step(parser: argparse.ArgumentParser) -> int:
    """Handler for ``codingjepa data`` with no subcommand: print help, fail."""

    parser.print_help()
    return 2


def _dispatch_for(step: str) -> Callable[[argparse.Namespace], int]:
    """Return a callable that re-resolves ``cmd_<step>`` on this module.

    The indirection lets tests ``monkeypatch.setattr(data_cli, "cmd_mirror",
    stub)`` and have the patched stub take effect even though ``func`` was
    bound at parser-construction time.
    """

    def _run(args: argparse.Namespace) -> int:
        import codingjepa.data.cli as _self

        handler = getattr(_self, f"cmd_{step}")
        rc = handler(args)
        if rc is None:
            return 0
        if not isinstance(rc, int):
            return 0
        return rc

    _run.__name__ = f"_run_{step}"
    return _run
