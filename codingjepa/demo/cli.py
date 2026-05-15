"""CLI refactor command (RFC-0006 §D2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_INTENTS = [
    "extract-helper",
    "inline-helper",
    "comprehension-rewrite",
    "early-return",
    "guard-clause",
    "decompose-condition",
    "rename-for-clarity",
    "simplify-boolean",
    "NONE",
]


def build_refactor_parser() -> argparse.ArgumentParser:
    """Build the argument parser for `codingjepa refactor`."""
    parser = argparse.ArgumentParser(
        prog="codingjepa refactor",
        description="Refactor a Python snippet using CodingJEPA (RFC-0006 §D2).",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source",
        metavar="SOURCE",
        help="Python source string (use '-' to read from stdin).",
    )
    source_group.add_argument(
        "--file",
        metavar="FILE",
        type=Path,
        help="Path to a .py file to refactor.",
    )
    parser.add_argument(
        "--node",
        metavar="NODE",
        default=None,
        help="Qualified name of the function/class to refactor.",
    )
    parser.add_argument(
        "--intent",
        metavar="INTENT",
        default="NONE",
        choices=_INTENTS,
        help=("Refactoring intent, one of: " + ", ".join(_INTENTS) + ".  (default: NONE)"),
    )
    parser.add_argument(
        "--k",
        metavar="K",
        type=int,
        default=10,
        help="Top-k candidates to retrieve (default: 10).",
    )
    parser.add_argument(
        "--threshold",
        metavar="THRESHOLD",
        type=float,
        default=0.55,
        help="Confidence threshold below which no recommendation is returned (default: 0.55).",
    )
    parser.add_argument(
        "--out",
        metavar="OUT",
        type=Path,
        default=None,
        help="Output path for an HTML diff card (optional).",
    )
    return parser


def cmd_refactor(args: argparse.Namespace) -> int:
    """Run the refactor command.

    Exit codes: 0=success, 1=usage error, 2=no acceptable candidate.

    At this stage (no trained checkpoint) the command validates args,
    prints a "no checkpoint loaded" message, and exits 0. The actual
    inference path wires up when a checkpoint is present.
    """
    # Validate: exactly one of --source / --file must be given
    if args.source is None and args.file is None:
        print(
            "error: one of --source or --file is required.",
            file=sys.stderr,
        )
        return 1

    # Read source from file when --file is provided
    if args.file is not None:
        file_path: Path = args.file
        if not file_path.exists():
            print(f"error: file not found: {file_path}", file=sys.stderr)
            return 1
        source = file_path.read_text(encoding="utf-8")
    else:
        raw = args.source
        if raw == "-":
            source = sys.stdin.read()
        else:
            source = raw

    _ = source  # will be consumed by inference when a checkpoint is present

    print(
        "CodingJEPA: no checkpoint loaded. "
        "Pass a checkpoint via CODINGJEPA_CHECKPOINT or train one with `make pretrain`."
    )
    return 0


__all__ = ["build_refactor_parser", "cmd_refactor"]
