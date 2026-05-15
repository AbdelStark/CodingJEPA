"""Top-level ``codingjepa`` command-line entry point.

This is the implementation behind ``python -m codingjepa`` and the
``codingjepa`` console script (see ``pyproject.toml`` ``[project.scripts]``).
The contract is documented in ``docs/spec/02-public-api.md``.

Subcommand groups are added by the modules that own them — currently only
``codingjepa.data.cli.add_data_subparser`` — so this file stays a thin
dispatcher.
"""

from __future__ import annotations

import argparse
import sys

from codingjepa import __version__


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level :class:`argparse.ArgumentParser`."""

    parser = argparse.ArgumentParser(
        prog="codingjepa",
        description="CodingJEPA: Joint-Embedding Predictive Architecture for Python code.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # Register subcommand groups. Each group's `add_*` function attaches an
    # `args.func` callable used by `main` to dispatch.
    from codingjepa.data.cli import add_data_subparser

    add_data_subparser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Top-level entry point.

    Parses ``argv`` and dispatches to the matching subcommand's ``func``. With
    no subcommand, prints a brief banner and returns 0 — backwards-compatible
    with the previous placeholder behavior so existing scripts that just call
    ``python -m codingjepa`` don't break.
    """

    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    try:
        args = parser.parse_args(args_list)
    except SystemExit as exc:
        # argparse exits the process on ``--help`` and ``--version``; surface
        # the exit code so :func:`main` can be tested without ``pytest.raises``.
        return int(exc.code or 0)

    if not args_list or getattr(args, "command", None) is None:
        # Preserve the legacy banner behavior of the placeholder CLI.
        print(f"codingjepa {__version__}")
        print("subcommands not yet implemented; see docs/spec/02-public-api.md")
        return 0

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    rc = func(args)
    if rc is None:
        return 0
    if not isinstance(rc, int):
        return 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
