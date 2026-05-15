from __future__ import annotations

import sys

from codingjepa import __version__


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-V", "--version"}:
        print(__version__)
        return 0
    print(f"codingjepa {__version__}")
    print("subcommands not yet implemented; see docs/spec/02-public-api.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
