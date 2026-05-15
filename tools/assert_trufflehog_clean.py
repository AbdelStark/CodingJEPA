#!/usr/bin/env python3
"""Assert truffleHog finds no secrets. Falls back gracefully if not installed.

Usage:
    python tools/assert_trufflehog_clean.py [path]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", default=".", nargs="?")
    p.add_argument(
        "--no-fail-on-missing",
        action="store_true",
        help="Exit 0 if trufflehog is not installed",
    )
    args = p.parse_args()

    trufflehog = shutil.which("trufflehog")
    if not trufflehog:
        if args.no_fail_on_missing:
            print("SKIP: trufflehog not found; skipping.")
            return
        sys.exit("FAIL: trufflehog not installed. Install it or pass --no-fail-on-missing.")

    result = subprocess.run(
        [trufflehog, "filesystem", args.path, "--fail"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    print("OK: trufflehog found no secrets.")


if __name__ == "__main__":
    main()
