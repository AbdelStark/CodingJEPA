#!/usr/bin/env python3
"""Assert that the corpus contains no secrets. Exit 1 if any are found.

Uses :func:`codingjepa.data.secrets_scan.scan_chunk` directly so that this
script works with bare ``source`` columns (no chunk_id metadata required) and
also surfaces multi-pattern hits inside a single chunk.

Usage:
    python tools/assert_no_secrets.py data/pairs/train.parquet ...
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parquet_files", nargs="+", type=pathlib.Path)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    try:
        import pandas as pd  # type: ignore[import-untyped,unused-ignore]
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}")

    from codingjepa.data.secrets_scan import scan_chunk

    total_hits = 0
    for path in args.parquet_files:
        df = pd.read_parquet(path)
        if "source" not in df.columns:
            continue
        for idx, src in enumerate(df["source"].dropna().tolist()):
            chunk_id = f"{path.name}#{idx}"
            hits = scan_chunk(str(src), chunk_id=chunk_id)
            if hits:
                total_hits += len(hits)
                if args.verbose:
                    for h in hits:
                        print(f"MATCH: {h}")

    if total_hits:
        print(f"FAIL: {total_hits} secret matches found.", file=sys.stderr)
        sys.exit(1)
    print("OK: no secrets found.")


if __name__ == "__main__":
    main()
