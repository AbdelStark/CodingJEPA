#!/usr/bin/env python3
"""Generate ``NOTICE`` from ``data/manifest.lock.json`` (#30).

The frozen-dataset manifest enumerates every upstream repository included in
the CodingJEPA human-python-refactors dataset. Each entry carries a
``license_spdx``, ``commit_sha``, and (optionally) ``copyright`` field —
this script consolidates them into a single, deterministic ``NOTICE`` file
that ships alongside the dataset for license-compliance purposes.

Usage:
    python tools/notice_gen.py
    python tools/notice_gen.py --manifest path/to/manifest.lock.json --out NOTICE

Exit codes:
    0 — NOTICE written successfully.
    1 — manifest missing, malformed, or unreadable.

Notes:
    The full verbatim text of every SPDX license used by the corpus lives in
    ``LICENSES/<SPDX-ID>.txt``. The NOTICE points to the SPDX identifier; the
    license text itself is shipped separately to avoid bloating the NOTICE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("data/manifest.lock.json")
DEFAULT_OUTPUT = Path("NOTICE")

_HEADER = "CodingJEPA — human-python-refactors dataset"
_INTRO = "This dataset includes code derived from the following repositories:"
_COPYRIGHT_FALLBACK = "(see upstream repository)"


def _load_manifest(path: Path) -> dict[str, Any]:
    """Read ``path`` as JSON. Raises ``FileNotFoundError`` or ``ValueError``."""

    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON ({path}): {exc}") from exc
    if not isinstance(data, dict) or "repos" not in data:
        raise ValueError(f"manifest missing 'repos' key: {path}")
    return data


def _format_repo(repo: dict[str, Any]) -> str:
    """Render a single repo entry as a NOTICE section."""

    name = str(repo.get("name", "<unknown>"))
    spdx = str(repo.get("license_spdx", "<unknown>"))
    sha = str(repo.get("commit_sha", "<unknown>"))
    url = str(repo.get("url", "")).strip()

    # ``copyright`` is forward-compatible: not in the schema today, but the
    # NOTICE format reserves space for it. Fall back to the repo URL (which
    # always points at the upstream copyright holder) when absent.
    copyright_line = repo.get("copyright")
    if not isinstance(copyright_line, str) or not copyright_line.strip():
        copyright_line = f"{_COPYRIGHT_FALLBACK} {url}".strip()

    lines = [
        f"## {name}",
        f"License: {spdx}",
        f"Commit: {sha}",
        f"Copyright: {copyright_line}",
    ]
    return "\n".join(lines)


def render_notice(manifest: dict[str, Any]) -> str:
    """Render the full NOTICE text for ``manifest``. Pure function (no I/O)."""

    repos = manifest.get("repos") or []
    sections = [_format_repo(repo) for repo in repos]

    parts = [
        _HEADER,
        "",
        _INTRO,
        "",
    ]
    parts.extend(s + "\n" for s in sections)
    # End with a single trailing newline; no double-newline at EOF.
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"path to manifest.lock.json (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output NOTICE path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    try:
        manifest = _load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

    notice = render_notice(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(notice, encoding="utf-8")
    print(f"OK: wrote {len(manifest.get('repos') or [])} repo(s) to {args.out}")


if __name__ == "__main__":
    main()
