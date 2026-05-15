"""HTML diff gallery for the gold subset (RFC-0010 §D12, #120).

Each pair gets one self-contained HTML page rendered via
:func:`codingjepa.demo.diff.render_diff_html`; an ``index.html`` links to
every page so reviewers can browse the gallery offline.
"""

from __future__ import annotations

import html
import pathlib

from codingjepa.demo.diff import render_diff_html


def generate_diff_gallery(
    pairs: list[dict[str, str]],
    out_dir: pathlib.Path,
) -> pathlib.Path:
    """Write one HTML file per pair plus an ``index.html``. Returns ``out_dir``.

    Each pair dict must have ``before``, ``after``, ``intent`` and
    ``chunk_id`` keys. The index renders a simple table with intent and
    chunk_id columns so reviewers can sort by intent.
    """

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for i, pair in enumerate(pairs):
        page = render_diff_html(
            pair["before"],
            pair["after"],
            cosine=0.0,
            confidence=0.0,
            provenance=pair.get("chunk_id", ""),
        )
        page_name = f"pair-{i:03d}.html"
        (out_dir / page_name).write_text(page, encoding="utf-8")
        intent = html.escape(pair.get("intent", ""))
        chunk_id = html.escape(pair.get("chunk_id", ""))
        rows.append(
            f"<tr><td>{i:03d}</td><td>{intent}</td><td>{chunk_id}</td>"
            f'<td><a href="{page_name}">view</a></td></tr>'
        )

    index = _render_index(rows)
    (out_dir / "index.html").write_text(index, encoding="utf-8")
    return out_dir


def _render_index(rows: list[str]) -> str:
    body = "\n".join(rows) if rows else "<tr><td colspan=4>No pairs.</td></tr>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CodingJEPA diff gallery</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; min-width: 60ch; }}
td, th {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
th {{ background: #f3f3f3; }}
</style>
</head>
<body>
<h1>CodingJEPA diff gallery</h1>
<table>
<thead><tr><th>#</th><th>Intent</th><th>Chunk ID</th><th>Page</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
</body>
</html>
"""


__all__ = ["generate_diff_gallery"]
