"""Worst-50 error pages per intent (RFC-0010 §D12, #120).

For each intent we emit one HTML page listing the worst retrievals (where
the model placed the gold candidate beyond top-K). The ``index.html`` links
to every intent page.
"""

from __future__ import annotations

import html
import pathlib

_PAGE_CSS = """
body { font-family: sans-serif; margin: 2rem; color: #222; }
h1 { font-size: 1.4rem; }
table { border-collapse: collapse; min-width: 60ch; }
td, th { border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }
th { background: #f3f3f3; }
"""


def generate_confusions(
    errors: dict[str, list[dict[str, str]]],
    out_dir: pathlib.Path,
) -> pathlib.Path:
    """Write one HTML page per intent plus an ``index.html``. Returns ``out_dir``."""

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[str] = []
    for intent, rows in errors.items():
        page_name = _safe_filename(intent) + ".html"
        page = _render_intent_page(intent, rows)
        (out_dir / page_name).write_text(page, encoding="utf-8")
        index_rows.append(
            f"<tr><td>{html.escape(intent)}</td><td>{len(rows)}</td>"
            f'<td><a href="{page_name}">view</a></td></tr>'
        )

    index = _render_index(index_rows)
    (out_dir / "index.html").write_text(index, encoding="utf-8")
    return out_dir


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _render_intent_page(intent: str, rows: list[dict[str, str]]) -> str:
    table_rows = (
        "\n".join(
            f"<tr><td>{html.escape(r.get('rank', '?'))}</td>"
            f"<td><pre>{html.escape(r.get('query', ''))}</pre></td>"
            f"<td><pre>{html.escape(r.get('retrieved', ''))}</pre></td></tr>"
            for r in rows
        )
        if rows
        else "<tr><td colspan=3>No errors.</td></tr>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Confusions — {html.escape(intent)}</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
<h1>Worst retrievals for intent: {html.escape(intent)}</h1>
<table>
<thead><tr><th>Rank</th><th>Query</th><th>Retrieved (top-1)</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>
"""


def _render_index(rows: list[str]) -> str:
    body = "\n".join(rows) if rows else "<tr><td colspan=3>No intents.</td></tr>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CodingJEPA confusions</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
<h1>CodingJEPA — confusions per intent</h1>
<table>
<thead><tr><th>Intent</th><th># errors</th><th>Page</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
</body>
</html>
"""


__all__ = ["generate_confusions"]
