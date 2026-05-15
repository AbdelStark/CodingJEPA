"""Unified-diff renderer using difflib + pygments (RFC-0006 §D4)."""

from __future__ import annotations

import difflib

from pygments import highlight  # type: ignore[import-untyped]
from pygments.formatters import (  # type: ignore[import-untyped]
    HtmlFormatter,
    TerminalTrueColorFormatter,
)
from pygments.lexers import PythonLexer  # type: ignore[import-untyped]


def render_diff_terminal(before: str, after: str, n: int = 3) -> str:
    """Render a unified diff with terminal ANSI color using pygments.

    Returns the diff as a string ready to print to stdout.
    Lines are syntax-highlighted using PythonLexer where applicable.
    """
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
            n=n,
        )
    )
    if not diff_lines:
        return ""
    diff_text = "".join(diff_lines)
    formatter = TerminalTrueColorFormatter(style="monokai")
    return str(highlight(diff_text, PythonLexer(), formatter))


def render_diff_html(
    before: str,
    after: str,
    *,
    cosine: float = 0.0,
    confidence: float = 0.0,
    provenance: str = "",
    n: int = 3,
) -> str:
    """Render a self-contained HTML diff card (RFC-0006 §D5).

    Returns a complete HTML string with inlined CSS and no external requests.
    Includes cosine/confidence/provenance metadata.
    """
    formatter = HtmlFormatter(full=False, style="monokai", cssclass="highlight")
    css = formatter.get_style_defs(".highlight")

    _fmt = HtmlFormatter(style="monokai", cssclass="highlight")
    before_html = highlight(before, PythonLexer(), _fmt)
    after_html = highlight(after, PythonLexer(), _fmt)

    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
            n=n,
        )
    )
    diff_text = "".join(diff_lines) if diff_lines else "(no changes)"
    diff_html = highlight(diff_text, PythonLexer(), _fmt)

    meta_parts = [f"cos {cosine:.2f}", f"conf {confidence:.2f}"]
    if provenance:
        meta_parts.append(provenance)
    meta_str = " · ".join(meta_parts)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CodingJEPA diff card</title>
<style>
body {{
    font-family: 'Courier New', Courier, monospace;
    background: #1a1a1a;
    color: #f8f8f2;
    margin: 0;
    padding: 1rem;
}}
.card {{
    background: #272822;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid #3d3d3d;
}}
.meta {{
    font-size: 0.85rem;
    color: #a6e22e;
    margin-bottom: 0.75rem;
    padding: 0.25rem 0.5rem;
    background: #1e1e1e;
    border-radius: 3px;
    display: inline-block;
}}
.diff-section h3, .side-by-side h3 {{
    color: #66d9ef;
    font-size: 0.9rem;
    margin: 0 0 0.5rem 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.side-by-side {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-bottom: 1rem;
}}
.side-by-side .panel {{
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    overflow: auto;
}}
.diff-section {{
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    overflow: auto;
}}
.highlight {{
    background: #272822 !important;
    margin: 0;
    padding: 0.5rem;
}}
pre {{
    margin: 0;
}}
{css}
</style>
</head>
<body>
<div class="card">
  <div class="meta">{meta_str}</div>
  <div class="side-by-side">
    <div class="panel">
      <h3>Before</h3>
      {before_html}
    </div>
    <div class="panel">
      <h3>After</h3>
      {after_html}
    </div>
  </div>
  <div class="diff-section">
    <h3>Unified diff</h3>
    {diff_html}
  </div>
</div>
</body>
</html>"""
    return page


__all__ = ["render_diff_html", "render_diff_terminal"]
