"""HTMX template fragments (RFC-0006 §D3)."""

from __future__ import annotations

# HTMX is loaded from unpkg CDN (https://unpkg.com/htmx.org@1.9.12).
# Per spec/06 supply-chain rule, CDN references are acceptable for dev-mode
# tooling and demo surfaces. Production deployments should vendor the asset.
_HTMX_CDN = "https://unpkg.com/htmx.org@1.9.12"

_INTENTS = [
    "NONE",
    "extract-helper",
    "inline-helper",
    "comprehension-rewrite",
    "early-return",
    "guard-clause",
    "decompose-condition",
    "rename-for-clarity",
    "simplify-boolean",
]

_FORM_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CodingJEPA Demo</title>
  <script src="{htmx_cdn}"></script>
  <style>
    body {{
      font-family: 'Courier New', Courier, monospace;
      background: #1a1a1a;
      color: #f8f8f2;
      max-width: 900px;
      margin: 2rem auto;
      padding: 0 1rem;
    }}
    h1 {{ color: #66d9ef; font-size: 1.5rem; margin-bottom: 0.25rem; }}
    .subtitle {{ color: #75715e; font-size: 0.85rem; margin-bottom: 1.5rem; }}
    label {{ display: block; color: #a6e22e; margin-bottom: 0.25rem; font-size: 0.9rem; }}
    textarea, select, input[type="number"] {{
      width: 100%;
      background: #272822;
      color: #f8f8f2;
      border: 1px solid #3d3d3d;
      border-radius: 4px;
      padding: 0.5rem;
      font-family: inherit;
      font-size: 0.9rem;
      box-sizing: border-box;
    }}
    textarea {{ min-height: 200px; resize: vertical; }}
    .row {{ display: grid; grid-template-columns: 2fr 1fr; gap: 1rem; margin-top: 1rem; }}
    button {{
      margin-top: 1rem;
      background: #66d9ef;
      color: #1a1a1a;
      border: none;
      border-radius: 4px;
      padding: 0.6rem 1.5rem;
      font-family: inherit;
      font-size: 0.95rem;
      font-weight: bold;
      cursor: pointer;
    }}
    button:hover {{ background: #a6e22e; }}
    #results {{ margin-top: 2rem; }}
    .htmx-indicator {{ color: #75715e; display: none; }}
    .htmx-request .htmx-indicator {{ display: inline; }}
  </style>
</head>
<body>
  <h1>CodingJEPA Demo</h1>
  <p class="subtitle">Joint-Embedding Predictive Architecture for Python code refactoring.</p>

  <form hx-post="/refactor" hx-target="#results" hx-swap="innerHTML">
    <label for="source">Python source</label>
    <textarea id="source" name="source"
      placeholder="Paste a Python function or class here..."></textarea>

    <div class="row">
      <div>
        <label for="intent">Intent</label>
        <select id="intent" name="intent">
          {intent_options}
        </select>
      </div>
      <div>
        <label for="k">Top-k</label>
        <input id="k" name="k" type="number" value="10" min="1" max="50">
      </div>
    </div>

    <button type="submit">
      Refactor
      <span class="htmx-indicator">&#8230;</span>
    </button>
  </form>

  <div id="results"></div>
</body>
</html>
"""


def render_form() -> str:
    """Return the main form HTML page."""
    intent_options = "\n          ".join(
        f'<option value="{intent}">{intent}</option>' for intent in _INTENTS
    )
    return _FORM_PAGE.format(htmx_cdn=_HTMX_CDN, intent_options=intent_options)


def render_candidates(candidates: list[dict[str, object]]) -> str:
    """Return an HTMX-friendly candidates fragment."""
    if not candidates:
        return "<div id='results'><p>No candidates found.</p></div>"

    items: list[str] = []
    for i, candidate in enumerate(candidates, start=1):
        cosine = float(candidate.get("cosine", 0.0))  # type: ignore[arg-type]
        confidence = float(candidate.get("confidence", 0.0))  # type: ignore[arg-type]
        provenance = str(candidate.get("provenance", ""))
        after = str(candidate.get("after", ""))
        accepted = bool(candidate.get("accepted", True))
        reason = str(candidate.get("reason", ""))

        status_class = "accepted" if accepted else "rejected"
        status_icon = "&#10003;" if accepted else "&#10007;"
        reason_html = f"<span class='reason'>{reason}</span>" if reason else ""

        items.append(
            f"<div class='candidate {status_class}'>"
            f"<div class='candidate-header'>"
            f"<span class='rank'>#{i}</span>"
            f"<span class='status-icon'>{status_icon}</span>"
            f"<span class='meta'>cos {cosine:.2f} · conf {confidence:.2f}"
            + (f" · {provenance}" if provenance else "")
            + f"</span>{reason_html}</div>"
            f"<pre><code>{after}</code></pre>"
            f"</div>"
        )

    inner = "\n".join(items)
    style = """<style>
.candidate { background: #272822; border: 1px solid #3d3d3d; border-radius: 4px;
  margin-bottom: 0.75rem; padding: 0.75rem; }
.candidate.rejected { opacity: 0.55; border-color: #f92672; }
.candidate-header { display: flex; gap: 0.75rem; align-items: center;
  margin-bottom: 0.5rem; font-size: 0.85rem; }
.rank { color: #66d9ef; font-weight: bold; }
.status-icon { color: #a6e22e; }
.candidate.rejected .status-icon { color: #f92672; }
.meta { color: #75715e; }
.reason { color: #fd971f; }
pre { background: #1a1a1a; border-radius: 3px; padding: 0.5rem; overflow: auto; margin: 0; }
code { font-family: 'Courier New', Courier, monospace; font-size: 0.85rem; }
</style>"""
    return f"<div id='results'>{style}{inner}</div>"


__all__ = ["render_candidates", "render_form"]
