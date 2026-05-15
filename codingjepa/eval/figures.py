"""PDF figure generator for ``paper/`` (RFC-0010 §D12, #121).

``matplotlib`` is optional — when it is not installed we return an empty
list, which keeps the eval harness functional in minimal environments and
on CI runners that do not pin the paper deps.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any


def generate_figures(
    results_json: pathlib.Path | str,
    out_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Generate bar-chart PDFs for the paper. Returns list of generated paths.

    Currently emits:
      - ``retrieval_at_k.pdf`` — R@1, R@5, R@10, MRR for CJ-RET-100.
      - ``robustness.pdf`` — rank-change % per robustness perturbation.

    Falls back to a no-op stub when matplotlib is not installed.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend; works on CI without DISPLAY.
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    path = pathlib.Path(results_json)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[pathlib.Path] = []

    ret = _find(data, "CJ-RET-100")
    if ret is not None:
        ret_path = out_dir / "retrieval_at_k.pdf"
        _plot_retrieval(plt, ret, ret_path)
        generated.append(ret_path)

    rob_rows = [
        ("CJ-ROB-FMT", _find(data, "CJ-ROB-FMT")),
        ("CJ-ROB-RENAME", _find(data, "CJ-ROB-RENAME")),
        ("CJ-ROB-DOC", _find(data, "CJ-ROB-DOC")),
    ]
    if any(bm is not None for _id, bm in rob_rows):
        rob_path = out_dir / "robustness.pdf"
        _plot_robustness(plt, rob_rows, rob_path)
        generated.append(rob_path)

    return generated


def _find(data: dict[str, Any], bid: str) -> dict[str, Any] | None:
    for bm in data.get("benchmarks", []):
        if bm.get("benchmark_id") == bid:
            return dict(bm)
    return None


def _plot_retrieval(plt: Any, ret: dict[str, Any], out_path: pathlib.Path) -> None:
    metrics = ret["metrics"]
    labels = ["R@1", "R@5", "R@10", "MRR"]
    values = [float(metrics.get(k, 0.0)) for k in labels]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("CJ-RET-100")
    ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def _plot_robustness(
    plt: Any,
    rows: list[tuple[str, dict[str, Any] | None]],
    out_path: pathlib.Path,
) -> None:
    labels: list[str] = []
    values: list[float] = []
    for bid, bm in rows:
        if bm is None:
            continue
        labels.append(bid)
        values.append(float(bm["metrics"].get("rank_change_pct", 0.0)))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values)
    ax.set_ylim(0.0, 100.0)
    ax.set_title("Robustness — rank-change %")
    ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


__all__ = ["generate_figures"]
