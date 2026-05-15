"""RESULTS-MEMO.md generator (RFC-0010 §D6, #119).

The memo renders a structured report in the section order locked by RFC-0010
§D6. When a benchmark is missing from ``results.json`` we emit a
``> _No data available._`` placeholder so the memo always renders the full
template — operators can spot gaps at a glance.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

NO_DATA = "> _No data available._"


def generate_memo(
    results_json: pathlib.Path | str,
    out_path: pathlib.Path | str,
) -> str:
    """Read ``results_json``, render ``RESULTS-MEMO.md``. Returns the text."""

    path = pathlib.Path(results_json)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = ["# CodingJEPA — Results Memo", ""]
    lines += _tldr(data)
    lines += _setup(data)
    lines += _main_results(data)
    lines += _per_intent(data)
    lines += _robustness(data)
    lines += _ood(data)
    lines += _probes(data)
    lines += _human(data)
    lines += _ablations(data)
    lines += _failure_modes(data)
    lines += _limitations(data)
    text = "\n".join(lines) + "\n"
    pathlib.Path(out_path).write_text(text, encoding="utf-8")
    return text


def _find(data: dict[str, Any], bid: str) -> dict[str, Any] | None:
    for bm in data.get("benchmarks", []):
        if bm.get("benchmark_id") == bid:
            return dict(bm)
    return None


def _metric(bm: dict[str, Any] | None, key: str, default: str = "—") -> str:
    if bm is None:
        return default
    val = bm.get("metrics", {}).get(key)
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _tldr(data: dict[str, Any]) -> list[str]:
    lines = ["## TL;DR", ""]
    ret = _find(data, "CJ-RET-100")
    if ret is None:
        lines.append(NO_DATA)
    else:
        r5 = _metric(ret, "R@5")
        mrr = _metric(ret, "MRR")
        lines.append(
            f"CodingJEPA achieves R@5={r5} and MRR={mrr} on CJ-RET-100 "
            f"(pool=100). See sections below for the full breakdown."
        )
    lines.append("")
    return lines


def _setup(data: dict[str, Any]) -> list[str]:
    lines = [
        "## Setup",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Schema version | {data.get('schema_version', '—')} |",
        f"| Global seed | {data.get('global_seed', '—')} |",
        f"| Benchmarks reported | {len(data.get('benchmarks', []))} |",
        "",
    ]
    return lines


def _main_results(data: dict[str, Any]) -> list[str]:
    lines = ["## Main results", ""]
    ret100 = _find(data, "CJ-RET-100")
    ret1k = _find(data, "CJ-RET-1k")
    execb = _find(data, "CJ-EXEC")
    if ret100 is None and ret1k is None and execb is None:
        lines.append(NO_DATA)
        lines.append("")
        return lines
    lines.extend(
        [
            "| Benchmark | R@1 | R@5 | R@10 | MRR | Pass rate |",
            "| --- | --- | --- | --- | --- | --- |",
            "| CJ-RET-100 | "
            f"{_metric(ret100, 'R@1')} | {_metric(ret100, 'R@5')} | "
            f"{_metric(ret100, 'R@10')} | {_metric(ret100, 'MRR')} | — |",
            "| CJ-RET-1k | "
            f"{_metric(ret1k, 'R@1')} | {_metric(ret1k, 'R@5')} | "
            f"{_metric(ret1k, 'R@10')} | {_metric(ret1k, 'MRR')} | — |",
            f"| CJ-EXEC | — | — | — | — | {_metric(execb, 'pass_rate')} |",
            "",
        ]
    )
    return lines


def _per_intent(data: dict[str, Any]) -> list[str]:
    lines = ["## Per-intent retrieval (R@5)", ""]
    intent = _find(data, "CJ-INTENT")
    if intent is None:
        lines.append(NO_DATA)
        lines.append("")
        return lines
    lines.extend(
        [
            "| Variant | R@5 |",
            "| --- | --- |",
            f"| conditioned | {_metric(intent, 'R@5_conditioned')} |",
            f"| unconditioned | {_metric(intent, 'R@5_unconditioned')} |",
            f"| ΔR@5 | {_metric(intent, 'delta_R5')} |",
            "",
        ]
    )
    return lines


def _robustness(data: dict[str, Any]) -> list[str]:
    lines = ["## Robustness probes", ""]
    rows = [
        ("CJ-ROB-FMT", _find(data, "CJ-ROB-FMT")),
        ("CJ-ROB-RENAME", _find(data, "CJ-ROB-RENAME")),
        ("CJ-ROB-DOC", _find(data, "CJ-ROB-DOC")),
    ]
    if all(bm is None for _id, bm in rows):
        lines.append(NO_DATA)
        lines.append("")
        return lines
    lines.append("| Perturbation | Rank-change % | Mean cosine drift |")
    lines.append("| --- | --- | --- |")
    for bid, bm in rows:
        lines.append(
            f"| {bid} | {_metric(bm, 'rank_change_pct')} | {_metric(bm, 'mean_cosine_drift')} |"
        )
    lines.append("")
    return lines


def _ood(data: dict[str, Any]) -> list[str]:
    lines = ["## OOD probe", ""]
    bm = _find(data, "CJ-OOD")
    if bm is None:
        lines.append(NO_DATA)
        lines.append("")
        return lines
    lines.extend(
        [
            "| Metric | Value |",
            "| --- | --- |",
            f"| R@10 | {_metric(bm, 'R@10')} |",
            "",
        ]
    )
    return lines


def _probes(data: dict[str, Any]) -> list[str]:
    lines = ["## Code understanding (linear probes)", ""]
    rows = [
        ("CJ-PROBE-NAME", _find(data, "CJ-PROBE-NAME"), "top1_accuracy"),
        ("CJ-PROBE-DEFECT", _find(data, "CJ-PROBE-DEFECT"), "f1"),
        ("CJ-PROBE-CLONE", _find(data, "CJ-PROBE-CLONE"), "f1"),
    ]
    if all(bm is None for _id, bm, _key in rows):
        lines.append(NO_DATA)
        lines.append("")
        return lines
    lines.append("| Probe | Metric | Value |")
    lines.append("| --- | --- | --- |")
    for bid, bm, key in rows:
        lines.append(f"| {bid} | {key} | {_metric(bm, key)} |")
    lines.append("")
    return lines


def _human(data: dict[str, Any]) -> list[str]:
    lines = ["## Human review", ""]
    bm = _find(data, "CJ-HUMAN")
    if bm is None:
        lines.append(NO_DATA)
        lines.append("")
        return lines
    lines.extend(
        [
            "| Metric | Value |",
            "| --- | --- |",
            f"| Mean Likert | {_metric(bm, 'mean_likert')} |",
            f"| Cohen κ | {_metric(bm, 'cohen_kappa')} |",
            f"| Status | {_metric(bm, 'status')} |",
            "",
        ]
    )
    return lines


def _ablations(_data: dict[str, Any]) -> list[str]:
    return [
        "## Ablations",
        "",
        "> Ablation matrix (RFC-0005 §D6) — populated when seed sweeps are committed.",
        "",
    ]


def _failure_modes(_data: dict[str, Any]) -> list[str]:
    return [
        "## Failure modes",
        "",
        "> Worst-50 retrievals per intent are embedded under `results/confusions/`.",
        "",
    ]


def _limitations(_data: dict[str, Any]) -> list[str]:
    return [
        "## Limitations",
        "",
        "- Corpus contamination risks for pretrained baselines (RFC-0010 §D11).",
        "- OOD scope is restricted to cpython `Lib/` 3.11→3.13 dev cycle.",
        "- Statistical caveats: ≥ 3 seeds, 95% CI via 1000-resample bootstrap.",
        "",
    ]


__all__ = ["generate_memo"]
