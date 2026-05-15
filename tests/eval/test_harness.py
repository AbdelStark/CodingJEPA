"""Eval harness smoke tests — 10-example fixture (#123).

Covers RFC-0010 acceptance criteria: every benchmark has at least one unit
test on a 10-example fixture; `make eval` orchestrator emits a valid
`results.json`; the memo generator round-trips it.
"""

from __future__ import annotations

import os

# Suppress macOS OpenMP duplicate-lib crash when numpy / torch / faiss all
# pull in libomp. See tests/inference/test_retrieve.py for the same fix.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json  # noqa: E402
import pathlib  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from codingjepa.eval.benchmarks import (  # noqa: E402
    ExecBenchmark,
    HumanBenchmark,
    IntentBenchmark,
    OodBenchmark,
    ProbeCloneBenchmark,
    ProbeDefectBenchmark,
    ProbeNameBenchmark,
    RetBenchmark1k,
    RetBenchmark100,
    RobDocBenchmark,
    RobFmtBenchmark,
    RobRenameBenchmark,
)
from codingjepa.eval.confusions import generate_confusions  # noqa: E402
from codingjepa.eval.diff_gallery import generate_diff_gallery  # noqa: E402
from codingjepa.eval.figures import generate_figures  # noqa: E402
from codingjepa.eval.harness import Benchmark, run_suite  # noqa: E402
from codingjepa.eval.memo import generate_memo  # noqa: E402
from codingjepa.eval.sandbox import run_in_sandbox  # noqa: E402

# ---- Benchmark base class --------------------------------------------------


def test_benchmark_base_class_is_abstract() -> None:
    """The Benchmark base class is an ABC and cannot be instantiated."""

    with pytest.raises(TypeError):
        Benchmark()  # type: ignore[abstract]


def test_benchmark_seed_is_derived_from_id_and_global() -> None:
    """Two benchmarks with different IDs get different seeds for the same global."""

    a = RetBenchmark100(n_queries=10)
    b = RetBenchmark1k(n_queries=10)
    assert a.seed != b.seed
    # Same id + same global_seed → same seed
    a2 = RetBenchmark100(n_queries=10, global_seed=0)
    assert a.seed == a2.seed


# ---- CJ-RET-100 ------------------------------------------------------------


def test_ret100_prepare_and_run(tmp_path: pathlib.Path) -> None:
    bm = RetBenchmark100(n_queries=10)
    bm.prepare(tmp_path)
    result = bm.run()
    assert result.benchmark_id == "CJ-RET-100"
    for k in ("R@1", "R@5", "R@10", "MRR"):
        v = result.metrics[k]
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0
    assert result.metrics["MRR"] > 0.0
    assert result.metrics["n_queries"] == 10
    assert result.metrics["pool_size"] == 100


def test_ret1k_prepare_and_run(tmp_path: pathlib.Path) -> None:
    bm = RetBenchmark1k(n_queries=10)
    bm.prepare(tmp_path)
    result = bm.run()
    assert result.benchmark_id == "CJ-RET-1k"
    assert result.metrics["pool_size"] == 1000
    assert isinstance(result.metrics["R@10"], float)
    assert 0.0 <= float(result.metrics["R@10"]) <= 1.0


# ---- CJ-INTENT -------------------------------------------------------------


def test_intent_benchmark_conditioned_beats_unconditioned(tmp_path: pathlib.Path) -> None:
    bm = IntentBenchmark(n_queries=10)
    bm.prepare(tmp_path)
    result = bm.run()
    assert result.benchmark_id == "CJ-INTENT"
    r5_cond = float(result.metrics["R@5_conditioned"])
    r5_uncond = float(result.metrics["R@5_unconditioned"])
    assert 0.0 <= r5_cond <= 1.0
    assert 0.0 <= r5_uncond <= 1.0
    # Conditioned embeddings are constructed to be closer to the target,
    # so they should hit the top-5 at least as often.
    assert r5_cond >= r5_uncond
    assert "delta_R5" in result.metrics


# ---- CJ-ROB-FMT / RENAME / DOC ---------------------------------------------


@pytest.mark.parametrize(
    "cls,bid",
    [
        (RobFmtBenchmark, "CJ-ROB-FMT"),
        (RobRenameBenchmark, "CJ-ROB-RENAME"),
        (RobDocBenchmark, "CJ-ROB-DOC"),
    ],
)
def test_rob_benchmark(cls: type[Benchmark], bid: str, tmp_path: pathlib.Path) -> None:
    bm = cls(n_chunks=10)  # type: ignore[call-arg]
    bm.prepare(tmp_path)
    result = bm.run()
    assert result.benchmark_id == bid
    rc = float(result.metrics["rank_change_pct"])
    cd = float(result.metrics["mean_cosine_drift"])
    assert 0.0 <= rc <= 100.0
    assert 0.0 <= cd <= 2.0  # cosine drift is in [0, 2]


# ---- CJ-OOD ----------------------------------------------------------------


def test_ood_benchmark(tmp_path: pathlib.Path) -> None:
    bm = OodBenchmark(n_queries=10)
    bm.prepare(tmp_path)
    result = bm.run()
    assert result.benchmark_id == "CJ-OOD"
    v = float(result.metrics["R@10"])
    assert 0.0 <= v <= 1.0
    assert result.metrics["pool_size"] == 200


# ---- CJ-PROBE-* ------------------------------------------------------------


def test_probe_name_benchmark(tmp_path: pathlib.Path) -> None:
    bm = ProbeNameBenchmark(n_examples=20)
    bm.prepare(tmp_path)
    result = bm.run()
    assert result.benchmark_id == "CJ-PROBE-NAME"
    # Either we got a real metric or we explicitly skipped.
    skipped = result.metrics.get("skipped", False)
    if not skipped:
        v = float(result.metrics["top1_accuracy"])
        assert 0.0 <= v <= 1.0


def test_probe_defect_benchmark(tmp_path: pathlib.Path) -> None:
    bm = ProbeDefectBenchmark(n_examples=20)
    bm.prepare(tmp_path)
    result = bm.run()
    assert result.benchmark_id == "CJ-PROBE-DEFECT"
    skipped = result.metrics.get("skipped", False)
    if not skipped:
        v = float(result.metrics["f1"])
        assert 0.0 <= v <= 1.0


def test_probe_clone_benchmark(tmp_path: pathlib.Path) -> None:
    bm = ProbeCloneBenchmark(n_examples=20)
    bm.prepare(tmp_path)
    result = bm.run()
    assert result.benchmark_id == "CJ-PROBE-CLONE"
    skipped = result.metrics.get("skipped", False)
    if not skipped:
        v = float(result.metrics["f1"])
        assert 0.0 <= v <= 1.0


# ---- CJ-EXEC / CJ-HUMAN stubs ----------------------------------------------


def test_exec_benchmark_stub_when_no_data(tmp_path: pathlib.Path) -> None:
    bm = ExecBenchmark()
    bm.prepare(tmp_path)
    result = bm.run()
    assert result.benchmark_id == "CJ-EXEC"
    assert float(result.metrics["pass_rate"]) == 0.0
    assert result.metrics["status"] == "no_executable_pairs"
    assert result.metrics["n_pairs"] == 0


def test_human_benchmark_stub_when_no_data(tmp_path: pathlib.Path) -> None:
    bm = HumanBenchmark()
    bm.prepare(tmp_path)
    result = bm.run()
    assert result.benchmark_id == "CJ-HUMAN"
    assert result.metrics["status"] == "no_human_annotations"
    assert result.metrics["mean_likert"] is None
    assert result.metrics["cohen_kappa"] is None


# ---- run_suite orchestrator ------------------------------------------------


def test_run_suite_writes_results_json(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    agg = run_suite(
        [RetBenchmark100(n_queries=10)],
        data_dir=data_dir,
        out_dir=out_dir,
        global_seed=0,
    )
    assert agg["schema_version"] == "v1"
    assert agg["global_seed"] == 0
    results_path = out_dir / "results.json"
    assert results_path.exists()
    payload: dict[str, Any] = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v1"
    assert len(payload["benchmarks"]) == 1
    assert payload["benchmarks"][0]["benchmark_id"] == "CJ-RET-100"
    # Per-benchmark file also exists.
    assert (out_dir / "CJ-RET-100.json").exists()


def test_run_suite_records_elapsed_time(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    agg = run_suite(
        [RetBenchmark100(n_queries=5)],
        data_dir=data_dir,
        out_dir=out_dir,
    )
    elapsed = agg["benchmarks"][0]["elapsed_seconds"]
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


# ---- RESULTS-MEMO.md generator --------------------------------------------


def test_generate_memo_writes_file(tmp_path: pathlib.Path) -> None:
    """run_suite → generate_memo round trip emits a parseable memo."""

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    run_suite(
        [RetBenchmark100(n_queries=10), HumanBenchmark()],
        data_dir=data_dir,
        out_dir=out_dir,
    )
    memo_path = tmp_path / "RESULTS-MEMO.md"
    text = generate_memo(out_dir / "results.json", memo_path)
    assert memo_path.exists()
    body = memo_path.read_text(encoding="utf-8")
    assert "# CodingJEPA — Results Memo" in body
    assert "TL;DR" in body or "TLDR" in body or "## Setup" in body
    assert text == body


def test_generate_memo_handles_missing_benchmarks(tmp_path: pathlib.Path) -> None:
    """When a benchmark is missing from results.json, emit a placeholder."""

    rj = tmp_path / "results.json"
    rj.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "global_seed": 0,
                "benchmarks": [],
            }
        ),
        encoding="utf-8",
    )
    memo_path = tmp_path / "RESULTS-MEMO.md"
    body = generate_memo(rj, memo_path)
    assert "No data available" in body


# ---- Sandbox runner --------------------------------------------------------


def test_sandbox_run_in_sandbox_pass() -> None:
    passed, stdout, rc = run_in_sandbox(
        "def f():\n    return 1\n",
        "from subject import f\n\ndef test_f():\n    assert f() == 1\n",
        timeout=30,
        backend="none",
    )
    assert passed is True
    assert rc == 0
    assert isinstance(stdout, str)


def test_sandbox_run_in_sandbox_fail() -> None:
    passed, _stdout, rc = run_in_sandbox(
        "def f():\n    return 2\n",
        "from subject import f\n\ndef test_f():\n    assert f() == 1\n",
        timeout=30,
        backend="none",
    )
    assert passed is False
    assert rc != 0


# ---- HTML output generators -----------------------------------------------


def test_diff_gallery_generates_index(tmp_path: pathlib.Path) -> None:
    pairs = [
        {
            "before": "def f(x):\n    return x\n",
            "after": "def f(x):\n    return x + 1\n",
            "intent": "INC",
            "chunk_id": f"chunk-{i:03d}",
        }
        for i in range(3)
    ]
    out_dir = generate_diff_gallery(pairs, tmp_path / "diffs")
    assert (out_dir / "index.html").exists()
    pages = sorted(out_dir.glob("pair-*.html"))
    assert len(pages) == 3
    index_html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "pair-000.html" in index_html
    assert "<table" in index_html


def test_confusions_generates_index(tmp_path: pathlib.Path) -> None:
    errors = {
        "INC": [
            {"query": "q1", "retrieved": "r1", "rank": "5"},
            {"query": "q2", "retrieved": "r2", "rank": "12"},
        ],
    }
    out_dir = generate_confusions(errors, tmp_path / "confusions")
    assert (out_dir / "index.html").exists()
    assert (out_dir / "INC.html").exists()
    intent_html = (out_dir / "INC.html").read_text(encoding="utf-8")
    assert "<table" in intent_html
    assert "q1" in intent_html


# ---- Figures generator -----------------------------------------------------


def test_generate_figures_runs_or_skips(tmp_path: pathlib.Path) -> None:
    """The figures generator either emits PDFs or returns [] when matplotlib is absent."""

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    run_suite(
        [RetBenchmark100(n_queries=10), RobFmtBenchmark(n_chunks=10)],
        data_dir=data_dir,
        out_dir=out_dir,
    )
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    paths = generate_figures(out_dir / "results.json", fig_dir)
    # Either we generated some PDFs, or matplotlib is unavailable.
    assert isinstance(paths, list)
    for p in paths:
        assert p.exists()
        assert p.suffix == ".pdf"
