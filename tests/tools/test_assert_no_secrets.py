"""Tests for ``tools/assert_no_secrets.py`` — the corpus secrets gate (#177).

The script reads parquet files, extracts the ``source`` column, runs
:func:`codingjepa.data.secrets_scan.scan_chunks`, and exits non-zero if any
hits are found. We skip the entire module if ``pandas``/``pyarrow`` are not
available so the test runs in minimal environments.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")  # noqa: F841 — used implicitly via DataFrame below
pytest.importorskip("pyarrow")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "assert_no_secrets.py"


def _write_parquet(path: Path, sources: list[str]) -> None:
    import pandas as pd

    df = pd.DataFrame({"source": sources})
    df.to_parquet(path, index=False)


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *[str(a) for a in args]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_assert_no_secrets_clean(tmp_path: Path) -> None:
    """A parquet with clean source returns exit code 0 and a success line."""

    parquet = tmp_path / "clean.parquet"
    _write_parquet(parquet, ["def foo(x):\n    return x + 1\n"])

    result = _run(parquet)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_assert_no_secrets_hit(tmp_path: Path) -> None:
    """A parquet that contains a fake AWS key triggers exit 1."""

    parquet = tmp_path / "dirty.parquet"
    _write_parquet(
        parquet,
        [
            "def foo(x):\n    return x + 1\n",
            'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n',
        ],
    )

    result = _run(parquet)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stderr or "FAIL" in result.stdout


def test_assert_no_secrets_skips_missing_source_column(tmp_path: Path) -> None:
    """Parquets without a ``source`` column are silently skipped, not failed."""

    import pandas as pd

    parquet = tmp_path / "nocol.parquet"
    pd.DataFrame({"other": ["x"]}).to_parquet(parquet, index=False)

    result = _run(parquet)
    assert result.returncode == 0, result.stdout + result.stderr


def test_assert_no_secrets_handles_multiple_files(tmp_path: Path) -> None:
    """The CLI accepts multiple parquet positional arguments."""

    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    _write_parquet(p1, ["def f(): return 1\n"])
    _write_parquet(p2, ["def g(): return 2\n"])

    result = _run(p1, p2)
    assert result.returncode == 0, result.stdout + result.stderr
