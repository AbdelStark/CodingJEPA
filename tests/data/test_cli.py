"""Tests for codingjepa.data.cli — the ``codingjepa data <step>`` subcommands."""

from __future__ import annotations

from typing import Any

import pytest

from codingjepa import cli as cli_mod
from codingjepa.data import cli as data_cli


def test_data_subcommand_registered() -> None:
    """The top-level CLI exposes a ``data`` subcommand."""

    rc = cli_mod.main(["data", "--help"])
    # argparse prints help to stdout and exits with rc 0 by SystemExit; we
    # caught that via main returning. rc==0 is sufficient.
    assert rc == 0


def test_data_cli_help_shows_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    """The ``data --help`` text lists every Phase 1 subcommand."""

    rc = cli_mod.main(["data", "--help"])
    captured = capsys.readouterr()
    assert rc == 0
    for sub in ("mirror", "chunk", "pairs", "dedup", "splits", "audit", "manifest", "all"):
        assert sub in captured.out


def test_data_cli_no_subcommand_returns_usage(capsys: pytest.CaptureFixture[str]) -> None:
    """`codingjepa data` with no sub returns non-zero and prints help."""

    rc = cli_mod.main(["data"])
    captured = capsys.readouterr()
    assert rc != 0
    # Help or usage text should mention the available subcommands.
    text = captured.out + captured.err
    assert "mirror" in text or "usage" in text.lower()


def test_data_cli_mirror_invokes_mirror(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ``mirror`` subcommand calls ``codingjepa.data.mirror.mirror``."""

    called: dict[str, Any] = {}

    def fake_mirror(*args: Any, **kwargs: Any) -> list[Any]:
        called["args"] = args
        called["kwargs"] = kwargs
        return []

    monkeypatch.setattr("codingjepa.data.mirror.mirror", fake_mirror)
    rc = cli_mod.main(["data", "mirror"])
    assert rc == 0
    assert "args" in called or "kwargs" in called


def test_data_cli_manifest_invokes_write_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``manifest`` subcommand reads from mirror's registry and calls write_manifest.

    We patch out the actual disk write to keep the test fast and pure.
    """

    called: dict[str, Any] = {}

    def fake_write_manifest(repos: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        called["repos"] = repos
        called["kwargs"] = kwargs
        return {"manifest_hash": "0" * 64}

    monkeypatch.setattr("codingjepa.data.manifest.write_manifest", fake_write_manifest)
    rc = cli_mod.main(["data", "manifest"])
    assert rc == 0
    assert "repos" in called


def test_data_cli_dispatches_each_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ``data <step>`` dispatches to the corresponding cmd_<step> function."""

    seen: list[str] = []

    for sub in ("mirror", "chunk", "pairs", "dedup", "splits", "audit", "manifest"):

        def make_stub(name: str) -> Any:
            def stub(args: Any) -> int:
                seen.append(name)
                return 0

            return stub

        monkeypatch.setattr(data_cli, f"cmd_{sub}", make_stub(sub))

    # Re-build the parser so it sees the patched cmd_* attributes.
    for sub in ("mirror", "chunk", "pairs", "dedup", "splits", "audit", "manifest"):
        rc = cli_mod.main(["data", sub])
        assert rc == 0, f"{sub} returned {rc}"

    assert set(seen) >= {"mirror", "chunk", "pairs", "dedup", "splits", "audit", "manifest"}


def test_data_cli_all_runs_full_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``data all`` chains every subcommand in pipeline order."""

    order: list[str] = []

    for sub in ("mirror", "chunk", "pairs", "dedup", "splits", "audit", "manifest"):

        def make_stub(name: str) -> Any:
            def stub(args: Any) -> int:
                order.append(name)
                return 0

            return stub

        monkeypatch.setattr(data_cli, f"cmd_{sub}", make_stub(sub))

    rc = cli_mod.main(["data", "all"])
    assert rc == 0
    assert order == ["mirror", "chunk", "pairs", "dedup", "splits", "audit", "manifest"]
