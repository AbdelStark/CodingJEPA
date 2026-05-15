"""Tests for all demo modules (RFC-0006 §D2–D6)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from codingjepa.demo.cli import build_refactor_parser, cmd_refactor
from codingjepa.demo.diff import render_diff_html, render_diff_terminal
from codingjepa.demo.messages import get_demo_message
from codingjepa.demo.web.app import create_app
from codingjepa.demo.web.templates import render_candidates, render_form

# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


def test_get_demo_message_known_key() -> None:
    """get_demo_message returns correct string for a known key."""
    msg = get_demo_message("source_too_long")
    assert "512 BPE tokens" in msg


def test_get_demo_message_unknown_key() -> None:
    """get_demo_message raises KeyError for an unknown key."""
    with pytest.raises(KeyError):
        get_demo_message("nonexistent_key_xyz")


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

_BEFORE = "def foo():\n    return 1\n"
_AFTER = "def foo():\n    return 2\n"


def test_render_diff_terminal_nonempty() -> None:
    """render_diff_terminal returns a nonempty string for differing inputs."""
    result = render_diff_terminal(_BEFORE, _AFTER)
    assert isinstance(result, str)
    assert len(result) > 0


def test_render_diff_terminal_identical() -> None:
    """render_diff_terminal returns empty string when inputs are identical."""
    result = render_diff_terminal(_BEFORE, _BEFORE)
    assert result == ""


def test_render_diff_html_is_html() -> None:
    """render_diff_html returns a string containing <!DOCTYPE html>."""
    result = render_diff_html(_BEFORE, _AFTER)
    assert "<!DOCTYPE html>" in result


def test_render_diff_html_has_metadata() -> None:
    """render_diff_html includes cosine and confidence metadata in output."""
    result = render_diff_html(
        _BEFORE, _AFTER, cosine=0.84, confidence=0.71, provenance="cpython@abc1234"
    )
    assert "0.84" in result
    assert "0.71" in result
    assert "cpython@abc1234" in result


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def test_cli_parser_source_arg() -> None:
    """Parser accepts --source and --intent flags."""
    parser = build_refactor_parser()
    args = parser.parse_args(["--source", "def foo(): pass", "--intent", "extract-helper"])
    assert args.source == "def foo(): pass"
    assert args.intent == "extract-helper"


def test_cli_parser_file_arg(tmp_path: pytest.TempPathFactory) -> None:
    """Parser accepts --file and --intent flags."""
    py_file = tmp_path / "sample.py"  # type: ignore[operator]
    py_file.write_text("def bar(): pass\n")
    parser = build_refactor_parser()
    args = parser.parse_args(["--file", str(py_file), "--intent", "NONE"])
    assert str(args.file) == str(py_file)
    assert args.intent == "NONE"


def test_cmd_refactor_no_args_exits_nonzero() -> None:
    """cmd_refactor with neither --source nor --file returns error code 1."""
    parser = build_refactor_parser()
    args = parser.parse_args([])
    # source and file are both None when no args are passed
    result = cmd_refactor(args)
    assert result == 1


# ---------------------------------------------------------------------------
# FastAPI web app
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz_returns_ok(client: TestClient) -> None:
    """GET /healthz returns JSON with status=ok."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "checkpoint_hash" in data
    assert "index_id" in data


def test_index_returns_html(client: TestClient) -> None:
    """GET / returns HTML with the refactor form."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<form" in response.text


def test_refactor_endpoint(client: TestClient) -> None:
    """POST /refactor returns an HTML fragment containing div#results."""
    response = client.post(
        "/refactor",
        data={"source": "def foo(): pass", "intent": "NONE", "k": "5"},
    )
    assert response.status_code == 200
    assert "results" in response.text


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


def test_render_form_has_textarea() -> None:
    """render_form() contains a <textarea element."""
    html = render_form()
    assert "<textarea" in html


def test_render_candidates_nonempty() -> None:
    """render_candidates returns nonempty string for a non-empty list."""
    candidates: list[dict[str, object]] = [
        {
            "cosine": 0.82,
            "confidence": 0.70,
            "provenance": "cpython@deadbeef",
            "after": "def foo():\n    return 42\n",
            "accepted": True,
            "reason": "",
        }
    ]
    result = render_candidates(candidates)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "results" in result
