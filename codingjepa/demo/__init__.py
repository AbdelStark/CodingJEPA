"""CodingJEPA demo surface (RFC-0006)."""

from codingjepa.demo.cli import build_refactor_parser, cmd_refactor
from codingjepa.demo.diff import render_diff_html, render_diff_terminal
from codingjepa.demo.messages import DEMO_MESSAGES, get_demo_message

__all__ = [
    "DEMO_MESSAGES",
    "build_refactor_parser",
    "cmd_refactor",
    "get_demo_message",
    "render_diff_html",
    "render_diff_terminal",
]
