"""FastAPI route definitions (RFC-0006 §D3)."""

from __future__ import annotations

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, JSONResponse

from codingjepa.demo.web.templates import render_form

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the refactor form."""
    return HTMLResponse(content=render_form())


@router.post("/refactor", response_class=HTMLResponse)
async def refactor(
    source: str = Form(...),
    intent: str = Form(default="NONE"),
    k: int = Form(default=10),
) -> HTMLResponse:
    """Run refactor, return HTMX fragment.

    At this stage (no trained checkpoint) returns a placeholder fragment.
    """
    # Stub: return an HTMX fragment noting no checkpoint is loaded
    html = f"<div id='results'>" f"<p>No checkpoint loaded. intent={intent!r}, k={k}</p>" f"</div>"
    return HTMLResponse(content=html)


@router.get("/healthz")
async def healthz() -> JSONResponse:
    """Return checkpoint and index hashes (RFC-0009 §D8, spec/02)."""
    return JSONResponse({"status": "ok", "checkpoint_hash": None, "index_id": None})


@router.get("/version")
async def version() -> JSONResponse:
    """Return package version."""
    from importlib.metadata import PackageNotFoundError  # noqa: PLC0415
    from importlib.metadata import version as pkg_version  # noqa: PLC0415

    try:
        v = pkg_version("codingjepa")
    except PackageNotFoundError:
        v = "0.0.0"
    return JSONResponse({"version": v})


__all__ = ["router"]
