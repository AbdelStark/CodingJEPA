"""FastAPI app (RFC-0006 §D3, §D8)."""

from __future__ import annotations

from fastapi import FastAPI

from codingjepa.demo.web import routes


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="CodingJEPA Demo", version="0.1.0")
    app.include_router(routes.router)
    return app


app = create_app()

__all__ = ["app", "create_app"]
