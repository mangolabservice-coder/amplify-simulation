"""FastAPI entrypoint — the 'Unified LLM Interface' from the AMPLIFY diagram."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .api.chat import api as chat_api
from .api.ingest import api as ingest_api
from .api.traces import api as traces_api

app = FastAPI(
    title="AMPLIFY-clone",
    description=(
        "A small, faithful reproduction of NASA's AMPLIFY architecture: "
        "FastAPI + LlamaIndex + Qdrant + Model Router + MCP + tracing."
    ),
    version="0.1.0",
)

app.include_router(chat_api, tags=["chat"])
app.include_router(ingest_api, tags=["ingest"])
app.include_router(traces_api, tags=["observability"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


_UI = Path(__file__).parent / "ui" / "index.html"


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    if _UI.exists():
        return HTMLResponse(_UI.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AMPLIFY-clone</h1><p>UI missing.</p>")
