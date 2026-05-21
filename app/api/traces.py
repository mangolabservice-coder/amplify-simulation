"""/traces — Langfuse-style observability surface for the chat UI."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..observability import tracer

api = APIRouter()


@api.get("/traces")
def list_traces(limit: int = 50) -> dict:
    return {"traces": tracer.list(limit=limit)}


@api.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    t = tracer.get(trace_id)
    if t is None:
        raise HTTPException(404, "trace not found")
    return t
