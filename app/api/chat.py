"""
/chat — the end-to-end pipeline.

Flow (mirrors Figure 1 of the AMPLIFY paper, right to left):
  Chat Interface
    -> Unified LLM Interface (this endpoint)
    -> Model Router (decides backend)
    -> Model Context Protocol (packs query + retrieved chunks)
    -> Self-hosted Llama OR Managed OpenAI
  with retrieval against the embeddings stored in Qdrant.

Every stage is wrapped in a Langfuse-style span via `tracer`.
"""
from __future__ import annotations

import time
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..mcp import build_context
from ..observability import tracer
from ..rag import retrieve
from ..router import router as model_router

api = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = None
    route: Optional[Literal["internal", "external"]] = None  # explicit override


class Citation(BaseModel):
    n: int
    source: str
    page: Optional[int | str] = None
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    routing: dict
    trace_id: str
    latency_ms: float


@api.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    with tracer.trace("chat", query=req.query, route_override=req.route) as t:
        # 1) Routing decision
        with tracer.span(t, "router.decide", query=req.query) as sp:
            try:
                decision = model_router.decide(req.query, override=req.route)
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
            sp.attributes.update(
                backend=decision.backend.name,
                network=decision.backend.network,
                model=decision.backend.model,
                sensitive=decision.sensitive,
                reason=decision.reason,
            )

        # 2) Retrieval
        with tracer.span(t, "rag.retrieve", top_k=req.top_k) as sp:
            chunks = retrieve(req.query, top_k=req.top_k)
            sp.attributes["num_chunks"] = len(chunks)
            sp.attributes["scores"] = [round(c.score, 4) for c in chunks]
            sp.attributes["sources"] = [c.source for c in chunks]

        # 3) MCP packaging
        with tracer.span(t, "mcp.pack") as sp:
            ctx = build_context(
                req.query,
                chunks,
                metadata={
                    "routing_reason": decision.reason,
                    "backend": decision.backend.name,
                },
            )
            sp.attributes["prompt_chars"] = sum(len(m["content"]) for m in ctx.to_messages())

        # 4) LLM synthesis
        with tracer.span(t, "llm.chat", backend=decision.backend.name) as sp:
            try:
                resp = await decision.backend.chat(ctx.to_messages())
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(502, f"LLM backend failed: {exc!r}") from exc
            sp.attributes["model"] = resp.model
            sp.attributes["network"] = resp.network
            sp.attributes["answer_chars"] = len(resp.text)

        citations = [
            Citation(n=i + 1, source=c.source, page=c.page, score=round(c.score, 4))
            for i, c in enumerate(chunks)
        ]
        t.output = {
            "answer": resp.text,
            "citations": [c.model_dump() for c in citations],
            "backend": resp.backend,
            "network": resp.network,
            "model": resp.model,
        }

        latency = time.time() - t.started_at
        return ChatResponse(
            answer=resp.text,
            citations=citations,
            routing={
                "backend": resp.backend,
                "network": resp.network,
                "model": resp.model,
                "reason": decision.reason,
                "sensitive": decision.sensitive,
            },
            trace_id=t.trace_id,
            latency_ms=round(latency * 1000, 2),
        )


@api.get("/router/status")
def router_status() -> dict:
    return {"available_backends": model_router.available}
