"""Retrieval: query → embed → Qdrant top-k → MCP-friendly chunks."""
from __future__ import annotations

from typing import List

from ..config import settings
from ..mcp import RetrievedChunk
from .ingest import get_index


def retrieve(query: str, top_k: int | None = None) -> List[RetrievedChunk]:
    k = top_k or settings.top_k
    index = get_index()
    retriever = index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)

    chunks: List[RetrievedChunk] = []
    for n in nodes:
        meta = n.node.metadata or {}
        chunks.append(
            RetrievedChunk(
                text=n.node.get_content(),
                score=float(n.score) if n.score is not None else 0.0,
                source=meta.get("file_name") or meta.get("file_path") or "unknown",
                page=meta.get("page_label") or meta.get("page_number"),
                chunk_id=n.node.node_id,
                extra={k: v for k, v in meta.items() if k not in {"file_name", "file_path", "page_label", "page_number"}},
            )
        )
    return chunks
