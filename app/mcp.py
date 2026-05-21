"""
Model Context Protocol (MCP) layer.

Per the AMPLIFY paper, MCP "ensures that retrieved context, metadata, and
supporting evidence are structured and passed to the model in a consistent
and auditable manner". This module owns the schema for that hand-off so
every backend (OpenAI, Ollama, future agents) sees the same shape.

If you swap retrievers or LLMs you only need to keep producing/consuming
this contract — that is the modularity / interoperability claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedChunk:
    """A single retrieved passage. Carries enough metadata to cite it."""
    text: str
    score: float
    source: str                       # e.g. "AMPLIFY_whitepaper.pdf"
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPContext:
    """The structured payload handed to the LLM."""
    query: str
    chunks: List[RetrievedChunk]
    system_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- formatting helpers -------------------------------------------------

    def render_context_block(self) -> str:
        """Format chunks as numbered citations the model is told to reference."""
        lines: List[str] = []
        for i, c in enumerate(self.chunks, start=1):
            loc = f"{c.source}" + (f", p.{c.page}" if c.page is not None else "")
            lines.append(f"[{i}] ({loc}, score={c.score:.3f})\n{c.text.strip()}")
        return "\n\n".join(lines) if lines else "(no retrieved context)"

    def render_user_prompt(self) -> str:
        return (
            "Use the numbered context below to answer the question. "
            "Cite sources inline as [1], [2], etc. If the context is "
            "insufficient, say so explicitly.\n\n"
            f"# Context\n{self.render_context_block()}\n\n"
            f"# Question\n{self.query}"
        )

    def to_messages(self) -> List[Dict[str, str]]:
        """OpenAI/Ollama compatible chat-message list."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.render_user_prompt()},
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "system_prompt": self.system_prompt,
            "chunks": [asdict(c) for c in self.chunks],
            "metadata": self.metadata,
        }


DEFAULT_SYSTEM_PROMPT = (
    "You are AMPLIFY, an assistant for mission-critical retrieval and reasoning. "
    "Answer using ONLY the provided context. Be precise, cite sources with [n], "
    "and if the context does not contain the answer, say so plainly."
)


def build_context(
    query: str,
    chunks: List[RetrievedChunk],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    metadata: Optional[Dict[str, Any]] = None,
) -> MCPContext:
    return MCPContext(
        query=query,
        chunks=chunks,
        system_prompt=system_prompt,
        metadata=metadata or {},
    )
