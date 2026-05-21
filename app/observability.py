"""
Observability layer (Langfuse-style).

The AMPLIFY paper uses Langfuse to capture end-to-end query flows: input,
retrieval results, system response, latency, and errors. This is a small,
in-process equivalent that produces the same shape of data and exposes it
via a REST endpoint so the chat UI (and external tooling) can inspect it.

A Trace contains a list of Spans. Each Span records a single step of the
pipeline (retrieve, route, mcp_pack, llm_call, synthesize).
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, Iterator, List, Optional


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        return None if self.end_ms is None else round(self.end_ms - self.start_ms, 2)


@dataclass
class Trace:
    trace_id: str
    name: str
    started_at: float
    ended_at: Optional[float] = None
    input: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["latency_ms"] = (
            None if self.ended_at is None else round((self.ended_at - self.started_at) * 1000, 2)
        )
        return d


class Tracer:
    """Tiny in-memory tracer. In production you'd ship spans to Langfuse."""

    def __init__(self, max_traces: int = 200) -> None:
        self._traces: Deque[Trace] = deque(maxlen=max_traces)

    @contextmanager
    def trace(self, name: str, **input_kwargs: Any) -> Iterator[Trace]:
        t = Trace(
            trace_id=str(uuid.uuid4()),
            name=name,
            started_at=time.time(),
            input=input_kwargs,
        )
        self._traces.append(t)
        try:
            yield t
        finally:
            t.ended_at = time.time()

    @contextmanager
    def span(self, trace: Trace, name: str, **attrs: Any) -> Iterator[Span]:
        s = Span(name=name, start_ms=time.time() * 1000, attributes=dict(attrs))
        trace.spans.append(s)
        try:
            yield s
        except Exception as exc:  # noqa: BLE001
            s.error = repr(exc)
            raise
        finally:
            s.end_ms = time.time() * 1000

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in list(self._traces)[-limit:][::-1]]

    def get(self, trace_id: str) -> Optional[Dict[str, Any]]:
        for t in self._traces:
            if t.trace_id == trace_id:
                return t.to_dict()
        return None


# Module-level singleton — analogous to a Langfuse client.
tracer = Tracer()
