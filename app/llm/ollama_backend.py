"""
Ollama = the "Self-Hosted Foundational Model (Llama)" box in the AMPLIFY diagram.

This stays inside the trusted network — sensitive queries are pinned here
by the Model Router so proprietary content never leaves the perimeter.
"""
from __future__ import annotations

from typing import Dict, List

import httpx

from .base import LLMBackend, LLMResponse


class OllamaBackend(LLMBackend):
    name = "ollama"
    network = "internal"

    def __init__(self, base_url: str, model: str = "llama3.2") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        text = (data.get("message") or {}).get("content", "")
        return LLMResponse(
            text=text,
            model=self.model,
            backend=self.name,
            network=self.network,
            raw={"eval_count": data.get("eval_count"), "total_duration": data.get("total_duration")},
        )
