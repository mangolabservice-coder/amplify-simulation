"""
OpenAI = the "Managed Foundational Model (OpenAI)" box in the AMPLIFY diagram.

In the paper this lives on the External Network and handles non-sensitive
queries when permitted by policy.
"""
from __future__ import annotations

from typing import Dict, List

from openai import AsyncOpenAI

from .base import LLMBackend, LLMResponse


class OpenAIBackend(LLMBackend):
    name = "openai"
    network = "external"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty — set it in .env to use this backend.")
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(
            text=text,
            model=self.model,
            backend=self.name,
            network=self.network,
            raw={"id": resp.id, "usage": getattr(resp, "usage", None).model_dump() if resp.usage else None},
        )
