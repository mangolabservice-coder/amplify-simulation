"""Common backend interface so the router can treat OpenAI and Ollama uniformly."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str          # "openai" | "ollama"
    network: str          # "external" | "internal"
    raw: dict | None = None


class LLMBackend(ABC):
    """A minimal chat-style interface."""

    name: str          # backend id, e.g. "openai"
    network: str       # "external" or "internal"
    model: str

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Run a chat completion. `messages` is in OpenAI's format."""
