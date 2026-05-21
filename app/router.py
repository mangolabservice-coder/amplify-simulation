"""
Model Router.

The paper's Model Router "dynamically directs queries between secure internal
models and external services based on compliance requirements, latency
constraints, and expected performance." This is a simple rule-based version
of that policy:

  1. Explicit override:  client passes route="internal" or "external"
  2. Sensitivity check:  any sensitive keyword present → internal (Ollama)
  3. Fallback:           external (OpenAI), if available; otherwise internal
  4. Availability gate:  if the chosen backend is unconfigured, fall through

The decision (and *why*) is logged into the trace span so it's auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .config import settings
from .llm import LLMBackend, OllamaBackend, OpenAIBackend


@dataclass
class RoutingDecision:
    backend: LLMBackend
    reason: str       # human-readable justification
    sensitive: bool


class ModelRouter:
    def __init__(self) -> None:
        self._backends: Dict[str, LLMBackend] = {}
        # Internal (always try to register — no API key needed)
        try:
            self._backends["internal"] = OllamaBackend(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )
        except Exception:  # noqa: BLE001
            pass
        # External (only if API key is set)
        if settings.openai_api_key:
            try:
                self._backends["external"] = OpenAIBackend(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                )
            except Exception:  # noqa: BLE001
                pass

    # ---- policy ------------------------------------------------------------

    def is_sensitive(self, query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in settings.sensitive_keyword_list)

    def decide(self, query: str, override: Optional[str] = None) -> RoutingDecision:
        sensitive = self.is_sensitive(query)

        # 1. Explicit override (still subject to availability)
        if override in {"internal", "external"} and override in self._backends:
            return RoutingDecision(
                backend=self._backends[override],
                reason=f"explicit override → {override}",
                sensitive=sensitive,
            )

        # 2. Sensitivity → internal
        if settings.enforce_sensitive_routing and sensitive and "internal" in self._backends:
            return RoutingDecision(
                backend=self._backends["internal"],
                reason="sensitive keyword detected → internal/self-hosted",
                sensitive=True,
            )

        # 3. Default → external if available
        if "external" in self._backends:
            return RoutingDecision(
                backend=self._backends["external"],
                reason="default policy → external/managed",
                sensitive=sensitive,
            )

        # 4. Fallback → internal
        if "internal" in self._backends:
            return RoutingDecision(
                backend=self._backends["internal"],
                reason="external unavailable → fallback to internal",
                sensitive=sensitive,
            )

        raise RuntimeError(
            "No LLM backends configured. Set OPENAI_API_KEY or run Ollama locally."
        )

    @property
    def available(self) -> Dict[str, str]:
        return {k: v.model for k, v in self._backends.items()}


router = ModelRouter()
