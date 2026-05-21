"""Centralized settings loaded from .env. Mirrors AMPLIFY's modular config story."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dotenv load order (pydantic merges with .update — later files win):
#   repo .env  →  fritz-hub/.env  →  amplify-simulation/.env
# So OPENAI_API_KEY matches fritz-hub / n8n OpenAI when stored in fritz-hub/.env or a shared repo root .env.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_ROOT.parent
_ENV_FILES: Tuple[str, ...] = tuple(
    str(p)
    for p in (
        _REPO_ROOT / ".env",
        _REPO_ROOT / "fritz-hub" / ".env",
        _PACKAGE_ROOT / ".env",
    )
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILES, env_file_encoding="utf-8", extra="ignore")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "amplify_docs"

    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # OpenAI (managed external)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Ollama (self-hosted)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Routing
    enforce_sensitive_routing: bool = True
    sensitive_keywords: str = "classified,proprietary,sensitive,internal,sop,faa,nasa"

    # Retrieval
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    @property
    def sensitive_keyword_list(self) -> List[str]:
        return [k.strip().lower() for k in self.sensitive_keywords.split(",") if k.strip()]


settings = Settings()
