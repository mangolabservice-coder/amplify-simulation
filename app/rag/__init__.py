"""RAG pipeline: ingestion + retrieval, built on LlamaIndex + Qdrant."""
from .ingest import ingest_paths, get_index
from .retrieve import retrieve

__all__ = ["ingest_paths", "get_index", "retrieve"]
