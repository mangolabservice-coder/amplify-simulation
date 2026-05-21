"""
Ingestion: documents → chunks → embeddings → Qdrant.

Mirrors the AMPLIFY paper's "documents are normalized, chunked, and embedded
into high-dimensional vectors". We use HuggingFace's BGE embeddings locally
so this works without any cloud key.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from llama_index.core import (
    Settings as LISettings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ..config import settings


@lru_cache(maxsize=1)
def _embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=settings.embed_model)


@lru_cache(maxsize=1)
def _qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def _configure_llama_index() -> None:
    """LlamaIndex reads embed_model and node_parser from its global Settings."""
    LISettings.embed_model = _embed_model()
    LISettings.node_parser = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    # The Model Router handles synthesis itself; LlamaIndex only does retrieval
    # here, so we install a no-op LLM to avoid any default OpenAI initialization.
    LISettings.llm = MockLLM()


def _vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=_qdrant_client(),
        collection_name=settings.qdrant_collection,
    )


@lru_cache(maxsize=1)
def get_index() -> VectorStoreIndex:
    """Open (or create) the index over the configured Qdrant collection."""
    _configure_llama_index()
    storage = StorageContext.from_defaults(vector_store=_vector_store())
    # from_vector_store loads the existing collection (or creates an empty handle)
    return VectorStoreIndex.from_vector_store(
        vector_store=_vector_store(),
        storage_context=storage,
    )


def ingest_paths(paths: List[str | Path]) -> dict:
    """Read files from `paths`, chunk + embed them, and insert into Qdrant."""
    _configure_llama_index()
    paths = [str(p) for p in paths]
    docs = SimpleDirectoryReader(input_files=paths).load_data()

    storage = StorageContext.from_defaults(vector_store=_vector_store())
    index = VectorStoreIndex.from_documents(docs, storage_context=storage)

    # Bust the cached index so subsequent retrievals see the new data.
    get_index.cache_clear()

    return {
        "files_ingested": len(paths),
        "documents_loaded": len(docs),
        "collection": settings.qdrant_collection,
        "embed_model": settings.embed_model,
    }
