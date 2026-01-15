from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # OpenAI
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Paths
    docs_path: str = os.getenv("DOCS_PATH", "./docs")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    bm25_path: str = os.getenv("BM25_PATH", "./bm25_index.pkl")

    # Retrieval
    top_k_vector: int = int(os.getenv("TOP_K_VECTOR", "20"))
    top_k_keyword: int = int(os.getenv("TOP_K_KEYWORD", "20"))
    top_k_fused: int = int(os.getenv("TOP_K_FUSED", "8"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))  # characters
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


settings = Settings()
