"""Configuration for the RAG pipeline (env and .env)."""
import os
from pathlib import Path

from pydantic_settings import BaseSettings

# Project root (parent of backend/) — paths stay at root level
_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """App settings loaded from environment."""

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunk sizes
    child_chunk_tokens: int = 150
    parent_chunk_tokens: int = 700
    chunk_strategy: str = "semantic_tokens"
    parent_chunk_paragraphs: int = 4
    child_chunk_paragraphs: int = 1
    parent_chunk_sentences: int = 8
    child_chunk_sentences: int = 3
    parent_chunk_words: int = 400
    child_chunk_words: int = 120

    # Retrieval
    query_expansion_count: int = 3
    hybrid_top_k: int = 20
    rerank_top_k: int = 5

    # Paths (relative to project root)
    data_dir: Path = _BASE_DIR / "data"
    persist_dir: Path = _BASE_DIR / "qdrant_db"
    upload_dir: Path = _BASE_DIR / "uploads"

    class Config:
        env_file = str(_BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    """Load settings; fill OpenAI key from env if missing in .env."""
    s = Settings()
    if not s.openai_api_key:
        s.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    return s
