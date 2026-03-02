"""Configuration for the RAG pipeline (env and .env)."""
import os
from pathlib import Path

from pydantic_settings import BaseSettings


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

    # Paths
    data_dir: Path = Path("data")
    persist_dir: Path = Path("qdrant_db")
    upload_dir: Path = Path("uploads")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    """Load settings; fill OpenAI key from env if missing in .env."""
    s = Settings()
    if not s.openai_api_key:
        s.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    return s
