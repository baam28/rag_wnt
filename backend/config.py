"""Configuration for the RAG pipeline (env and .env)."""
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# Project root (parent of backend/) — paths stay at root level
_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """App settings loaded from environment."""

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Chunk sizes
    child_chunk_tokens: int = 150
    parent_chunk_tokens: int = 700
    chunk_strategy: str = "auto"
    parent_chunk_paragraphs: int = 4
    child_chunk_paragraphs: int = 1
    parent_chunk_sentences: int = 8
    child_chunk_sentences: int = 3
    parent_chunk_words: int = 400
    child_chunk_words: int = 120

    # Embedding parallelism
    embed_batch_size: int = 64
    embed_max_workers: int = 4

    # Retrieval
    query_expansion_count: int = 3
    hybrid_top_k: int = 20
    rerank_top_k: int = 5

    # Rate limiting (slowapi format: "N/period" where period = second/minute/hour)
    ask_rate_limit: str = "20/minute"

    # Multi-collection retrieval: how many collections (sorted by top rerank score)
    # to merge results from.  1 = best-collection-only (original behaviour).
    # Set to 0 to merge ALL collections.
    retrieve_top_k_collections: int = 1

    # Fixed collection names for the two domain-specific RAG agents.
    # Override in .env if your Qdrant collection names differ.
    legal_collection_name: str = "legal"
    drug_info_collection_name: str = "drug"

    # MongoDB & auth
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "rag_chatbot"
    jwt_secret: str = "CHANGE_ME"
    jwt_algorithm: str = "HS256"
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = ""
    admin_emails: list[str] = []

    # Paths (relative to project root)
    data_dir: Path = _BASE_DIR / "data"
    persist_dir: Path = _BASE_DIR / "qdrant_db"
    upload_dir: Path = _BASE_DIR / "uploads"

    class Config:
        env_file = str(_BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process (cached).

    The .env file and environment variables are read a single time and the
    result is reused for every subsequent call.  Use ``get_settings.cache_clear()``
    in tests when you need to reload the configuration.
    """
    s = Settings()
    # Fall back to the bare env var in case .env omits OPENAI_API_KEY
    if not s.openai_api_key:
        s.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    return s
