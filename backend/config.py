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
    llm_fallback_model: str = "gpt-4o-mini"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    llm_total_budget_tokens: int = 12000
    llm_context_budget_ratio: float = 0.78
    llm_history_budget_ratio: float = 0.12
    llm_max_output_tokens_default: int = 900
    llm_max_output_tokens_legal: int = 1400
    llm_context_chunk_soft_cap_tokens: int = 320

    # Chunk sizes
    article_max_parent_tokens: int = 1200  # oversized articles are split further

    # Embedding parallelism
    embed_batch_size: int = 64
    embed_max_workers: int = 4
    pgvector_upsert_batch_size: int = 128

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

    # Whether to apply context-aware rescoring after retrieval
    enable_context_rescore: bool = True

    # BM25 sparse scoring parameters (Okapi BM25)
    bm25_k1: float = 1.5   # term frequency saturation
    bm25_b: float = 0.75   # document length normalization

    # Fixed collection names for the two domain-specific RAG agents.
    # Override in .env if your collection names differ.
    legal_collection_name: str = "legal"
    drug_collection_name: str = "drug"

    # PostgreSQL database
    database_url: str = "postgresql://admin:password@localhost:5433/pharmanet"

    # Auth
    jwt_secret: str = "CHANGE_ME"
    jwt_algorithm: str = "HS256"
    admin_emails: list[str] = []

    # Evaluation (RAGAS)
    eval_judge_model: str = "gpt-4o-mini"
    eval_random_seed: int = 42
    eval_max_workers: int = 2
    eval_timeout_seconds: int = 90
    eval_max_samples: int = 0

    # Paths (relative to project root)
    data_dir: Path = _BASE_DIR / "data"
    upload_dir: Path = _BASE_DIR / "uploads"
    eval_output_dir: Path = _BASE_DIR / "eval_results"

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
