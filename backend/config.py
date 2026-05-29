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
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    llm_provider: str = "openai"  # "openai" | "anthropic"
    embedding_model: str = "voyage-law-2"
    llm_model: str = "gpt-4o-mini"
    llm_fallback_model: str = "gpt-4o-mini"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    llm_total_budget_tokens: int = 12000
    llm_context_budget_ratio: float = 0.78
    llm_history_budget_ratio: float = 0.12
    llm_max_output_tokens_default: int = 900
    llm_max_output_tokens_legal: int = 1400
    llm_context_chunk_soft_cap_tokens: int = 700

    # Chunk sizes
    article_max_parent_tokens: int = 3000  # oversized articles are split further

    # Embedding parallelism
    embed_batch_size: int = 64
    embed_max_workers: int = 4
    pgvector_upsert_batch_size: int = 128

    # Retrieval
    query_expansion_count: int = 3
    hybrid_top_k: int = 20
    rerank_top_k: int = 5
    rerank_top_k_legal: int = 8  # higher top-k for legal queries (multi-khoản articles)
    dense_score_threshold: float = 0.30  # discard results below this cosine similarity
    dense_multi_query_k: int = 2  # number of query variations to embed for dense search (including primary)
    rerank_input_k: int = 20  # max candidates fed to CrossEncoder across all collections (pre-filter by RRF score)
    rerank_score_threshold: float = 0.0  # CrossEncoder logit below this → excluded before LLM prompt

    # Rate limiting (slowapi format: "N/period" where period = second/minute/hour)
    ask_rate_limit: str = "20/minute"

    # Multi-collection retrieval: how many collections (sorted by top rerank score)
    # to merge results from.  1 = best-collection-only (original behaviour).
    # Set to 0 to merge ALL collections.
    retrieve_top_k_collections: int = 1

    # Whether to apply context-aware rescoring after retrieval
    enable_context_rescore: bool = True

    # Cross-reference resolution: auto-detect article references in LLM answers
    # and fetch + re-generate with the referenced content (legal queries only).
    enable_crossref_resolution: bool = True
    crossref_max_refs: int = 3  # max refs to resolve per answer

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
    if not s.anthropic_api_key:
        s.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not s.voyage_api_key:
        s.voyage_api_key = os.getenv("VOYAGE_API_KEY", "")
    return s


def get_runtime_llm_settings() -> dict:
    """Return the active LLM provider/model and DB-backed API keys."""
    from pg_client import get_app_setting  # local import to avoid circular deps

    static = get_settings()
    provider = get_app_setting("llm_provider") or static.llm_provider or "openai"
    model = get_app_setting("llm_model") or ""
    openai_key = get_app_setting("openai_api_key") or static.openai_api_key
    anthropic_key = get_app_setting("anthropic_api_key") or static.anthropic_api_key

    # Default model per provider when nothing is set in DB
    if not model:
        if provider == "anthropic":
            model = "claude-sonnet-4-6"
        else:
            model = static.llm_model or "gpt-4o-mini"

    return {
        "provider": provider,
        "model": model,
        "openai_api_key": openai_key,
        "anthropic_api_key": anthropic_key,
    }


def get_runtime_embedding_settings() -> dict:
    """Return Voyage AI embedding settings from environment."""
    static = get_settings()
    return {
        "model": "voyage-law-2",
        "voyage_api_key": static.voyage_api_key,
    }
