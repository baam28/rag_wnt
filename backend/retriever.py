"""Retrieval: query expansion, hybrid search (vector + Qdrant native sparse), re-ranking, parent context."""
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from cachetools import TTLCache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import get_settings
from utils import fix_position_ids as _fix_position_ids, get_qdrant_client, tokenize_for_sparse as _tokenize_for_sparse

logger = logging.getLogger(__name__)


# --- Query expansion ---

def expand_query(query: str, llm: ChatOpenAI, num_variations: int = 3) -> list[str]:
    """Generate multiple query variations to capture different semantic angles."""
    prompt = """Tạo đúng {n} biến thể câu hỏi (bằng tiếng Việt hoặc tiếng Anh tùy ngữ cảnh) 
dựa trên câu hỏi gốc dưới đây. Mỗi biến thể diễn đạt khác một chút hoặc nhấn mạnh góc độ khác 
để tìm thông tin liên quan. Chỉ trả lời bằng các câu hỏi, mỗi câu một dòng, không đánh số.

Câu hỏi gốc: {query}"""
    try:
        resp = llm.invoke(prompt.format(n=num_variations, query=query))
        content = resp.content if hasattr(resp, "content") else str(resp)
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        cleaned = []
        for line in lines[: num_variations + 2]:
            line = re.sub(r"^[\d\.\)\-\*]+\s*", "", line).strip()
            if line and line not in cleaned:
                cleaned.append(line)
        if not cleaned:
            return [query]
        return [query] + cleaned[:num_variations]
    except Exception:
        return [query]


def reformulate_with_history(
    question: str,
    history: list[dict[str, str]],
) -> str:
    """Rewrite a follow-up question into a self-contained standalone query.

    When the user asks "explain the second point in more detail" the retriever
    has no idea what "the second point" refers to.  This function uses the
    singleton LLM to fold the most recent chat context into the question so
    that ``retrieve()`` can find the right documents.

    Returns the original ``question`` unchanged if:
    - ``history`` is empty (first turn — no reformulation needed)
    - the LLM call fails for any reason (safe fallback)
    """
    if not history:
        return question

    # Use only the last 3 turns to keep the prompt short
    recent = history[-6:]  # 3 user + 3 assistant turns at most
    history_text = "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content'][:400]}"
        for m in recent
        if m.get("content")
    )

    prompt = (
        "Dựa vào lịch sử hội thoại bên dưới, hãy viết lại câu hỏi cuối cùng của người dùng "
        "thành một câu hỏi độc lập, đầy đủ ý nghĩa (không cần lịch sử để hiểu). "
        "Chỉ trả lời bằng câu hỏi được viết lại, không thêm giải thích.\n\n"
        f"Lịch sử:\n{history_text}\n\n"
        f"Câu hỏi cần viết lại: {question}"
    )
    try:
        llm = get_llm()
        resp = llm.invoke(prompt)
        rewritten = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        if rewritten and len(rewritten) > 5:
            logger.debug("Query reformulated: %r → %r", question, rewritten)
            return rewritten
    except Exception:
        logger.warning("Query reformulation failed; using original question.", exc_info=True)
    return question


# --- Qdrant client & hybrid search ---


def _query_to_sparse_vector(query: str, query_variations: list[str], vocab: dict[str, int]) -> qmodels.SparseVector | None:
    """Build sparse vector for query + variations. Returns None if vocab empty or no tokens."""
    from math import log
    tokens: list[str] = []
    tokens.extend(_tokenize_for_sparse(query))
    for qv in query_variations:
        tokens.extend(_tokenize_for_sparse(qv))
    tf: dict[int, float] = {}
    for t in tokens:
        idx = vocab.get(t)
        if idx is None:
            continue
        tf[idx] = tf.get(idx, 0) + 1
    if not tf:
        return None
    for k in tf:
        tf[k] = 1.0 + log(tf[k])
    indices = sorted(tf.keys())
    values = [float(tf[i]) for i in indices]
    return qmodels.SparseVector(indices=indices, values=values)


def _load_sparse_vocab(persist_dir: Path, collection_name: str) -> dict[str, int] | None:
    """Load sparse vocab for collection. Returns None if not found."""
    path = persist_dir / f"{collection_name}_sparse_vocab.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hybrid_search(
    query: str,
    query_variations: list[str],
    embeddings,
    client: QdrantClient,
    collection_name: str,
    persist_dir: Path,
    top_k: int = 20,
) -> list[tuple[str, dict, float]]:
    """
    Hybrid search using Qdrant native sparse + dense vectors (prefetch + RRF).
    Returns list of (doc_id, metadata, score) with payload.
    """
    q_embeddings = embeddings.embed_documents(query_variations)
    sparse_vocab = _load_sparse_vocab(persist_dir, collection_name)
    query_sparse = _query_to_sparse_vector(query, query_variations, sparse_vocab) if sparse_vocab else None

    prefetches: list[qmodels.Prefetch] = []
    for qe in q_embeddings:
        prefetches.append(
            qmodels.Prefetch(
                query=qe,
                using="dense",
                limit=max(20, top_k),
            )
        )
    if query_sparse is not None:
        prefetches.append(
            qmodels.Prefetch(
                query=query_sparse,
                using="sparse",
                limit=max(20, top_k),
            )
        )

    if not prefetches:
        return []

    resp = client.query_points(
        collection_name=collection_name,
        prefetch=prefetches,
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    out: list[tuple[str, dict, float]] = []
    for point in (resp.points or []):
        score = float(point.score) if point.score is not None else 0.0
        payload = point.payload or {}
        out.append((str(point.id), payload, score))
    return out


# --- Re-ranking ---

_reranker: Optional[CrossEncoder] = None
_reranker_lock = threading.Lock()


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:  # double-checked locking
                settings = get_settings()
                _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


# ---------------------------------------------------------------------------
# Singleton LLM & embeddings clients
# ---------------------------------------------------------------------------

_embeddings: Any = None
_embeddings_lock = threading.Lock()
_llm: Optional[ChatOpenAI] = None
_llm_lock = threading.Lock()


def get_embeddings():
    """Return a process-wide singleton embedding client.

    HuggingFace model loading takes 3-10 s; we only want to pay that cost once.
    Thread-safe via double-checked locking.
    """
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                settings = get_settings()
                if settings.embedding_model.startswith("text-embedding"):
                    _embeddings = OpenAIEmbeddings(
                        model=settings.embedding_model,
                        api_key=settings.openai_api_key,
                    )
                else:
                    emb = HuggingFaceEmbeddings(
                        model_name=settings.embedding_model,
                        model_kwargs={"trust_remote_code": True, "device": "cpu"},
                    )
                    _fix_position_ids(emb.client)
                    _embeddings = emb
    return _embeddings


def get_llm() -> ChatOpenAI:
    """Return a process-wide singleton ChatOpenAI client for query expansion."""
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                settings = get_settings()
                _llm = ChatOpenAI(
                    model=settings.llm_model,
                    api_key=settings.openai_api_key,
                    temperature=0.1,
                )
    return _llm


def rerank(
    query: str,
    candidates: list[tuple[str, dict, float]],
    all_docs: list[str],
    all_ids: list[str],
    top_k: int = 5,
) -> list[tuple[str, dict, float]]:
    """Re-rank candidates with cross-encoder, return top_k."""
    if not candidates:
        return []
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
    pairs = []
    for id_, meta, _ in candidates:
        idx = id_to_idx.get(id_)
        if idx is not None:
            text = all_docs[idx]
            pairs.append((query, text))
        else:
            pairs.append((query, meta.get("parent_content", "")[:2000] or ""))
    reranker = get_reranker()
    scores = reranker.predict(pairs)
    scored = [(candidates[i][0], candidates[i][1], float(scores[i])) for i in range(len(candidates))]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


# --- Parent context ---

# Cache parent JSON files in memory for up to 5 minutes.
# Each entry is keyed by (collection_name, file_mtime) so the cache
# is automatically invalidated when the file is updated by a new ingest.
_parents_cache: TTLCache = TTLCache(maxsize=32, ttl=300)
_parents_cache_lock = threading.Lock()


def load_parents(collection_name: str) -> dict[str, dict]:
    """Load parent_id -> {content, summary, target_question, source} from JSON.

    Results are cached in memory for up to 5 minutes and invalidated whenever
    the underlying file changes (mtime-based key), so a new ingest is always
    picked up within at most one cache period.
    """
    settings = get_settings()
    path = settings.persist_dir / f"{collection_name}_parents.json"
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    cache_key = (collection_name, mtime)
    with _parents_cache_lock:
        cached = _parents_cache.get(cache_key)
        if cached is not None:
            return cached
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with _parents_cache_lock:
        _parents_cache[cache_key] = data
    return data


def fetch_parent_context(
    top_children: list[tuple[str, dict, float]],
    collection_name: str = "rag_chatbot",
) -> list[dict[str, Any]]:
    """
    Get parent chunks for the top child chunks. Returns list of context dicts
    with content, summary, source for the LLM.
    """
    parents = load_parents(collection_name)
    seen_parent_ids = set()
    context_list = []
    for i, (_, meta, score) in enumerate(top_children):
        parent_id = meta.get("parent_id")
        page = meta.get("page")
        if not parent_id or parent_id in seen_parent_ids:
            content = meta.get("parent_content", "")
            if content:
                context_list.append({
                    "content": content,
                    "summary": meta.get("summary", ""),
                    "source": meta.get("source", "Unknown"),
                    "rank": i + 1,
                    "score": score,
                    "page": page,
                })
            continue
        seen_parent_ids.add(parent_id)
        parent = parents.get(parent_id, {})
        context_list.append({
            "content": parent.get("content", meta.get("parent_content", "")),
            "summary": parent.get("summary", meta.get("summary", "")),
            "source": parent.get("source", meta.get("source", "Unknown")),
            "rank": len(context_list) + 1,
            "score": score,
            "page": page,
        })
    return context_list


def _retrieve_single_collection(
    query: str,
    collection_name: str,
    embeddings,
    llm: ChatOpenAI,
    client: QdrantClient,
    settings,
) -> list[dict[str, Any]]:
    """
    Run full retrieval pipeline on a single collection.
    Uses Qdrant native sparse + dense hybrid search (prefetch + RRF).
    Returns context list (may be empty).
    """
    variations = expand_query(
        query,
        llm,
        num_variations=settings.query_expansion_count,
    )

    candidates = hybrid_search(
        query,
        variations,
        embeddings,
        client,
        collection_name,
        persist_dir=settings.persist_dir,
        top_k=settings.hybrid_top_k,
    )

    if not candidates:
        return []

    all_ids = [c[0] for c in candidates]
    all_docs = [(c[1].get("text", "") or "") for c in candidates]

    top_5 = rerank(
        query,
        candidates,
        all_docs,
        all_ids,
        top_k=settings.rerank_top_k,
    )

    context_list = fetch_parent_context(top_5, collection_name=collection_name)
    for ctx in context_list:
        ctx["collection_name"] = collection_name
    return context_list


# --- Full retrieval pipeline ---

def retrieve(
    query: str,
    collections_to_search: list[str],
) -> list[dict[str, Any]]:
    """
    Run full retrieval using process-wide singleton clients.

    Queries all collections in `collections_to_search` in parallel via a thread pool.
    Gathers the top-K hybrid search results from each collection into a single pool,
    then runs the cross-encoder re-ranker over the combined pool to find the global top-K.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required.")

    # Singletons: first call initialises; subsequent calls return instantly.
    embeddings = get_embeddings()
    llm = get_llm()
    client = get_qdrant_client(settings.persist_dir)

    if not collections_to_search:
        return []

    # Parallel hybrid search: get un-ranked candidates from each collection
    def _hybrid_search_one(coll_name: str) -> list[tuple[str, dict, float, str]]:
        try:
            variations = expand_query(query, llm, num_variations=settings.query_expansion_count)
            candidates = hybrid_search(
                query, variations, embeddings, client, coll_name,
                persist_dir=settings.persist_dir, top_k=settings.hybrid_top_k
            )
            # Tag each candidate with its origin collection
            return [(doc_id, meta, score, coll_name) for (doc_id, meta, score) in candidates]
        except ValueError as e:
            if "not found" in str(e).lower():
                logger.debug("Collection '%s' does not exist yet.", coll_name)
            else:
                logger.warning("Hybrid search ValueError for collection '%s': %s", coll_name, e)
            return []
        except Exception:
            logger.warning("Hybrid search failed for collection '%s'.", coll_name, exc_info=False)
            return []

    combined_candidates: list[tuple[str, dict, float, str]] = []
    with ThreadPoolExecutor(max_workers=min(len(collections_to_search), 4)) as pool:
        futures = {pool.submit(_hybrid_search_one, name): name for name in collections_to_search}
        for future in as_completed(futures):
            combined_candidates.extend(future.result())

    if not combined_candidates:
        return []

    # Unified cross-encoder re-ranking over the combined candidate pool
    # Strip the collection name temporarily for the rerank function signature
    candidates_for_rerank = [(doc_id, meta, score) for doc_id, meta, score, _ in combined_candidates]
    all_ids = [c[0] for c in combined_candidates]
    all_docs = [(c[1].get("text", "") or "") for c in combined_candidates]

    top_k = rerank(
        query,
        candidates_for_rerank,
        all_docs,
        all_ids,
        top_k=settings.rerank_top_k,
    )

    # Re-attach collection names and fetch parent contexts
    final_context_list = []
    for rank, (doc_id, meta, score) in enumerate(top_k, 1):
        # find original collection name
        coll_name = next(orig_c for orig_id, _, _, orig_c in combined_candidates if orig_id == doc_id)
        
        # We fetch the parent block for this chunk
        parent_contexts = fetch_parent_context([(doc_id, meta, score)], collection_name=coll_name)
        if parent_contexts:
            ctx = parent_contexts[0]
            ctx["collection_name"] = coll_name
            ctx["rank"] = rank
            final_context_list.append(ctx)

    return final_context_list
