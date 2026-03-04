"""Retrieval: query expansion, hybrid search (vector + Qdrant native sparse), re-ranking, parent context."""
import json
import re
from pathlib import Path
from typing import Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import get_settings


def _fix_position_ids(model) -> None:
    """Repair corrupted ``position_ids`` buffers after model load.

    ``transformers >= 5`` lazy-materialises weights which can leave
    ``persistent=False`` buffers (like ``position_ids``) filled with
    uninitialised memory.  Re-creating the buffer with ``torch.arange``
    restores the correct values.
    """
    import torch

    for module in model.modules():
        if hasattr(module, "position_ids") and isinstance(module.position_ids, torch.Tensor):
            n = module.position_ids.size(0)
            module.register_buffer(
                "position_ids", torch.arange(n, dtype=torch.long), persistent=False,
            )


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


# --- Qdrant client & hybrid search ---

def get_qdrant_client(persist_dir: Path) -> QdrantClient:
    """Connect to local Qdrant (embedded) under persist_dir."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    path = persist_dir / "storage"
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def _tokenize_for_sparse(text: str) -> list[str]:
    """Tokenize for sparse vector (Vietnamese and English)."""
    text = text.lower().strip()
    return re.findall(r"\b\w+\b", text)


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


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        settings = get_settings()
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


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

def load_parents(collection_name: str) -> dict[str, dict]:
    """Load parent_id -> {content, summary, target_question, source} from JSON."""
    settings = get_settings()
    path = settings.persist_dir / f"{collection_name}_parents.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    collection_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Run full retrieval. If collection_name is given, query only that collection.
    Otherwise, try all collections and return the one with the best score.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required.")

    if settings.embedding_model.startswith("text-embedding"):
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"trust_remote_code": True, "device": "cpu"},
        )
        _fix_position_ids(embeddings.client)
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
    )
    client = get_qdrant_client(settings.persist_dir)

    if collection_name:
        return _retrieve_single_collection(
            query=query,
            collection_name=collection_name,
            embeddings=embeddings,
            llm=llm,
            client=client,
            settings=settings,
        )

    try:
        collections_response = client.get_collections()
        collections = [c.name for c in collections_response.collections]
    except Exception:
        return []

    best_context: list[dict[str, Any]] = []
    best_score: float = float("-inf")

    for name in collections:
        ctx = _retrieve_single_collection(
            query=query,
            collection_name=name,
            embeddings=embeddings,
            llm=llm,
            client=client,
            settings=settings,
        )
        if not ctx:
            continue
        top_score = ctx[0].get("score", 0.0)
        if top_score > best_score:
            best_score = top_score
            best_context = ctx

    return best_context
