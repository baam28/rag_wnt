"""Retrieval: query expansion, hybrid search (vector + BM25), re-ranking, parent context."""
import json
import re
from pathlib import Path
from typing import Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import get_settings


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
        # Remove numbering like "1.", "2."
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


def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple tokenization for BM25 (Vietnamese and English)."""
    text = text.lower().strip()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def hybrid_search(
    query: str,
    query_variations: list[str],
    embeddings,
    client: QdrantClient,
    collection_name: str,
    all_docs: list[str],
    all_metadatas: list[dict],
    all_ids: list[str],
    top_k: int = 20,
) -> list[tuple[str, dict, float]]:
    """
    Vector search (query + variations) plus BM25 on child chunks. Merge by id, return top_k.
    Returns list of (doc_id, metadata, score) sorted by combined relevance.
    """
    q_embeddings = embeddings.embed_documents(query_variations)
    seen_ids = set()
    vector_scores: dict[str, float] = {}
    n_results = max(20, top_k)
    for qe in q_embeddings:
        resp = client.query_points(
            collection_name=collection_name,
            query=qe,
            limit=n_results,
            with_payload=False,
            with_vectors=False,
        )
        for point in (resp.points or []):
            id_ = str(point.id)
            if id_ in seen_ids:
                continue
            seen_ids.add(id_)
            sim = float(point.score)
            vector_scores[id_] = max(vector_scores.get(id_, 0.0), sim)

    tokenized_corpus = [_tokenize_for_bm25(d) for d in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = _tokenize_for_bm25(query)
    for qv in query_variations:
        query_tokens.extend(_tokenize_for_bm25(qv))
    query_tokens = list(dict.fromkeys(query_tokens))
    bm25_scores = bm25.get_scores(query_tokens)

    # Normalize BM25 to ~[0,1] and merge with vector
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
    max_bm = max(bm25_scores) if len(bm25_scores) else 1.0
    combined: list[tuple[str, dict, float]] = []
    for id_ in set(vector_scores.keys()) | set(all_ids):
        idx = id_to_idx.get(id_)
        vs = vector_scores.get(id_, 0.0)
        bs = (bm25_scores[idx] / max_bm) if max_bm and idx is not None else 0.0
        combined_score = 0.6 * vs + 0.4 * bs
        meta = all_metadatas[idx] if idx is not None else {}
        combined.append((id_, meta, combined_score))

    combined.sort(key=lambda x: x[2], reverse=True)
    return combined[:top_k]


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
        if not parent_id or parent_id in seen_parent_ids:
            # Fallback: use parent_content from child metadata if stored
            content = meta.get("parent_content", "")
            if content:
                context_list.append({
                    "content": content,
                    "summary": meta.get("summary", ""),
                    "source": meta.get("source", "Unknown"),
                    "rank": i + 1,
                    "score": score,
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
    Returns context list (may be empty).
    """
    # Load all points (child chunks) from Qdrant to build BM25 corpus
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metadatas: list[dict] = []

    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            all_ids.append(str(p.id))
            payload = p.payload or {}
            all_docs.append(payload.get("text", ""))
            all_metadatas.append(payload)
        if next_offset is None:
            break

    if not all_ids:
        return []

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
        all_docs,
        all_metadatas,
        all_ids,
        top_k=settings.hybrid_top_k,
    )

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
        )
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
