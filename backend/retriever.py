"""Retrieval: query expansion, hybrid search (pgvector dense + BM25 sparse), re-ranking, parent context."""
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from config import get_settings
from legal_tokenizer import legal_tokenize_query as _legal_tokenize_query
from pg_client import (
    load_sparse_bm25_stats,
    load_sparse_vocab_map,
    load_vector_parents,
    similarity_search,
)
from utils import fix_position_ids as _fix_position_ids, tokenize_for_sparse as _tokenize_for_sparse

logger = logging.getLogger(__name__)


# --- Query expansion ---

def _ingredient_hint_variations(ingredient_hints: list[str] | None) -> list[str]:
    if not ingredient_hints:
        return []
    out: list[str] = []
    for ingredient in ingredient_hints:
        ing = (ingredient or "").strip()
        if not ing:
            continue
        out.extend([ing, f"hoạt chất {ing}", f"thành phần {ing}"])
    dedup: list[str] = []
    seen: set[str] = set()
    for q in out:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(q)
    return dedup


def _context_pack_variations(context_pack: dict[str, Any] | None) -> list[str]:
    if not context_pack:
        return []

    entities = [str(x).strip() for x in (context_pack.get("active_entities") or []) if str(x).strip()]
    topics = [str(x).strip() for x in (context_pack.get("active_topics") or []) if str(x).strip()]
    latest_summary = str(context_pack.get("latest_summary") or "").strip()

    out: list[str] = []
    if entities:
        joined_entities = ", ".join(entities[:6])
        out.extend([
            f"thực thể liên quan: {joined_entities}",
            f"công dụng của {joined_entities}",
        ])
    if topics:
        out.append(f"chủ đề: {', '.join(topics[:4])}")
    if latest_summary:
        out.append(f"ngữ cảnh hội thoại gần nhất: {latest_summary[:240]}")

    dedup: list[str] = []
    seen: set[str] = set()
    for q in out:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(q)
    return dedup


def rewrite_and_expand_query(
    query: str,
    llm: ChatOpenAI,
    num_variations: int = 3,
    history: list[dict[str, str]] | None = None,
    ingredient_hints: list[str] | None = None,
    context_pack: dict[str, Any] | None = None,
) -> list[str]:
    """Build standalone query + retrieval variations in a single LLM call.

    Output order is:
    1) standalone/re-written query (or original query as fallback)
    2) hint-based synthetic variations
    3) model-generated semantic variations
    """

    def _context_pack_anchor(pack: dict[str, Any] | None) -> str:
        if not pack:
            return ""
        entities = [str(x).strip() for x in (pack.get("active_entities") or []) if str(x).strip()]
        topics = [str(x).strip() for x in (pack.get("active_topics") or []) if str(x).strip()]
        latest_summary = str(pack.get("latest_summary") or "").strip()
        parts: list[str] = []
        if entities:
            parts.append("Thực thể đang trao đổi: " + ", ".join(entities[:6]))
        if topics:
            parts.append("Chủ đề đang trao đổi: " + ", ".join(topics[:4]))
        if latest_summary:
            parts.append("Tóm tắt gần nhất: " + latest_summary[:240])
        return "\n".join(parts)

    base = [query] + _ingredient_hint_variations(ingredient_hints) + _context_pack_variations(context_pack)

    # Keep prompt context bounded.
    recent = (history or [])[-6:]
    history_lines: list[str] = []
    for m in recent:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        role_label = "Người dùng" if m.get("role") == "user" else "Trợ lý"
        history_lines.append(f"{role_label}: {content[:400]}")
    history_text = "\n".join(history_lines)
    context_anchor = _context_pack_anchor(context_pack)

    prompt = (
        "Bạn là trợ lý truy vấn tài liệu. Hãy thực hiện đồng thời 2 việc:\n"
        "(1) Viết lại câu hỏi thành 1 truy vấn độc lập (standalone_query).\n"
        f"(2) Tạo đúng {num_variations} biến thể truy vấn ngắn để tăng recall (variations).\n\n"
        "YÊU CẦU:\n"
        "- variations KHÔNG chứa standalone_query y nguyên.\n"
        "- Giữ nguyên thực thể/thuật ngữ pháp lý quan trọng nếu có.\n"
        "- Trả lời CHỈ bằng JSON hợp lệ theo schema:\n"
        '{"standalone_query": "...", "variations": ["...", "..."]}\n\n'
        f"Lịch sử gần đây:\n{history_text or '(không có)'}\n\n"
        f"Ngữ cảnh bổ sung:\n{context_anchor or '(không có)'}\n\n"
        f"Câu hỏi gốc: {query}"
    )

    try:
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        text = content.strip()
        parsed: dict[str, Any] | None = None

        for candidate in [text, None]:
            if candidate is None:
                start = text.find("{")
                end = text.rfind("}")
                if start == -1 or end <= start:
                    break
                candidate = text[start : end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    parsed = obj
                    break
            except Exception:
                continue

        if not parsed:
            return expand_query(
                query,
                llm,
                num_variations=num_variations,
                ingredient_hints=ingredient_hints,
                context_pack=context_pack,
            )

        standalone = str(parsed.get("standalone_query") or "").strip() or query
        raw_variations = parsed.get("variations") or []
        if not isinstance(raw_variations, list):
            raw_variations = []

        cleaned: list[str] = []
        for item in raw_variations:
            line = re.sub(r"^[\d\.\)\-\*]+\s*", "", str(item or "")).strip()
            if line and line not in cleaned:
                cleaned.append(line)

        merged = [standalone] + base[1:] + cleaned[:num_variations]
        dedup: list[str] = []
        seen: set[str] = set()
        for q in merged:
            key = q.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            dedup.append(q)
        return dedup or [query]
    except Exception:
        return expand_query(
            query,
            llm,
            num_variations=num_variations,
            ingredient_hints=ingredient_hints,
            context_pack=context_pack,
        )


def expand_query(
    query: str,
    llm: ChatOpenAI,
    num_variations: int = 3,
    ingredient_hints: list[str] | None = None,
    context_pack: dict[str, Any] | None = None,
) -> list[str]:
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
        base = [query] + _ingredient_hint_variations(ingredient_hints) + _context_pack_variations(context_pack)
        if not cleaned:
            return base
        merged = base + cleaned[:num_variations]
        dedup: list[str] = []
        seen: set[str] = set()
        for q in merged:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(q)
        return dedup
    except Exception:
        base = [query] + _ingredient_hint_variations(ingredient_hints) + _context_pack_variations(context_pack)
        dedup: list[str] = []
        seen: set[str] = set()
        for q in base:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(q)
        return dedup


def reformulate_with_history(
    question: str,
    history: list[dict[str, str]],
    context_pack: dict[str, Any] | None = None,
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
    def _is_context_dependent(text: str) -> bool:
        t = (text or "").lower()
        if not t:
            return False
        patterns = [
            r"\btrên\b",
            r"\bở trên\b",
            r"\bđó\b",
            r"\bnày\b",
            r"\bvừa nêu\b",
            r"\bcác chất\b",
            r"\btừng chất\b",
            r"\bchúng\b",
            r"\bnó\b",
            r"\bthose\b",
            r"\bthat\b",
            r"\bit\b",
        ]
        return any(re.search(p, t) for p in patterns)

    def _context_pack_anchor(pack: dict[str, Any] | None) -> str:
        if not pack:
            return ""
        entities = [str(x).strip() for x in (pack.get("active_entities") or []) if str(x).strip()]
        topics = [str(x).strip() for x in (pack.get("active_topics") or []) if str(x).strip()]
        parts: list[str] = []
        if entities:
            parts.append("Thực thể đang trao đổi: " + ", ".join(entities[:6]))
        if topics:
            parts.append("Chủ đề đang trao đổi: " + ", ".join(topics[:4]))
        return "; ".join(parts)

    if not history and not context_pack:
        return question

    # Very conservative upper-bounds so we never risk blowing the LLM context
    # window, even if caller passes huge messages.
    MAX_HISTORY_CHARS = 4000
    MAX_QUESTION_CHARS = 1000

    # Use only the last 3 turns to keep the prompt short
    recent = history[-6:]  # 3 user + 3 assistant turns at most
    history_chunks: list[str] = []
    for m in recent:
        content = m.get("content") or ""
        if not content:
            continue
        role_label = "Người dùng" if m.get("role") == "user" else "Trợ lý"
        # Hard-cap each message to avoid any single giant block dominating.
        snippet = content[:400]
        history_chunks.append(f"{role_label}: {snippet}")

    history_text = "\n".join(history_chunks)
    if len(history_text) > MAX_HISTORY_CHARS:
        # Keep the most recent portion of the history text.
        history_text = history_text[-MAX_HISTORY_CHARS:]

    safe_question = (question or "").strip()
    if len(safe_question) > MAX_QUESTION_CHARS:
        safe_question = safe_question[:MAX_QUESTION_CHARS]

    context_anchor = _context_pack_anchor(context_pack)
    context_dependent = _is_context_dependent(safe_question)

    prompt = (
        "Dựa vào lịch sử hội thoại, hãy viết lại câu hỏi cuối cùng của người dùng "
        "thành một câu hỏi tìm kiếm trọn vẹn, độc lập. BẮT BUỘC phải giữ lại tên các Luật, "
        "Nghị định, số Điều, Khoản hoặc chủ đề pháp lý đang được nhắc đến trong bối cảnh. "
        "Nếu người dùng dùng đại từ thay thế như 'trường hợp này', 'điều đó', 'luật trên', "
        "hãy thay thế bằng danh từ/ngữ cảnh cụ thể từ lịch sử.\n\n"
        "Chỉ trả lời bằng câu hỏi được viết lại, không thêm giải thích.\n\n"
        f"Lịch sử:\n{history_text}\n\n"
        f"Câu hỏi cần viết lại: {safe_question}"
    )
    if context_anchor:
        prompt += f"\n\nNgữ cảnh phiên hội thoại bổ sung: {context_anchor}"

    try:
        llm = get_llm()
        resp = llm.invoke(prompt)
        rewritten = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        if rewritten and len(rewritten) > 5:
            if context_dependent and context_anchor:
                rewritten = f"{rewritten}. {context_anchor}"
            logger.debug("Query reformulated: %r → %r", question, rewritten)
            return rewritten
    except Exception:
        logger.warning("Query reformulation failed; using original question.", exc_info=True)
    if context_dependent and context_anchor:
        return f"{question}. {context_anchor}"
    return question


def _query_to_sparse_indices_values(
    query: str,
    query_variations: list[str],
    vocab: dict[str, int],
    idf_map: dict[str, float] | None = None,
) -> tuple[list[int], list[float]]:
    """Build sparse vector (indices, values) for query + variations using BM25 IDF weighting."""
    from math import log
    tokens: list[str] = []
    tokens.extend(_legal_tokenize_query(query))
    for qv in query_variations:
        tokens.extend(_legal_tokenize_query(qv))

    tf: dict[int, int] = {}
    for t in tokens:
        idx = vocab.get(t)
        if idx is None:
            continue
        tf[idx] = tf.get(idx, 0) + 1
    if not tf:
        return [], []

    idx_to_token: dict[int, str] = {}
    if idf_map:
        idx_to_token = {v: k for k, v in vocab.items()}

    scores: dict[int, float] = {}
    for idx, raw_tf in tf.items():
        if idf_map:
            token = idx_to_token.get(idx, "")
            scores[idx] = idf_map.get(token, 1.0) * (1.0 + log(raw_tf))
        else:
            scores[idx] = 1.0 + log(raw_tf)

    indices = sorted(scores.keys())
    values = [float(scores[i]) for i in indices]
    return indices, values


def _load_sparse_vocab(collection_name: str) -> dict[str, int] | None:
    """Load sparse vocab for collection from PostgreSQL. Returns None if not found."""
    vocab = load_sparse_vocab_map(collection_name)
    return vocab or None


def _bm25_rescore(
    rows: list[dict],
    query_indices: list[int],
    query_values: list[float],
) -> list[dict]:
    """Add a BM25 sparse score component to dense similarity results."""
    if not query_indices or not rows:
        return rows
    q_vec = {idx: val for idx, val in zip(query_indices, query_values)}
    for row in rows:
        sparse_indices = row.get("sparse_indices") or []
        sparse_values = row.get("sparse_values") or []
        d_vec = {idx: val for idx, val in zip(sparse_indices, sparse_values)}
        dot = sum(q_vec.get(idx, 0.0) * val for idx, val in d_vec.items())
        dense_score = float(row.get("score", 0.0))
        # Simple RRF-style combination: 0.7 dense + 0.3 sparse (normalised)
        row["score"] = 0.7 * dense_score + 0.3 * min(dot, 1.0)
    return rows


def hybrid_search(
    query: str,
    query_variations: list[str],
    embeddings,
    collection_name: str,
    top_k: int = 20,
    pharma_only: bool = False,
) -> list[tuple[str, dict, float]]:
    """
    Hybrid search using pgvector dense + BM25 sparse re-scoring.
    Returns list of (doc_id, metadata, score).
    """
    q_embeddings = embeddings.embed_documents(query_variations)
    # Use the first embedding (primary query) for dense search
    primary_embedding = q_embeddings[0] if q_embeddings else []

    payload_filter = {"is_pharma_related": True} if pharma_only else None
    rows = similarity_search(
        collection_name=collection_name,
        query_embedding=primary_embedding,
        top_k=top_k,
        payload_filter=payload_filter,
    )

    # Load sparse vocab for BM25 enrichment
    sparse_vocab = _load_sparse_vocab(collection_name)
    if sparse_vocab:
        _, bm25_idf_map = load_sparse_bm25_stats(collection_name)
        q_indices, q_values = _query_to_sparse_indices_values(
            query, query_variations, sparse_vocab, bm25_idf_map or None
        )
        rows = _bm25_rescore(rows, q_indices, q_values)

    out: list[tuple[str, dict, float]] = []
    for row in rows:
        score = float(row.get("score", 0.0))
        payload = row.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        # Attach sparse vectors so rerank can access them
        payload["_sparse_indices"] = row.get("sparse_indices") or []
        payload["_sparse_values"] = row.get("sparse_values") or []
        out.append((str(row["id"]), payload, score))
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
                kwargs: dict[str, Any] = {
                    "model": settings.llm_model,
                    "api_key": settings.openai_api_key,
                }
                if not (settings.llm_model or "").lower().startswith("gpt-5"):
                    kwargs["temperature"] = 0.1
                _llm = ChatOpenAI(**kwargs)
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

def load_parents(collection_name: str) -> dict[str, dict]:
    """Load parent_id -> {content, summary, target_question, source} from MongoDB."""
    return load_vector_parents(collection_name)


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
    settings,
    ingredient_hints: list[str] | None = None,
    context_pack: dict[str, Any] | None = None,
    pharma_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Run full retrieval pipeline on a single collection.
    Uses pgvector dense + BM25 sparse hybrid search.
    Returns context list (may be empty).
    """
    variations = expand_query(
        query,
        llm,
        num_variations=settings.query_expansion_count,
        ingredient_hints=ingredient_hints,
        context_pack=context_pack,
    )

    candidates = hybrid_search(
        query,
        variations,
        embeddings,
        collection_name,
        top_k=settings.hybrid_top_k,
        pharma_only=pharma_only,
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
    history: list[dict[str, str]] | None = None,
    ingredient_hints: list[str] | None = None,
    context_pack: dict[str, Any] | None = None,
    pharma_only: bool = False,
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

    if not collections_to_search:
        return []

    query_variations = rewrite_and_expand_query(
        query,
        llm,
        num_variations=settings.query_expansion_count,
        history=history,
        ingredient_hints=ingredient_hints,
        context_pack=context_pack,
    )
    effective_query = query_variations[0] if query_variations else query

    # Parallel hybrid search: get un-ranked candidates from each collection
    def _hybrid_search_one(coll_name: str) -> list[tuple[str, dict, float, str]]:
        try:
            candidates = hybrid_search(
                effective_query,
                query_variations,
                embeddings,
                coll_name,
                top_k=settings.hybrid_top_k,
                pharma_only=pharma_only,
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
        effective_query,
        candidates_for_rerank,
        all_docs,
        all_ids,
        top_k=settings.rerank_top_k,
    )

    if settings.legal_collection_name in collections_to_search:
        top_k = _augment_legal_article_candidates(
            query=effective_query,
            reranked=top_k,
            combined_candidates=combined_candidates,
            legal_collection_name=settings.legal_collection_name,
            extra_k=2,
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

    if settings.enable_context_rescore and final_context_list:
        final_context_list = _context_aware_rescore(final_context_list, context_pack)

    return final_context_list


def _context_aware_rescore(
    context_list: list[dict[str, Any]],
    context_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not context_pack:
        return context_list

    entities = [str(x).strip().lower() for x in (context_pack.get("active_entities") or []) if str(x).strip()]
    topics = [str(x).strip().lower() for x in (context_pack.get("active_topics") or []) if str(x).strip()]
    anchors = entities + topics
    if not anchors:
        return context_list

    rescored: list[dict[str, Any]] = []
    for ctx in context_list:
        base_score = float(ctx.get("score", 0.0) or 0.0)
        haystack = " ".join(
            [
                str(ctx.get("content", "")),
                str(ctx.get("source", "")),
                str(ctx.get("collection_name", "")),
            ]
        ).lower()
        matches = sum(1 for anchor in anchors if anchor and anchor in haystack)
        boost = min(0.6, matches * 0.15)
        payload = dict(ctx)
        payload["score"] = base_score + boost
        rescored.append(payload)

    rescored.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    for index, item in enumerate(rescored, 1):
        item["rank"] = index
    return rescored


def _extract_first_article_number(text: str) -> str | None:
    t = (text or "").lower()
    m = re.search(r"\bđiều\s+(\d+)\b", t)
    if m:
        return m.group(1)
    m_ascii = re.search(r"\bdieu\s+(\d+)\b", t)
    if m_ascii:
        return m_ascii.group(1)
    return None


def _mentions_article(text: str, article_no: str) -> bool:
    if not article_no:
        return False
    t = (text or "").lower()
    return bool(
        re.search(rf"\bđiều\s+{re.escape(article_no)}\b", t)
        or re.search(rf"\bdieu\s+{re.escape(article_no)}\b", t)
    )


def _looks_like_follow_up_clause(text: str) -> bool:
    """Heuristic for legal clause fragments that often omit article heading.

    Typical examples:
    - "2. Người chịu trách nhiệm..."
    - "Khoản 2 ... Điều này"
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    if re.match(r"^\s*[2-9][\.|\)]\s+", t):
        return True
    if re.search(r"\bkhoản\s+[2-9]\b", t) and (
        "điều này" in t or "dieu nay" in t
    ):
        return True
    return False


def _augment_legal_article_candidates(
    query: str,
    reranked: list[tuple[str, dict, float]],
    combined_candidates: list[tuple[str, dict, float, str]],
    legal_collection_name: str,
    extra_k: int = 2,
) -> list[tuple[str, dict, float]]:
    """
    Improve legal coverage when the top result contains an article heading (e.g., "Điều 19")
    but rerank list only keeps one fragment of that article.

    Strategy:
    - detect target article number from query, else from top reranked legal chunk text
    - if current reranked list has <2 chunks mentioning that article,
      append up to `extra_k` additional legal chunks that also mention the same article
      (ordered by hybrid score descending).
    """
    if not reranked or not combined_candidates:
        return reranked

    article_no = _extract_first_article_number(query)
    if not article_no:
        for _doc_id, meta, _score in reranked:
            article_no = _extract_first_article_number(str(meta.get("text", "") or ""))
            if article_no:
                break
    if not article_no:
        return reranked

    current_mentions = sum(
        1 for _doc_id, meta, _score in reranked if _mentions_article(str(meta.get("text", "") or ""), article_no)
    )
    if current_mentions >= 2:
        return reranked

    selected_ids = {doc_id for doc_id, _meta, _score in reranked}
    anchor_source = None
    for _doc_id, meta, _score in reranked:
        text = str(meta.get("text", "") or "")
        if _mentions_article(text, article_no):
            anchor_source = str(meta.get("source", "") or "")
            break

    extras: list[tuple[str, dict, float]] = []
    for doc_id, meta, score, coll_name in combined_candidates:
        if coll_name != legal_collection_name:
            continue
        if doc_id in selected_ids:
            continue
        candidate_text = str(meta.get("text", "") or "")
        same_article = _mentions_article(candidate_text, article_no)
        same_source_follow_up = (
            bool(anchor_source)
            and str(meta.get("source", "") or "") == anchor_source
            and _looks_like_follow_up_clause(candidate_text)
        )
        if not (same_article or same_source_follow_up):
            continue
        extras.append((doc_id, meta, score))

    if not extras:
        return reranked

    extras.sort(key=lambda x: x[2], reverse=True)
    return reranked + extras[: max(0, int(extra_k))]
