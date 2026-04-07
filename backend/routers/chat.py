"""Chat router: chat sessions, messages, /ask endpoint, drug price."""

import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter

from config import get_settings
from deps import (
    AskRequest,
    AskResponse,
    ChatSession,
    ChatMessageOut,
    ChatMessage,
    DrugPriceRequest,
    CurrentUser,
    get_current_user,
)
from pg_client import (
    get_chat_sessions_collection,
    get_chat_messages_collection,
)
from prompts import (
    generate_answer,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    DRUG_SYSTEM_PROMPT,
    DRUG_USER_PROMPT_TEMPLATE,
    PRICE_SYSTEM_PROMPT,
    PRICE_USER_PROMPT_TEMPLATE,
    COMBINED_SYSTEM_PROMPT,
    COMBINED_USER_PROMPT_TEMPLATE,
)
from supervisor import get_intent_from_supervisor
from agents import run_price_agent, run_federated_rag_agent
from llm_usage import record_usage

router = APIRouter(tags=["chat"])
HISTORY_MAX_TURNS = 4
HISTORY_MAX_MESSAGES = HISTORY_MAX_TURNS * 2
HISTORY_MAX_CHARS = 800
HISTORY_RECENT_TURNS = 3
HISTORY_RECENT_MESSAGES = HISTORY_RECENT_TURNS * 2
HISTORY_SUMMARY_MAX_CHARS = 500


def _sanitize_history(history: list[ChatMessage] | None) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for m in history or []:
        role = (m.role or "").strip()
        content = (m.content or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        sanitized.append({"role": role, "content": content[:HISTORY_MAX_CHARS]})
    return sanitized[-HISTORY_MAX_MESSAGES:]


def _build_history_summary(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for m in history:
        role = "Người dùng" if m.get("role") == "user" else "Trợ lý"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:120]}")
    if not lines:
        return ""
    summary = " | ".join(lines)
    return summary[:HISTORY_SUMMARY_MAX_CHARS]


def _prepare_history_for_prompt(history: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    recent = history[-HISTORY_RECENT_MESSAGES:]
    older = history[:-HISTORY_RECENT_MESSAGES]
    summary = _build_history_summary(older)
    return recent, summary


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------

@router.post("/chat/sessions", response_model=ChatSession)
def create_chat_session(
    title: str = "New chat",
    current_user: CurrentUser = Depends(get_current_user),
):
    coll = get_chat_sessions_collection()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = {
        "user_id": current_user.id,
        "title": title,
        "created_at": now,
        "updated_at": now,
    }
    coll.insert_one(doc)
    return ChatSession(id=str(doc["_id"]), title=doc["title"], created_at=doc["created_at"], updated_at=doc["updated_at"])


@router.get("/chat/sessions", response_model=list[ChatSession])
def list_chat_sessions(current_user: CurrentUser = Depends(get_current_user)):
    docs = get_chat_sessions_collection().find({"user_id": current_user.id})
    return [
        ChatSession(id=d["_id"], title=d.get("title", "New chat"), created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""))
        for d in docs
    ]


@router.delete("/chat/sessions/{session_id}", status_code=204)
def delete_chat_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    sess_coll = get_chat_sessions_collection()
    result = sess_coll.delete_one({"_id": session_id, "user_id": current_user.id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    get_chat_messages_collection().delete_many({"session_id": session_id, "user_id": current_user.id})


@router.get("/chat/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
def get_chat_messages(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    docs = get_chat_messages_collection().find({"session_id": session_id, "user_id": current_user.id})
    return [
        ChatMessageOut(
            id=str(d["_id"]),
            role=d["role"],
            content=d["content"],
            created_at=d.get("created_at", ""),
            sources=d.get("sources"),
            priceData=d.get("priceData") or d.get("price_data"),
            feedback=d.get("feedback"),
            feedbackComment=d.get("feedbackComment") or d.get("feedback_comment"),
        )
        for d in docs
    ]


# ---------------------------------------------------------------------------
# /ask — main RAG endpoint (rate limit applied in app.py via limiter)
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse)
def ask(request: Request, req: AskRequest, current_user: CurrentUser = Depends(get_current_user)):
    """Retrieve context and return grounded answer with citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    try:
        intent = get_intent_from_supervisor(req.question)
        settings = get_settings()

        price_data: Optional[dict[str, Any]] = None
        price_ctx: Optional[dict[str, Any]] = None
        rag_docs: list[dict[str, Any]] = []

        # Build history first — needed by retrieval for follow-up reformulation
        history_payload = _sanitize_history(req.history)
        recent_history, history_summary = _prepare_history_for_prompt(history_payload)

        # 1. SQL Inventory Agent
        price_data, price_ctx = run_price_agent(req.question, intent)

        # 2. RAG agent — routed by intent["collections_to_search"]
        collections = intent.get("collections_to_search", ["drug"])
        physical_collections = []
        for c in collections:
            if c == "legal":
                physical_collections.append(settings.legal_collection_name)
            elif c == "drug":
                physical_collections.append(settings.drug_collection_name)
        if physical_collections:
            rag_docs = run_federated_rag_agent(
                req.question,
                physical_collections,
                history=recent_history,
                history_summary=history_summary,
            )

        # 3. Assemble context
        final_contexts: list[dict[str, Any]] = []
        if rag_docs:
            final_contexts.extend(rag_docs)
        if price_ctx:
            final_contexts.append(price_ctx)

        # 4. Choose prompt template
        has_rag = bool(rag_docs)
        has_price = price_ctx is not None
        has_legal = "legal" in collections
        is_legal_only = collections == ["legal"] or physical_collections == [settings.legal_collection_name]
        is_drug_only = not has_price and (collections == ["drug"] or physical_collections == [settings.drug_collection_name])
        if has_rag and has_price and has_legal:
            # Only use COMBINED when mixing legal docs with price data
            system_prompt = COMBINED_SYSTEM_PROMPT
            user_template = COMBINED_USER_PROMPT_TEMPLATE
        elif has_price:
            system_prompt = PRICE_SYSTEM_PROMPT
            user_template = PRICE_USER_PROMPT_TEMPLATE
        elif is_legal_only:
            system_prompt = SYSTEM_PROMPT
            user_template = USER_PROMPT_TEMPLATE
        elif is_drug_only:
            system_prompt = DRUG_SYSTEM_PROMPT
            user_template = DRUG_USER_PROMPT_TEMPLATE
        else:
            # mixed legal+drug without price
            system_prompt = COMBINED_SYSTEM_PROMPT
            user_template = COMBINED_USER_PROMPT_TEMPLATE

        answer, usage = generate_answer(
            req.question,
            final_contexts,
            history=recent_history,
            history_summary=history_summary,
            system_prompt=system_prompt,
            user_template=user_template,
        )
        record_usage(prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0))
        sources = [
            {
                "rank": c.get("rank"),
                "source": c.get("source"),
                "summary": c.get("summary", "")[:200],
                "content": c.get("content", "")[:1000],
                "collection_name": c.get("collection_name"),
                "page": c.get("page"),
            }
            for c in final_contexts
        ]

        # 5. Persist to Mongo
        sessions_coll = get_chat_sessions_collection()
        msgs_coll = get_chat_messages_collection()
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        import uuid as _uuid
        _NS = _uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

        def _canonical_session_id(raw: str, user_id: str) -> str:
            """Return raw if it is already a valid UUID, otherwise derive a stable UUID v5."""
            try:
                _uuid.UUID(raw)
                return raw
            except (ValueError, AttributeError):
                return str(_uuid.uuid5(_NS, f"{user_id}:{raw}"))

        session_id = req.session_id
        if session_id:
            session_id = _canonical_session_id(session_id, current_user.id)
            sessions_coll.update_one(
                {"_id": session_id, "user_id": current_user.id},
                {"$set": {"updated_at": now}, "$setOnInsert": {"title": req.question.strip()[:80] or "New chat", "created_at": now}},
                upsert=True,
            )
        else:
            sess_doc = {"user_id": current_user.id, "title": req.question.strip()[:80] or "New chat", "created_at": now, "updated_at": now}
            sessions_coll.insert_one(sess_doc)
            session_id = str(sess_doc["_id"])

        msgs_coll.insert_one({"session_id": session_id, "user_id": current_user.id, "role": "user", "content": req.question, "created_at": now})
        msgs_coll.insert_one({"session_id": session_id, "user_id": current_user.id, "role": "assistant", "content": answer, "created_at": now, "sources": sources, "priceData": None})

        return AskResponse(
            answer=answer,
            sources=sources,
            has_context=len(final_contexts) > 0,
            collection_name=final_contexts[0].get("collection_name") if final_contexts else None,
            price_data=price_data,
            session_id=session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Drug price lookup (standalone endpoint)
# ---------------------------------------------------------------------------

@router.post("/drug-price")
def drug_price(req: DrugPriceRequest):
    if not req.drug_name.strip():
        raise HTTPException(status_code=400, detail="drug_name is required")
    try:
        return get_vietnam_drug_price(req.drug_name.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
