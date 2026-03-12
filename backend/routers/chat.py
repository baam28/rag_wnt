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
from mongo_client import (
    get_chat_sessions_collection,
    get_chat_messages_collection,
)
from prompts import (
    generate_answer,
    PRICE_SYSTEM_PROMPT,
    PRICE_USER_PROMPT_TEMPLATE,
    COMBINED_SYSTEM_PROMPT,
    COMBINED_USER_PROMPT_TEMPLATE,
)
from supervisor import get_intent_from_supervisor
from agents import run_price_agent, run_federated_rag_agent
from drug_price_tool import get_vietnam_drug_price

router = APIRouter(tags=["chat"])


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
        "_id": str(uuid.uuid4()),
        "user_id": current_user.id,
        "title": title,
        "created_at": now,
        "updated_at": now,
    }
    coll.insert_one(doc)
    return ChatSession(id=doc["_id"], title=doc["title"], created_at=doc["created_at"], updated_at=doc["updated_at"])


@router.get("/chat/sessions", response_model=list[ChatSession])
def list_chat_sessions(current_user: CurrentUser = Depends(get_current_user)):
    docs = get_chat_sessions_collection().find({"user_id": current_user.id}).sort("updated_at", -1)
    return [
        ChatSession(id=d["_id"], title=d.get("title", "New chat"), created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""))
        for d in docs
    ]


@router.get("/chat/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
def get_chat_messages(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    docs = get_chat_messages_collection().find({"session_id": session_id, "user_id": current_user.id}).sort("created_at", 1)
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
        history_payload = [
            {"role": m.role, "content": m.content}
            for m in (req.history or [])
            if m.content and m.role in ("user", "assistant")
        ]

        # 1. Price agent
        price_data, price_ctx = run_price_agent(req.question, intent)

        # 2. RAG agent — routed by intent["collections_to_search"]
        collections = intent.get("collections_to_search", ["drug_info"])
        physical_collections = []
        for c in collections:
            if c == "legal":
                physical_collections.append(settings.legal_collection_name)
            elif c == "drug_info":
                physical_collections.append(settings.drug_info_collection_name)
                
        if physical_collections:
            rag_docs = run_federated_rag_agent(req.question, physical_collections, history=history_payload)

        # 3. Assemble context
        final_contexts: list[dict[str, Any]] = []
        if rag_docs:
            final_contexts.extend(rag_docs)
        if price_ctx:
            final_contexts.append(price_ctx)

        # 4. Choose prompt template
        has_rag = bool(rag_docs)
        has_price = price_ctx is not None
        if has_rag and has_price:
            system_prompt = COMBINED_SYSTEM_PROMPT
            user_template = COMBINED_USER_PROMPT_TEMPLATE
        elif has_price:
            system_prompt = PRICE_SYSTEM_PROMPT
            user_template = PRICE_USER_PROMPT_TEMPLATE
        else:
            system_prompt = None
            user_template = None

        answer = generate_answer(req.question, final_contexts, history=history_payload, system_prompt=system_prompt, user_template=user_template)
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

        session_id = req.session_id
        if session_id:
            sessions_coll.update_one(
                {"_id": session_id, "user_id": current_user.id},
                {"$set": {"updated_at": now}, "$setOnInsert": {"title": req.question.strip()[:80] or "New chat", "created_at": now}},
                upsert=True,
            )
        else:
            sess_doc = {"_id": str(uuid.uuid4()), "user_id": current_user.id, "title": req.question.strip()[:80] or "New chat", "created_at": now, "updated_at": now}
            sessions_coll.insert_one(sess_doc)
            session_id = sess_doc["_id"]

        msgs_coll.insert_one({"session_id": session_id, "user_id": current_user.id, "role": "user", "content": req.question, "created_at": now})
        msgs_coll.insert_one({"session_id": session_id, "user_id": current_user.id, "role": "assistant", "content": answer, "created_at": now, "sources": sources, "priceData": price_data})

        return AskResponse(
            answer=answer,
            sources=sources,
            has_context=len(final_contexts) > 0,
            collection_name=final_contexts[0].get("collection_name") if final_contexts else None,
            price_data=price_data,
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
