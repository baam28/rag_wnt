"""Admin router: collections, docs, users, analytics, feedback endpoints."""

import time
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from config import get_settings
from deps import (
    CurrentUser,
    CollectionInfo,
    DocumentInfo,
    DeleteDocumentRequest,
    UserOut,
    RoleUpdateRequest,
    AdminSetPasswordRequest,
    FeedbackRequest,
    hash_password,
    get_current_admin,
)
from pg_client import (
    list_vector_parents_by_source,
    delete_collection_vector_meta,
    delete_vector_parents_by_source,
    delete_chunks_by_source,
    get_users_collection,
    get_chat_sessions_collection,
    get_chat_messages_collection,
    get_feedback_collection,
    list_collections,
    get_collection_count,
    _q,
)
from llm_usage import get_usage

router = APIRouter(tags=["admin"])


def _is_root_admin_identity(username: str | None, email: str | None) -> bool:
    u = (username or "").strip().lower()
    e = (email or "").strip().lower()
    return u == "admin" or e == "admin"


def _is_root_admin_doc(doc: dict[str, Any] | None) -> bool:
    if not doc:
        return False
    return _is_root_admin_identity(doc.get("username"), doc.get("email"))


def _is_root_admin_user(user: CurrentUser) -> bool:
    return _is_root_admin_identity(user.username, user.email)


def _last_n_dates(n: int = 30) -> list[str]:
    """Return list of date strings YYYY-MM-DD for the last n days (oldest first)."""
    utc = timezone.utc
    today = datetime.now(utc).date()
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n - 1, -1, -1)]


def _fill_series_by_date(date_list: list[str], raw: list[dict], date_key: str = "date", count_key: str = "count") -> list[dict]:
    """Merge raw [{ date, count }] into full date range; missing days get count 0."""
    by_date = {d[date_key]: d.get(count_key, 0) for d in raw}
    return [{"date": d, "count": by_date.get(d, 0)} for d in date_list]


# ---------------------------------------------------------------------------
# Feedback (public endpoint, no admin required)
# ---------------------------------------------------------------------------

@router.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "date": time.strftime("%Y-%m-%d"),
        "question": req.question[:500],
        "answer": req.answer,
        "rating": req.rating,
        "comment": (req.comment or "")[:500],
        "session_id": req.session_id or "",
    }
    get_feedback_collection().insert_one(entry)

    # Update the message record with feedback
    if req.session_id and req.answer:
        try:
            get_chat_messages_collection().update_one(
                {
                    "session_id": req.session_id,
                    "role": "assistant",
                    "content": req.answer,
                },
                {"$set": {"feedback": req.rating, "feedbackComment": req.comment or ""}},
            )
        except Exception:
            pass  # Non-fatal

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Admin-only endpoints
# ---------------------------------------------------------------------------

@router.get("/admin/feedback")
def get_feedback(current_user: CurrentUser = Depends(get_current_admin)):
    coll = get_feedback_collection()
    total = coll.count_documents({})
    up = coll.count_documents({"rating": "up"})
    down = coll.count_documents({"rating": "down"})
    data = coll.find({})
    return {
        "total": total,
        "up": up,
        "down": down,
        "down_entries": [d for d in data if d.get("rating") == "down"][:100],
        "all_entries": data[:200],
    }


@router.get("/admin/analytics")
def get_analytics(current_user: CurrentUser = Depends(get_current_admin)):
    total_users = get_users_collection().count_documents({})
    total_sessions = get_chat_sessions_collection().count_documents({})
    total_messages = get_chat_messages_collection().count_documents({})

    fb_coll = get_feedback_collection()
    fb_data = fb_coll.find({})

    date_list = _last_n_dates(30)

    # Users per day
    try:
        raw = get_users_collection().aggregate([])
        users_per_day = _fill_series_by_date(date_list, [{"date": d["_id"], "count": d["count"]} for d in raw])
    except Exception:
        users_per_day = [{"date": d, "count": 0} for d in date_list]

    # Sessions per day
    try:
        raw = get_chat_sessions_collection().aggregate([])
        sessions_per_day = _fill_series_by_date(date_list, [{"date": d["_id"], "count": d["count"]} for d in raw])
    except Exception:
        sessions_per_day = [{"date": d, "count": 0} for d in date_list]

    # Messages per day
    try:
        raw = get_chat_messages_collection().aggregate([])
        messages_per_day = _fill_series_by_date(date_list, [{"date": d["_id"], "count": d["count"]} for d in raw])
    except Exception:
        messages_per_day = [{"date": d, "count": 0} for d in date_list]

    # Feedback per day
    fb_by_date: dict[str, int] = {}
    for entry in fb_data:
        day = (entry.get("date") or "")[:10]
        if not day:
            ts = entry.get("timestamp", "")
            day = ts[:10] if len(ts) >= 10 else ""
        if day:
            fb_by_date[day] = fb_by_date.get(day, 0) + 1
    feedback_per_day = [{"date": d, "count": fb_by_date.get(d, 0)} for d in date_list]

    # LLM API usage
    llm = get_usage(days=30)
    daily_raw = {d["date"]: d for d in llm.get("daily", [])}
    llm_daily_filled = []
    for d in date_list:
        item = daily_raw.get(d, {})
        pt = item.get("prompt_tokens", 0)
        ct = item.get("completion_tokens", 0)
        llm_daily_filled.append({
            "date": d,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "requests": item.get("requests", 0),
            "count": pt + ct,
        })
    llm["daily"] = llm_daily_filled

    return {
        "total_users": total_users,
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "feedback": {
            "total": len(fb_data),
            "up": sum(1 for d in fb_data if d.get("rating") == "up"),
            "down": sum(1 for d in fb_data if d.get("rating") == "down"),
        },
        "users_per_day": users_per_day,
        "sessions_per_day": sessions_per_day,
        "messages_per_day": messages_per_day,
        "feedback_per_day": feedback_per_day,
        "llm_usage": llm,
    }


@router.get("/admin/collections", response_model=list[CollectionInfo])
def admin_list_collections(current_user: CurrentUser = Depends(get_current_admin)):
    try:
        names = list_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return [CollectionInfo(name=name) for name in names]


@router.get("/admin/docs", response_model=list[DocumentInfo])
def list_documents(
    collection_name: str = Query(..., alias="collection_name"),
    search: str = Query(""),
    sort_by: str = Query("document_date"),
    sort_order: str = Query("desc"),
    year_from: int | None = Query(None, ge=1900, le=2099),
    year_to: int | None = Query(None, ge=1900, le=2099),
    current_user: CurrentUser = Depends(get_current_admin),
):
    allowed_sort_by = {"document_date", "source"}
    allowed_sort_order = {"asc", "desc"}
    if sort_by not in allowed_sort_by:
        raise HTTPException(status_code=400, detail=f"sort_by must be one of: {', '.join(sorted(allowed_sort_by))}")
    if sort_order not in allowed_sort_order:
        raise HTTPException(status_code=400, detail=f"sort_order must be one of: {', '.join(sorted(allowed_sort_order))}")
    if year_from is not None and year_to is not None and year_from > year_to:
        raise HTTPException(status_code=400, detail="year_from must be <= year_to")

    try:
        rows = list_vector_parents_by_source(
            collection_name=collection_name,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            year_from=year_from,
            year_to=year_to,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return [
        DocumentInfo(
            source=str(row.get("source") or "Unknown"),
            parent_count=int(row.get("parent_count") or 0),
            document_year=row.get("document_year"),
            document_year_source=row.get("document_year_source"),
        )
        for row in rows
    ]


@router.delete("/admin/docs")
def delete_document(
    req: DeleteDocumentRequest,
    current_user: CurrentUser = Depends(get_current_admin),
):
    try:
        deleted = delete_chunks_by_source(req.collection_name, req.source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chunks: {e}")

    try:
        delete_vector_parents_by_source(req.collection_name, req.source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {e}")

    return {"message": f"Deleted document '{req.source}' from collection '{req.collection_name}'."}


@router.delete("/admin/collections/{collection_name}")
def delete_collection(
    collection_name: str,
    current_user: CurrentUser = Depends(get_current_admin),
):
    try:
        delete_collection_vector_meta(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": f"Cleared all data in collection '{collection_name}'.",
        "collection_name": collection_name,
    }


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

@router.get("/admin/users", response_model=list[UserOut])
def get_all_users(current_user: CurrentUser = Depends(get_current_admin)):
    docs = get_users_collection().find({})
    return [
        UserOut(
            id=str(d["_id"]),
            username=d.get("username") or d.get("email", ""),
            is_admin=bool(d.get("is_admin", False)),
        )
        for d in docs
    ]


@router.put("/admin/users/{user_id}/role")
def update_user_role(
    user_id: str,
    req: RoleUpdateRequest,
    current_user: CurrentUser = Depends(get_current_admin),
):
    if str(current_user.id) == str(user_id):
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    users_coll = get_users_collection()
    target = users_coll.find_one({"_id": user_id})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if _is_root_admin_doc(target) and not _is_root_admin_user(current_user):
        raise HTTPException(status_code=403, detail="Root admin role cannot be changed by other admin accounts")
    users_coll.update_one({"_id": user_id}, {"$set": {"is_admin": req.is_admin}})
    return {"message": "User role updated successfully"}


@router.put("/admin/users/{user_id}/password")
def admin_set_user_password(
    user_id: str,
    req: AdminSetPasswordRequest,
    current_user: CurrentUser = Depends(get_current_admin),
):
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password too short (min 6 chars)")
    users_coll = get_users_collection()
    target = users_coll.find_one({"_id": user_id})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if _is_root_admin_doc(target) and not _is_root_admin_user(current_user):
        raise HTTPException(status_code=403, detail="Root admin password cannot be changed by other admin accounts")
    users_coll.update_one({"_id": user_id}, {"$set": {"password_hash": hash_password(req.new_password)}})
    return {"message": "User password updated successfully"}


@router.delete("/admin/users/{user_id}")
def delete_user(
    user_id: str,
    current_user: CurrentUser = Depends(get_current_admin),
):
    if str(current_user.id) == str(user_id):
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    users_coll = get_users_collection()
    target = users_coll.find_one({"_id": user_id})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if _is_root_admin_doc(target) and not _is_root_admin_user(current_user):
        raise HTTPException(status_code=403, detail="Root admin account cannot be deleted by other admin accounts")
    res = users_coll.delete_one({"_id": user_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    # Cascade: delete sessions will cascade to messages via FK
    get_chat_sessions_collection().find({"user_id": user_id})  # sessions deleted by FK CASCADE
    return {"message": "User deleted successfully"}
