"""Admin router: collections, docs, users, analytics, feedback endpoints."""

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bson.objectid import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from qdrant_client.http import models as qmodels

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
from mongo_client import (
    get_users_collection,
    get_chat_sessions_collection,
    get_chat_messages_collection,
)
from utils import get_qdrant_client
from llm_usage import get_usage

router = APIRouter(tags=["admin"])

# ---------------------------------------------------------------------------
# Feedback helpers (file-backed store)
# ---------------------------------------------------------------------------

_FEEDBACK_FILE = Path(__file__).resolve().parent.parent.parent / "feedback.json"
_feedback_lock = threading.Lock()


def _load_feedback() -> list[dict]:
    if _FEEDBACK_FILE.exists():
        try:
            with open(_FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_feedback(data: list[dict]):
    with open(_FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
        "question": req.question[:500],
        "answer": req.answer,
        "rating": req.rating,
        "comment": (req.comment or "")[:500],
        "session_id": req.session_id or "",
    }
    with _feedback_lock:
        data = _load_feedback()
        data.append(entry)
        _save_feedback(data)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Admin-only endpoints
# ---------------------------------------------------------------------------

@router.get("/admin/feedback")
def get_feedback(current_user: CurrentUser = Depends(get_current_admin)):
    data = _load_feedback()
    up = sum(1 for d in data if d.get("rating") == "up")
    down = sum(1 for d in data if d.get("rating") == "down")
    return {
        "total": len(data),
        "up": up,
        "down": down,
        "down_entries": list(reversed([d for d in data if d.get("rating") == "down"]))[:100],
        "all_entries": list(reversed(data))[:200],
    }


@router.get("/admin/analytics")
def get_analytics(current_user: CurrentUser = Depends(get_current_admin)):
    total_users = get_users_collection().count_documents({})
    total_sessions = get_chat_sessions_collection().count_documents({})
    total_messages = get_chat_messages_collection().count_documents({})
    fb_data = _load_feedback()

    date_list = _last_n_dates(30)

    # Users per day
    try:
        raw = list(
            get_users_collection().aggregate([
                {"$project": {"day": {"$substr": ["$created_at", 0, 10]}}},
                {"$group": {"_id": "$day", "count": {"$sum": 1}}},
            ])
        )
        users_per_day = _fill_series_by_date(date_list, [{"date": d["_id"], "count": d["count"]} for d in raw])
    except Exception:
        users_per_day = [{"date": d, "count": 0} for d in date_list]

    # Sessions (conversations) per day
    try:
        raw = list(
            get_chat_sessions_collection().aggregate([
                {"$project": {"day": {"$substr": ["$created_at", 0, 10]}}},
                {"$group": {"_id": "$day", "count": {"$sum": 1}}},
            ])
        )
        sessions_per_day = _fill_series_by_date(date_list, [{"date": d["_id"], "count": d["count"]} for d in raw])
    except Exception:
        sessions_per_day = [{"date": d, "count": 0} for d in date_list]

    # Messages per day (assistant messages)
    try:
        raw = list(
            get_chat_messages_collection().aggregate([
                {"$match": {"role": "assistant"}},
                {"$project": {"day": {"$substr": ["$created_at", 0, 10]}}},
                {"$group": {"_id": "$day", "count": {"$sum": 1}}},
            ])
        )
        messages_per_day = _fill_series_by_date(date_list, [{"date": d["_id"], "count": d["count"]} for d in raw])
    except Exception:
        messages_per_day = [{"date": d, "count": 0} for d in date_list]

    # Feedback per day (from timestamp "YYYY-MM-DD HH:MM:SS")
    fb_by_date: dict[str, int] = {}
    for entry in fb_data:
        ts = entry.get("timestamp", "")
        day = ts[:10] if len(ts) >= 10 else ""
        if day:
            fb_by_date[day] = fb_by_date.get(day, 0) + 1
    feedback_per_day = [{"date": d, "count": fb_by_date.get(d, 0)} for d in date_list]

    # LLM API usage (total + daily, 30 days, normalized to same date range)
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
def list_collections(current_user: CurrentUser = Depends(get_current_admin)):
    settings = get_settings()
    client = get_qdrant_client(settings.persist_dir)
    try:
        resp = client.get_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return [CollectionInfo(name=c.name) for c in resp.collections]


@router.get("/admin/docs", response_model=list[DocumentInfo])
def list_documents(
    collection_name: str = Query(..., alias="collection_name"),
    current_user: CurrentUser = Depends(get_current_admin),
):
    settings = get_settings()
    parents_path = settings.persist_dir / f"{collection_name}_parents.json"
    if not parents_path.exists():
        return []
    try:
        with open(parents_path, "r", encoding="utf-8") as f:
            parents = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    counts: dict[str, int] = {}
    for meta in parents.values():
        src = meta.get("source", "Unknown")
        counts[src] = counts.get(src, 0) + 1
    return [DocumentInfo(source=s, parent_count=n) for s, n in counts.items()]


@router.delete("/admin/docs")
def delete_document(
    req: DeleteDocumentRequest,
    current_user: CurrentUser = Depends(get_current_admin),
):
    settings = get_settings()
    client = get_qdrant_client(settings.persist_dir)

    try:
        next_offset = None
        all_ids: list[int] = []
        while True:
            scroll_result, next_offset = client.scroll(
                collection_name=req.collection_name,
                limit=1000,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source",
                            match=qmodels.MatchValue(value=req.source),
                        )
                    ]
                ),
            )
            if not scroll_result:
                break
            all_ids.extend(p.id for p in scroll_result)
            if next_offset is None:
                break
        if all_ids:
            client.delete(
                collection_name=req.collection_name,
                points_selector=qmodels.PointIdsList(points=all_ids),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete points: {e}")

    parents_path = settings.persist_dir / f"{req.collection_name}_parents.json"
    if parents_path.exists():
        try:
            with open(parents_path, "r", encoding="utf-8") as f:
                parents = json.load(f)
            parents = {pid: meta for pid, meta in parents.items() if meta.get("source") != req.source}
            with open(parents_path, "w", encoding="utf-8") as f:
                json.dump(parents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update parents: {e}")

    return {"message": f"Deleted document '{req.source}' from collection '{req.collection_name}'."}


@router.delete("/admin/collections/{collection_name}")
def delete_collection(
    collection_name: str,
    current_user: CurrentUser = Depends(get_current_admin),
):
    settings = get_settings()
    client = get_qdrant_client(settings.persist_dir)
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    for suffix in ("_parents.json", "_sparse_vocab.json"):
        p = settings.persist_dir / f"{collection_name}{suffix}"
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    return {"message": f"Deleted collection '{collection_name}' and related metadata."}


@router.post("/db/clear")
def clear_db(current_user: CurrentUser = Depends(get_current_admin)):
    """Clear the entire Qdrant database directory."""
    import shutil
    settings = get_settings()
    db_path = Path(settings.persist_dir)
    try:
        if db_path.exists():
            shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"Cleared database at '{db_path}'. You can ingest again."}


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

@router.get("/admin/users", response_model=list[UserOut])
def get_all_users(current_user: CurrentUser = Depends(get_current_admin)):
    docs = get_users_collection().find().sort("created_at", -1)
    return [UserOut(id=str(d["_id"]), username=d.get("email", ""), is_admin=bool(d.get("is_admin", False))) for d in docs]


@router.put("/admin/users/{user_id}/role")
def update_user_role(
    user_id: str,
    req: RoleUpdateRequest,
    current_user: CurrentUser = Depends(get_current_admin),
):
    if str(current_user.id) == str(user_id):
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    users_coll = get_users_collection()
    target = users_coll.find_one({"_id": ObjectId(user_id)})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.get("email") == "admin":
        raise HTTPException(status_code=400, detail="Cannot change role of root admin")
    users_coll.update_one({"_id": ObjectId(user_id)}, {"$set": {"is_admin": req.is_admin}})
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
    res = users_coll.update_one({"_id": ObjectId(user_id)}, {"$set": {"password_hash": hash_password(req.new_password)}})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User password updated successfully"}


@router.delete("/admin/users/{user_id}")
def delete_user(
    user_id: str,
    current_user: CurrentUser = Depends(get_current_admin),
):
    if str(current_user.id) == str(user_id):
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    users_coll = get_users_collection()
    target = users_coll.find_one({"_id": ObjectId(user_id)})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.get("email") == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete root admin")
    res = users_coll.delete_one({"_id": ObjectId(user_id)})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    get_chat_sessions_collection().delete_many({"user_id": user_id})
    get_chat_messages_collection().delete_many({"user_id": user_id})
    return {"message": "User deleted successfully"}
