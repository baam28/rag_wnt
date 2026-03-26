"""MongoDB singleton with double-checked locking to prevent race conditions.

In a multi-threaded server (uvicorn with thread-pool), two concurrent requests
could both observe ``_client is None`` and create duplicate MongoClient
instances.  The double-checked locking pattern prevents this without adding
lock overhead on every call after initialisation.
"""

import threading
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING
from pymongo import MongoClient
from pymongo.collection import Collection

from config import get_settings


_client: MongoClient | None = None
_client_lock = threading.Lock()


def get_mongo_client() -> MongoClient:
    """Return a thread-safe singleton MongoClient configured from settings."""
    global _client
    if _client is None:                    # fast path — no lock acquired
        with _client_lock:
            if _client is None:            # double-checked locking
                _client = MongoClient(get_settings().mongo_uri)
    return _client


def get_db():
    """Return the main MongoDB database for this app."""
    settings = get_settings()
    return get_mongo_client()[settings.mongo_db_name]


def get_users_collection() -> Collection[Any]:
    return get_db()["users"]


def get_chat_sessions_collection() -> Collection[Any]:
    return get_db()["chat_sessions"]


def get_chat_messages_collection() -> Collection[Any]:
    return get_db()["chat_messages"]


def get_feedback_collection() -> Collection[Any]:
    return get_db()["feedback"]


def get_llm_usage_collection() -> Collection[Any]:
    return get_db()["llm_usage_daily"]


def get_vector_parents_collection() -> Collection[Any]:
    coll = get_db()["vector_parents"]
    coll.create_index([("collection_name", ASCENDING), ("parent_id", ASCENDING)], unique=True)
    coll.create_index([("collection_name", ASCENDING), ("source", ASCENDING)])
    return coll


def get_sparse_vocab_collection() -> Collection[Any]:
    coll = get_db()["vector_sparse_vocab"]
    coll.create_index([("collection_name", ASCENDING), ("token", ASCENDING)], unique=True)
    coll.create_index([("collection_name", ASCENDING), ("idx", ASCENDING)])
    return coll


def load_vector_parents(collection_name: str) -> dict[str, dict[str, Any]]:
    coll = get_vector_parents_collection()
    cursor = coll.find(
        {"collection_name": collection_name},
        {
            "_id": 0,
            "parent_id": 1,
            "content": 1,
            "summary": 1,
            "target_question": 1,
            "source": 1,
        },
    )
    out: dict[str, dict[str, Any]] = {}
    for doc in cursor:
        parent_id = str(doc.get("parent_id") or "").strip()
        if not parent_id:
            continue
        out[parent_id] = {
            "content": doc.get("content", ""),
            "summary": doc.get("summary", ""),
            "target_question": doc.get("target_question", ""),
            "source": doc.get("source", "Unknown"),
        }
    return out


def upsert_vector_parents(collection_name: str, parent_meta: dict[str, dict[str, Any]]) -> None:
    if not parent_meta:
        return
    coll = get_vector_parents_collection()
    now = datetime.now(timezone.utc)
    for parent_id, meta in parent_meta.items():
        coll.update_one(
            {"collection_name": collection_name, "parent_id": parent_id},
            {
                "$set": {
                    "content": meta.get("content", ""),
                    "summary": meta.get("summary", ""),
                    "target_question": meta.get("target_question", ""),
                    "source": meta.get("source", "Unknown"),
                    "updated_at": now,
                }
            },
            upsert=True,
        )


def count_vector_parents_by_source(collection_name: str) -> dict[str, int]:
    coll = get_vector_parents_collection()
    pipeline = [
        {"$match": {"collection_name": collection_name}},
        {"$group": {"_id": "$source", "count": {"$sum": 1}}},
    ]
    out: dict[str, int] = {}
    for row in coll.aggregate(pipeline):
        source = str(row.get("_id") or "Unknown")
        out[source] = int(row.get("count", 0))
    return out


def delete_vector_parents_by_source(collection_name: str, source: str) -> None:
    coll = get_vector_parents_collection()
    coll.delete_many({"collection_name": collection_name, "source": source})


def load_sparse_vocab_map(collection_name: str) -> dict[str, int]:
    coll = get_sparse_vocab_collection()
    cursor = coll.find(
        {"collection_name": collection_name},
        {"_id": 0, "token": 1, "idx": 1},
    )
    out: dict[str, int] = {}
    for doc in cursor:
        token = str(doc.get("token") or "")
        if not token:
            continue
        try:
            out[token] = int(doc.get("idx"))
        except Exception:
            continue
    return out


def insert_sparse_vocab_entries(collection_name: str, token_to_idx: dict[str, int]) -> None:
    if not token_to_idx:
        return
    coll = get_sparse_vocab_collection()
    now = datetime.now(timezone.utc)
    docs = [
        {
            "collection_name": collection_name,
            "token": token,
            "idx": int(idx),
            "updated_at": now,
        }
        for token, idx in token_to_idx.items()
    ]
    try:
        coll.insert_many(docs, ordered=False)
    except Exception:
        for doc in docs:
            coll.update_one(
                {"collection_name": collection_name, "token": doc["token"]},
                {"$setOnInsert": doc},
                upsert=True,
            )


def delete_collection_vector_meta(collection_name: str) -> None:
    get_vector_parents_collection().delete_many({"collection_name": collection_name})
    get_sparse_vocab_collection().delete_many({"collection_name": collection_name})
