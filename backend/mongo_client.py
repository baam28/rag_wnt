"""MongoDB singleton with double-checked locking to prevent race conditions.

In a multi-threaded server (uvicorn with thread-pool), two concurrent requests
could both observe ``_client is None`` and create duplicate MongoClient
instances.  The double-checked locking pattern prevents this without adding
lock overhead on every call after initialisation.
"""

import threading
from typing import Any

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
