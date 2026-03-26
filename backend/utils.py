"""Shared utilities: functions that are needed by both ingest.py and retriever.py.

Keeping them here eliminates the duplicated definitions in both modules.
"""

import re
import threading

from qdrant_client import QdrantClient

from config import get_settings


def fix_position_ids(model) -> None:
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


_QDRANT_CLIENTS: dict[str, QdrantClient] = {}
_QDRANT_CLIENTS_LOCK = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    """Return a process-wide cached HTTP Qdrant client (Docker/remote)."""
    settings = get_settings()
    key = f"http:{settings.qdrant_url}:{settings.qdrant_api_key or ''}"
    with _QDRANT_CLIENTS_LOCK:
        client = _QDRANT_CLIENTS.get(key)
        if client is None:
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=(settings.qdrant_api_key or None),
                timeout=30.0,
            )
            _QDRANT_CLIENTS[key] = client
        return client


def reset_qdrant_client() -> None:
    """Close and evict cached HTTP Qdrant client (if any)."""
    settings = get_settings()
    key = f"http:{settings.qdrant_url}:{settings.qdrant_api_key or ''}"

    with _QDRANT_CLIENTS_LOCK:
        client = _QDRANT_CLIENTS.pop(key, None)
    if client and hasattr(client, "close"):
        try:
            client.close()
        except Exception:
            pass


def tokenize_for_sparse(text: str) -> list[str]:
    """Tokenize text for sparse vector (Vietnamese and English BM25-style)."""
    text = text.lower().strip()
    return re.findall(r"\b\w+\b", text)
