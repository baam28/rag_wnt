"""Shared utilities: functions that are needed by both ingest.py and retriever.py.

Keeping them here eliminates the duplicated definitions in both modules.
"""

import re
from pathlib import Path

from qdrant_client import QdrantClient


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


def get_qdrant_client(persist_dir: Path) -> QdrantClient:
    """Connect to local Qdrant (embedded) under persist_dir."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    path = persist_dir / "storage"
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def tokenize_for_sparse(text: str) -> list[str]:
    """Tokenize text for sparse vector (Vietnamese and English BM25-style)."""
    text = text.lower().strip()
    return re.findall(r"\b\w+\b", text)
