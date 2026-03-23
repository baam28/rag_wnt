"""Dataset loading utilities for RAGAS evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from config import get_settings
except ImportError:  # pragma: no cover
    from backend.config import get_settings


REQUIRED_FIELDS = {"question", "ground_truth", "collections"}


def _validate_sample(sample: dict[str, Any], index: int) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in sample]
    if missing:
        raise ValueError(f"Sample #{index} missing required fields: {', '.join(missing)}")
    if not isinstance(sample.get("collections"), list) or not sample["collections"]:
        raise ValueError(f"Sample #{index} must have non-empty 'collections' list")


def load_curated_jsonl(path: str | Path, max_samples: int = 0) -> list[dict[str, Any]]:
    """Load curated evaluation samples from JSONL."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    samples: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            payload = line.strip()
            if not payload:
                continue
            sample = json.loads(payload)
            _validate_sample(sample, line_no)
            sample.setdefault("id", f"curated_{line_no}")
            sample.setdefault("metadata", {})
            sample["metadata"]["dataset_type"] = "curated"
            samples.append(sample)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def _build_synthetic_from_collection(collection_name: str, max_samples: int) -> list[dict[str, Any]]:
    settings = get_settings()
    parents_path = settings.persist_dir / f"{collection_name}_parents.json"
    if not parents_path.exists():
        return []

    try:
        with parents_path.open("r", encoding="utf-8") as file_obj:
            parents = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return []

    samples: list[dict[str, Any]] = []
    for index, (parent_id, payload) in enumerate(parents.items(), start=1):
        target_question = str(payload.get("target_question") or "").strip()
        if not target_question:
            continue
        summary = str(payload.get("summary") or "").strip()
        content = str(payload.get("content") or "").strip()
        source = str(payload.get("source") or "Unknown")
        ground_truth = summary or content[:400]
        if not ground_truth:
            continue
        samples.append(
            {
                "id": f"synthetic_{collection_name}_{parent_id}",
                "question": target_question,
                "ground_truth": ground_truth,
                "collections": [collection_name],
                "metadata": {
                    "dataset_type": "synthetic",
                    "source": source,
                    "parent_id": parent_id,
                },
            }
        )
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def load_synthetic_samples(max_per_collection: int = 40) -> list[dict[str, Any]]:
    """Build synthetic samples from persisted parent chunks."""
    settings = get_settings()
    collections = [settings.legal_collection_name, settings.drug_collection_name]
    out: list[dict[str, Any]] = []
    for collection in collections:
        out.extend(_build_synthetic_from_collection(collection, max_per_collection))
    return out


def load_eval_samples(
    curated_path: str | Path,
    *,
    include_synthetic: bool = False,
    synthetic_max_per_collection: int = 40,
    max_samples: int = 0,
) -> list[dict[str, Any]]:
    """Load and merge curated + optional synthetic samples."""
    curated = load_curated_jsonl(curated_path)
    merged = list(curated)
    if include_synthetic:
        merged.extend(load_synthetic_samples(max_per_collection=synthetic_max_per_collection))

    if max_samples and len(merged) > max_samples:
        merged = merged[:max_samples]
    return merged
