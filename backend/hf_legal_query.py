"""CLI utility to query pharmaceutical-related legal metadata from Hugging Face dataset."""

from __future__ import annotations

import argparse
import json

from config import get_settings


def _normalize_text_for_match(text: str) -> str:
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFD", (text or "").lower())
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip()


def _extract_pharma_tags(legal_sectors: str, keywords: list[str]) -> list[str]:
    sectors = [part.strip() for part in (legal_sectors or "").split("|") if part.strip()]
    normalized_keywords = [_normalize_text_for_match(k) for k in keywords if (k or "").strip()]
    tags: list[str] = []
    for sector in sectors:
        normalized_sector = _normalize_text_for_match(sector)
        if any(k and (k in normalized_sector or normalized_sector in k) for k in normalized_keywords):
            tags.append(sector)
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Query pharma-related legal docs from HF metadata.")
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to output")
    parser.add_argument("--offset", type=int, default=0, help="Offset in filtered results")
    args = parser.parse_args()

    settings = get_settings()
    from datasets import load_dataset

    ds = load_dataset(
        settings.hf_legal_dataset_repo,
        settings.hf_legal_metadata_config,
        split=settings.hf_legal_split,
    )

    matched = []
    for row in ds:
        legal_sectors = str(row.get("legal_sectors") or "")
        pharma_tags = _extract_pharma_tags(legal_sectors, settings.hf_pharma_sector_keywords)
        if not pharma_tags:
            continue
        matched.append(
            {
                "id": row.get("id"),
                "document_number": row.get("document_number"),
                "title": row.get("title"),
                "legal_type": row.get("legal_type"),
                "legal_sectors": legal_sectors,
                "issuing_authority": row.get("issuing_authority"),
                "issuance_date": row.get("issuance_date"),
                "url": row.get("url"),
                "pharma_tags": pharma_tags,
            }
        )

    total = len(matched)
    start = min(args.offset, total)
    end = min(start + args.limit, total)

    result = {
        "dataset": settings.hf_legal_dataset_repo,
        "total_pharma_related": total,
        "offset": start,
        "limit": args.limit,
        "items": matched[start:end],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
