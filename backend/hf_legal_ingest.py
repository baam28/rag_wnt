"""CLI utility for Hugging Face legal dataset preview and ingestion."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata

from ingest import ingest_hf_legal_dataset

_VALID_LEGAL_TYPES = {"Luật", "Nghị định", "Thông tư", "Văn bản hợp nhất"}
_HEALTH_AUTHORITIES = ("Bộ Y tế", "Quốc hội", "Chính phủ")
_HEALTH_TITLE_KEYWORDS = ("Dược", "Thuốc")
_VBHN_TITLE_KEYWORDS = ("duoc", "thuoc")


def _normalize_text_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFD", (text or "").lower())
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip()


def _vbhn_pharma_title_filter(meta: dict) -> bool:
    """Filter rows where legal_type is 'Văn bản hợp nhất' and title relates to thuốc/dược."""
    legal_type = _normalize_text_for_match(meta.get("legal_type", ""))
    if legal_type != "van ban hop nhat":
        return False

    title = _normalize_text_for_match(meta.get("title", ""))
    return any(keyword in title for keyword in _VBHN_TITLE_KEYWORDS)


def _health_legal_filter(meta: dict) -> bool:
    """Implements the SQL WHERE clause:
      (UPPER(legal_sectors) LIKE '%Y TẾ%'
       OR ((issuing_authority LIKE '%Bộ Y tế%' OR '%Quốc hội%' OR '%Chính phủ%')
           AND (title LIKE '%Dược%' OR title LIKE '%Thuốc%'))
      )
      AND legal_type IN ('Luật', 'Nghị định', 'Thông tư', 'Văn bản hợp nhất')
    """
    legal_type = meta.get("legal_type", "")
    if legal_type not in _VALID_LEGAL_TYPES:
        return False

    sectors = meta.get("legal_sectors", "").upper()
    if "Y TẾ" in sectors:
        return True

    authority = meta.get("issuing_authority", "")
    title = meta.get("title", "")
    authority_match = any(a in authority for a in _HEALTH_AUTHORITIES)
    title_match = any(k in title for k in _HEALTH_TITLE_KEYWORDS)
    return authority_match and title_match


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Hugging Face legal dataset into Qdrant.")
    parser.add_argument("--collection", default="legal", help="Target Qdrant collection name")
    parser.add_argument("--only-pharma", action="store_true", help="Ingest only pharmaceutical-related legal docs (legacy keyword filter)")
    parser.add_argument(
        "--filter",
        choices=["pharma", "health_legal", "vbhn_pharma_title", "proxy_strict"],
        default=None,
        help=(
            "Filter mode: 'pharma' = keyword-based sector match; "
            "'health_legal' = SQL-equivalent filter on sectors/authority/title + top legal types; "
            "'vbhn_pharma_title' = legal_type='Văn bản hợp nhất' and title contains thuốc/dược; "
            "'proxy_strict' = whitelist legal type + dedupe latest version by issuance_date"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Max docs to ingest")
    parser.add_argument("--start-offset", type=int, default=0, help="Start offset for dataset iteration")
    parser.add_argument("--batch-size", type=int, default=None, help="Docs per ingest batch")
    parser.add_argument(
        "--write-quarantine",
        choices=["true", "false"],
        default="false",
        help="When true, rejected proxy rows are ingested into quarantine collection",
    )
    parser.add_argument(
        "--quarantine-collection",
        default=None,
        help="Optional quarantine collection name (default: <collection>_quarantine)",
    )
    parser.add_argument(
        "--skip-summary",
        choices=["true", "false"],
        default="true",
        help="Skip LLM summary generation for speed (default: true)",
    )
    args = parser.parse_args()

    skip_summary = args.skip_summary.lower() == "true"
    write_quarantine = args.write_quarantine.lower() == "true"

    # Determine filter mode
    row_filter = None
    only_pharma = args.only_pharma
    mode = "default"
    if args.filter == "health_legal":
        row_filter = _health_legal_filter
        only_pharma = False
    elif args.filter == "vbhn_pharma_title":
        row_filter = _vbhn_pharma_title_filter
        only_pharma = False
    elif args.filter == "pharma":
        only_pharma = True
    elif args.filter == "proxy_strict":
        mode = "proxy_strict"
        only_pharma = True

    result = ingest_hf_legal_dataset(
        collection_name=args.collection,
        only_pharma_related=only_pharma,
        row_filter=row_filter,
        mode=mode,
        write_quarantine=write_quarantine,
        quarantine_collection_name=args.quarantine_collection,
        limit=args.limit,
        start_offset=args.start_offset,
        batch_size=args.batch_size,
        skip_summary=skip_summary,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
