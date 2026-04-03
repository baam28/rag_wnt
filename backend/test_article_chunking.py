"""
Quick smoke-test for article-based chunking.

Ingests 8 documents from the Hugging Face legal dataset into a
dedicated test collection, then inspects the Qdrant payloads to
verify that article/chapter metadata is present and the chunk
counts look reasonable.

Usage (from backend/):
    python test_article_chunking.py
"""
import json
import sys
from collections import Counter

from config import get_settings
from ingest import ingest_hf_legal_dataset, _detect_legal_articles
from utils import get_qdrant_client

TEST_COLLECTION = "article_chunk_test"
SAMPLE_DOCS = 8


def _progress(step: str, msg: str, cur: int, total: int) -> None:
    bar = f"[{cur}/{total}]" if total else ""
    print(f"  [{step}] {bar} {msg}", flush=True)


def run_test() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Ingesting {SAMPLE_DOCS} documents into '{TEST_COLLECTION}' ===\n")

    result = ingest_hf_legal_dataset(
        collection_name=TEST_COLLECTION,
        limit=SAMPLE_DOCS,
        skip_summary=True,       # faster — no LLM summary calls
        on_progress=_progress,
    )

    print("\n--- Ingestion result ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # --- Inspect Qdrant payloads ---
    print("\n=== Inspecting Qdrant payloads ===\n")
    client = get_qdrant_client()

    try:
        count = client.count(collection_name=TEST_COLLECTION, exact=True).count
    except Exception as e:
        print(f"ERROR: Could not reach Qdrant: {e}")
        return

    print(f"Total points in '{TEST_COLLECTION}': {count}")

    # Scroll up to 200 points to analyse
    points, _ = client.scroll(
        collection_name=TEST_COLLECTION,
        limit=200,
        with_payload=True,
        with_vectors=False,
    )

    strategy_counts: Counter = Counter()
    has_article_number: int = 0
    has_chapter_number: int = 0
    article_numbers: list[str] = []
    sample_payloads: list[dict] = []

    for p in points:
        payload = p.payload or {}
        strategy_counts[payload.get("chunk_strategy", "<none>")] += 1
        art_num = payload.get("article_number")
        chap_num = payload.get("chapter_number")
        if art_num not in (None, "", "0"):
            has_article_number += 1
            article_numbers.append(str(art_num))
        if chap_num not in (None, ""):
            has_chapter_number += 1
        if len(sample_payloads) < 3 and art_num not in (None, "", "0"):
            sample_payloads.append({
                "article_number": art_num,
                "article_title": payload.get("article_title", ""),
                "chapter_number": chap_num,
                "chapter_title": payload.get("chapter_title", ""),
                "text_preview": (payload.get("text") or "")[:120],
            })

    print(f"\nChunk strategy breakdown: {dict(strategy_counts)}")
    print(f"Points with article_number: {has_article_number}/{len(points)}")
    print(f"Points with chapter_number: {has_chapter_number}/{len(points)}")

    if article_numbers:
        unique_articles = sorted(set(article_numbers), key=lambda x: int(x) if x.isdigit() else 0)
        print(f"Unique article numbers seen (first 20): {unique_articles[:20]}")

    if sample_payloads:
        print("\n--- Sample article chunks ---")
        for i, sp in enumerate(sample_payloads, 1):
            print(f"\n  [{i}] Điều {sp['article_number']} — {sp['article_title']}")
            if sp["chapter_number"]:
                print(f"       Chương {sp['chapter_number']} — {sp['chapter_title']}")
            print(f"       Text: {sp['text_preview']}...")
    else:
        print("\nWARNING: No article-chunked points found. Documents may not have article structure.")

    print("\n=== Test complete ===")


if __name__ == "__main__":
    run_test()
