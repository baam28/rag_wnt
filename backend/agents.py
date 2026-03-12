"""Domain agents: price scraper + two RAG agents (legal and drug_info).

Intent routing
--------------
The supervisor emits {"collections_to_search": ["legal", "drug_info"], "price": bool, "price_name": str|None}.
  - run_price_agent         → always called when intent["price"] is True
  - run_federated_rag_agent → searches across all provided collections simultaneously

History-aware retrieval
-----------------------
Both RAG agents accept an optional ``history`` list of {"role", "content"} dicts.
When provided, the question is first reformulated into a standalone query via
``reformulate_with_history()`` so that follow-up questions resolve correctly.
"""

import logging
from typing import Any, Optional, Tuple, List, Dict

from drug_price_tool import get_vietnam_drug_price
from retriever import retrieve, reformulate_with_history

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price agent (external pharmacy scraper — unchanged)
# ---------------------------------------------------------------------------

def _format_price_as_context(price_data: dict[str, Any]) -> dict[str, Any]:
    """Turn structured price_data into a RAG context entry the LLM can cite."""
    drug = price_data.get("drug_name", "")
    lines = [f"KẾT QUẢ TRA CỨU GIÁ THUỐC: {drug}"]

    if price_data.get("is_prescription"):
        lines.append("Đây là thuốc kê đơn (Rx).")
        lines.append(price_data.get("notes", "Giá thuốc kê đơn không niêm yết công khai."))
    else:
        price_range = price_data.get("price_range", "")
        if price_range:
            lines.append(f"Khoảng giá: {price_range}")
        drugs = price_data.get("drugs", [])
        if drugs:
            lines.append(
                f"Có {len(drugs)} loại thuốc liên quan. Chi tiết từng thuốc và link xem ở bảng giá bên dưới."
            )
        else:
            for p in price_data.get("prices", []):
                name = p.get("source_name", "Nhà thuốc")
                price = p.get("price", "N/A")
                unit = p.get("unit", "")
                url = p.get("source_url", "")
                lines.append(f"  • {name}: {price}/{unit} — {url}")

    urls = price_data.get("source_urls", [])
    if urls:
        lines.append("Nguồn: " + ", ".join(urls[:3]))

    disclaimer = price_data.get("disclaimer", "")
    if disclaimer:
        lines.append(disclaimer)

    return {
        "content": "\n".join(lines),
        "source": "Tra cứu giá thuốc trực tuyến",
        "summary": price_data.get("price_range", f"Giá thuốc {drug}"),
        "collection_name": None,
        "rank": 0,
        "page": None,
    }


def run_price_agent(
    question: str,
    intent: dict[str, Any],
) -> Tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Invoke price scraper if intent indicates a price query.

    Returns (price_data, price_context) or (None, None).
    """
    if not (intent.get("price") and intent.get("price_name")):
        return None, None
    try:
        price_data = get_vietnam_drug_price(intent["price_name"], question=question)
    except Exception:
        logger.warning(
            "Price agent failed for drug '%s'.",
            intent["price_name"],
            exc_info=True,
        )
        return None, None

    if not price_data or not (
        price_data.get("prices")
        or price_data.get("price_range")
        or price_data.get("is_prescription")
    ):
        return None, None

    price_ctx = _format_price_as_context(price_data)
    return price_data, price_ctx


def run_federated_rag_agent(
    question: str,
    collections_to_search: list[str],
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve from multiple collections in parallel. Use query reformulation if history provided."""
    retrieval_query = reformulate_with_history(question, history or [])
    try:
        return retrieve(retrieval_query, collections_to_search=collections_to_search)
    except Exception:
        logger.error(
            "Federated RAG agent failed for collections=%s.",
            collections_to_search,
            exc_info=True,
        )
        return []

