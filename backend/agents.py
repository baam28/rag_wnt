"""Domain agents: price scraper + two RAG agents (legal and drug).

Intent routing
--------------
The supervisor emits {"collections_to_search": ["legal", "drug"], "price": bool, "price_name": str|None}.
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

from drug_price_tool import execute_drug_sql_query
from retriever import retrieve, reformulate_with_history

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL Database Agent Wrapper
# ---------------------------------------------------------------------------

def run_price_agent(
    question: str,
    intent: dict[str, Any],
) -> Tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Invoke SQL semantic agent if intent indicates an inventory query.
    Returns (None, price_context). We no longer emit raw price_data for the frontend widget.
    """
    if not intent.get("price"):
        return None, None
        
    try:
        sql_answer = execute_drug_sql_query(question)
        if sql_answer and "Xin lỗi" not in sql_answer:
            price_ctx = {
                "content": f"KẾT QUẢ TỪ DATABASE ERP/KHO:\n{sql_answer}",
                "source": "Cơ sở dữ liệu Thuốc Nội Bộ",
                "summary": "Truy vấn dữ liệu thuốc/tồn kho",
                "collection_name": None,
                "rank": 0,
                "page": None,
            }
            return None, price_ctx
    except Exception:
        logger.warning(
            "SQL agent failed for query '%s'.",
            question,
            exc_info=True,
        )
    return None, None


def run_federated_rag_agent(
    question: str,
    collections_to_search: list[str],
    history: Optional[List[Dict[str, str]]] = None,
    history_summary: Optional[str] = None,
    pharma_only: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieve from multiple collections in parallel. Use query reformulation if history provided."""
    context_pack = {"latest_summary": history_summary} if history_summary else None
    retrieval_query = reformulate_with_history(question, history or [], context_pack=context_pack)
    try:
        return retrieve(
            retrieval_query,
            collections_to_search=collections_to_search,
            pharma_only=pharma_only,
        )
    except Exception:
        logger.error(
            "Federated RAG agent failed for collections=%s.",
            collections_to_search,
            exc_info=True,
        )
        return []
