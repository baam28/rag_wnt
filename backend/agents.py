"""Domain agents: ERP/inventory SQL agent + two RAG agents (legal and drug).

Intent routing
--------------
The supervisor emits {"collections_to_search": ["legal", "drug"], "erp": bool, "erp_drug_name": str|None}.
  - run_erp_agent           → always called when intent["erp"] is True
  - run_federated_rag_agent → searches across all provided collections simultaneously

History-aware retrieval
-----------------------
Both RAG agents accept an optional ``history`` list of {"role", "content"} dicts.
When provided, the question is first reformulated into a standalone query via
``reformulate_with_history()`` so that follow-up questions resolve correctly.
"""

import logging
from typing import Any, Optional, List, Dict

from drug_price_tool import execute_erp_query, execute_drug_info_query
from retriever import retrieve, reformulate_with_history

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL Database Agent Wrapper
# ---------------------------------------------------------------------------

def run_erp_agent(
    question: str,
    intent: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Invoke the ERP SQL agent for any inventory query (price, stock, expiry, packaging).
    Returns an erp_context dict on success, or None.
    """
    if not intent.get("erp"):
        return None

    try:
        sql_answer = execute_erp_query(question)
        if sql_answer and "Xin lỗi" not in sql_answer:
            return {
                "content": f"KẾT QUẢ TỪ DATABASE ERP/KHO:\n{sql_answer}",
                "source": "Cơ sở dữ liệu Thuốc Nội Bộ",
                "summary": "Truy vấn dữ liệu ERP/kho thuốc",
                "collection_name": None,
                "rank": 0,
                "page": None,
            }
    except Exception:
        logger.warning(
            "ERP SQL agent failed for query '%s'.",
            question,
            exc_info=True,
        )
    return None


def run_drug_db_agent(
    question: str,
    collections: list[str],
) -> Optional[dict[str, Any]]:
    """Query drug_list for structured clinical info when the drug collection is being searched.

    Returns a context dict to merge into final_contexts, or None if no results / not applicable.
    Skipped automatically when "drug" is not in collections.
    """
    if "drug" not in collections:
        return None

    try:
        result = execute_drug_info_query(question)
        if result and len(result.strip()) > 20:
            return {
                "content": f"THÔNG TIN THUỐC TỪ DATABASE NỘI BỘ:\n{result}",
                "source": "Cơ sở dữ liệu Thuốc",
                "collection_name": None,
                "rank": 0,
                "page": None,
            }
    except Exception:
        logger.warning("Drug DB agent failed for query '%s'.", question, exc_info=True)
    return None


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
            history=history or [],
            pharma_only=pharma_only,
        )
    except Exception:
        logger.error(
            "Federated RAG agent failed for collections=%s.",
            collections_to_search,
            exc_info=True,
        )
        return []
