"""Heuristic intent classifier — used as fallback when the LLM supervisor fails.

Renamed from ``router.py`` to avoid confusion with FastAPI routers.

Emits the same schema as supervisor.get_intent_from_supervisor:
  {"collections_to_search": ["legal" | "drug_info"], "price": bool, "price_name": str | None}
"""

import re
from typing import Any

from drug_price_tool import detect_price_query


# Keywords that strongly indicate a legal / regulatory question.
_LEGAL_KEYWORDS = re.compile(
    r"""
    luật | nghị\s*định | thông\s*tư | quyết\s*định | chỉ\s*thị |
    quy\s*định | quy\s*chế | tiêu\s*chuẩn | điều\s*kiện\s*kinh\s*doanh |
    cấp\s*phép | giấy\s*phép | gmp | gdp | gsp | gpp | gcp |
    đăng\s*ký\s*thuốc | hành\s*chính | pháp\s*lý | pháp\s*luật |
    vi\s*phạm | xử\s*phạt | kiểm\s*tra | thanh\s*tra |
    law | regulation | decree | circular | permit | license |
    compliance | legal | regulatory
    """,
    re.IGNORECASE | re.VERBOSE,
)


def classify_intent(question: str) -> dict[str, Any]:
    """Classify intent using keyword heuristics.

    Returns:
        {"collections_to_search": ["legal"|"drug_info"], "price": bool, "price_name": str|None}
    """
    q = question.strip()

    # Price detection (unchanged logic)
    price, price_name = detect_price_query(q)

    # Collection routing
    if _LEGAL_KEYWORDS.search(q):
        collections = ["legal"]
    else:
        collections = ["drug_info"]

    return {
        "collections_to_search": collections,
        "price": price,
        "price_name": price_name if price_name else None,
    }
