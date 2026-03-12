"""LLM-based supervisor: classifies user intent and selects which collection to query.

New intent schema
-----------------
{
  "collections_to_search": ["legal", "drug_info"], # which RAG agents/collections to use
  "price":      true | false,                      # also run the price-scraper agent
  "price_name": "<drug name>" | null               # drug name for price lookup
}

Falls back to heuristic classify_intent from router when the LLM call fails.
"""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from config import get_settings
from classifier import classify_intent

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM = """Bạn là bộ phân loại ý định (intent) cho hệ thống hỏi đáp y tế – pháp lý.
Hệ thống có 2 kho tài liệu RAG:
  1. **drug_info** – Thông tin dược phẩm/hoạt chất: cơ chế, tác dụng, liều dùng, chống chỉ định, tác dụng phụ, tương tác thuốc, cách bảo quản, v.v.
  2. **legal**     – Văn bản pháp lý: luật dược, nghị định, thông tư, quy định, tiêu chuẩn, thủ tục hành chính liên quan đến dược phẩm.

Ngoài ra có agent tra cứu giá (**price**) hoạt động độc lập với RAG.

Quy tắc phân loại:
- Cho phép trả về NHIỀU collection nếu câu hỏi yêu cầu cả hai lĩnh vực (ví dụ: vừa hỏi pháp lý vừa hỏi tác dụng thuốc).
- Chọn **drug_info** khi câu hỏi liên quan đến thông tin dược lý/clinical của thuốc hoặc hoạt chất.
- Chọn **legal** khi câu hỏi hỏi về luật, nghị định, thông tư, quy định, điều kiện kinh doanh, cấp phép, v.v.
- Khi câu hỏi hỏi giá thuốc, set "price": true và "price_name" là tên thuốc/hoạt chất.
- Khi không hỏi giá, set "price": false và "price_name": null.

Trả lời ĐÚNG THEO format JSON sau (không thêm giải thích ngoài JSON):
{"collections_to_search": ["drug_info", "legal"], "price": true/false, "price_name": "tên thuốc hoặc null"}"""


def _parse_supervisor_response(text: str) -> dict[str, Any] | None:
    """Extract and validate JSON intent from LLM response. Returns None on failure."""
    if not text or not text.strip():
        return None
    text = text.strip()
    # Try full text first, then extract first {...} block
    for candidate in [text, None]:
        if candidate is None:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end <= start:
                return None
            candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            if candidate is text:
                continue
            return None
    else:
        return None

    if not isinstance(data, dict):
        return None

    # collections_to_search field — must be a list containing "legal", "drug_info", or both
    collections = data.get("collections_to_search", ["drug_info"])
    if not isinstance(collections, list) or not collections:
        collections = ["drug_info"]
    
    valid_collections = [c for c in collections if c in ("legal", "drug_info")]
    if not valid_collections:
        valid_collections = ["drug_info"]

    # price fields
    price = bool(data.get("price"))
    price_name = data.get("price_name")
    if price_name is not None and not isinstance(price_name, str):
        price_name = None
    if price_name is not None:
        price_name = str(price_name).strip() or None

    return {
        "collections_to_search": valid_collections,
        "price": price,
        "price_name": price_name,
    }


def get_intent_from_supervisor(question: str) -> dict[str, Any]:
    """Classify intent using the LLM supervisor. Falls back to heuristic router on failure."""
    settings = get_settings()
    if not settings.openai_api_key or not question.strip():
        return classify_intent(question)

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
    )
    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM},
        {"role": "user", "content": f"Câu hỏi: {question.strip()}\n\nTrả lời bằng JSON theo đúng format đã nêu."},
    ]
    try:
        resp = llm.invoke(messages)
        content = resp.content if hasattr(resp, "content") else str(resp)
        intent = _parse_supervisor_response(content)
        if intent is not None:
            return intent
    except Exception:
        logger.warning("Supervisor LLM call failed. Falling back to heuristic router.", exc_info=True)
    return classify_intent(question)
