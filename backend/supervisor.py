"""LLM-based supervisor: classifies user intent and selects which collection to query.

New intent schema
-----------------
{
  "collections_to_search": ["legal", "drug"], # which RAG agents/collections to use
  "erp":      true | false,                    # also run the ERP/inventory SQL agent
  "erp_drug_name": "<drug name>" | null        # drug name for ERP lookup
}
"""

import json
import logging
from typing import Any

from prompts import build_runtime_chat_client

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM = """Bạn là bộ phân loại ý định (intent) cho hệ thống hỏi đáp y tế – pháp lý.
Hệ thống có 2 kho tài liệu RAG:
  1. **drug**  – Thông tin dược phẩm/hoạt chất: cơ chế, tác dụng, liều dùng, chống chỉ định, tác dụng phụ, tương tác thuốc, cách bảo quản, v.v.
  2. **legal** – Văn bản pháp lý: luật dược, nghị định, thông tư, quy định, tiêu chuẩn, thủ tục hành chính liên quan đến dược phẩm.

Ngoài ra có agent tra cứu cơ sở dữ liệu ERP/kho (**erp**) hoạt động độc lập với RAG.
Agent này có thể trả lời mọi câu hỏi liên quan đến dữ liệu thực tế trong kho: giá bán, tồn kho, hạn sử dụng, quy cách đóng gói.

Quy tắc phân loại:
- Cho phép trả về NHIỀU collection nếu câu hỏi yêu cầu cả hai lĩnh vực (ví dụ: vừa hỏi pháp lý vừa hỏi tác dụng thuốc).
- Chọn **drug** khi câu hỏi liên quan đến thông tin dược lý/clinical của thuốc hoặc hoạt chất.
- Chọn **legal** khi câu hỏi hỏi về luật, nghị định, thông tư, quy định, điều kiện kinh doanh, cấp phép, v.v.
- Set "erp": true khi câu hỏi cần tra cứu dữ liệu ERP/kho, bao gồm: giá bán, giá thuốc, tồn kho, số lượng trong kho, hạn sử dụng, quy cách đóng gói, đơn vị bán. Kèm "erp_drug_name" là tên thuốc/hoạt chất được hỏi.
- Khi không cần tra cứu ERP/kho, set "erp": false và "erp_drug_name": null.

Trả lời ĐÚNG THEO format JSON sau (không thêm giải thích ngoài JSON):
{"collections_to_search": ["drug", "legal"], "erp": true/false, "erp_drug_name": "tên thuốc hoặc null"}"""


def _parse_supervisor_response(text: str) -> dict[str, Any] | None:
    """Extract and validate JSON intent from LLM response. Returns None on failure."""
    if not text or not text.strip():
        return None
    text = text.strip()
    # Try parsing the full response first; fall back to extracting the first {...} block.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    # collections_to_search field — must be a list containing "legal", "drug", or both
    collections = data.get("collections_to_search", ["drug"])
    if not isinstance(collections, list) or not collections:
        collections = ["drug"]

    valid_collections = [c for c in collections if c in ("legal", "drug")]
    if not valid_collections:
        valid_collections = ["drug"]

    # erp fields
    erp = bool(data.get("erp"))
    erp_drug_name = data.get("erp_drug_name")
    if erp_drug_name is not None and not isinstance(erp_drug_name, str):
        erp_drug_name = None
    if erp_drug_name is not None:
        erp_drug_name = str(erp_drug_name).strip() or None

    return {
        "collections_to_search": valid_collections,
        "erp": erp,
        "erp_drug_name": erp_drug_name,
    }


def get_intent_from_supervisor(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Classify intent using the AI supervisor."""
    if not question.strip():
        return {"collections_to_search": ["drug"], "erp": False, "erp_drug_name": None}

    history_text = ""
    if history:
        recent = history[-4:]  # last 2 user+assistant turns
        lines = []
        for m in recent:
            content = (m.get("content") or "").strip()[:300]
            if not content:
                continue
            role_label = "Người dùng" if m.get("role") == "user" else "Trợ lý"
            lines.append(f"{role_label}: {content}")
        history_text = "\n".join(lines)

    user_content = f"Câu hỏi: {question.strip()}\n\nTrả lời bằng JSON theo đúng format đã nêu."
    if history_text:
        user_content = f"Lịch sử hội thoại gần đây:\n{history_text}\n\n{user_content}"

    llm = build_runtime_chat_client(temperature=0.0)
    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = llm.invoke(messages)
        content = resp.content if hasattr(resp, "content") else str(resp)
        intent = _parse_supervisor_response(content)
        if intent is not None:
            return intent
    except Exception:
        logger.warning("Supervisor LLM call failed.", exc_info=True)
    # Safe default: search drug collection, no ERP lookup
    return {"collections_to_search": ["drug"], "erp": False, "erp_drug_name": None}
