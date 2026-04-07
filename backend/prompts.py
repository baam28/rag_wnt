from collections import defaultdict, deque
from typing import Any, Optional, List, Dict, Tuple

import logging
import re

try:
    import tiktoken
except Exception:  # pragma: no cover - fallback when tokenizer lib unavailable
    tiktoken = None

from langchain_openai import ChatOpenAI

from config import get_settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Bạn là trợ lý pháp lý – dược học, trả lời dựa trên context được cung cấp.

Nguyên tắc:
- Đọc KỸ toàn bộ context trước khi trả lời, kể cả các văn bản sửa đổi, bổ sung.
- Khi context chứa cả văn bản gốc lẫn văn bản sửa đổi/bổ sung, hãy **ưu tiên áp dụng quy định mới nhất** (văn bản sửa đổi có hiệu lực sau thay thế điều khoản cũ). Giải thích rõ sự thay đổi.
- Phân tích và đưa ra **kết luận rõ ràng, dứt khoát**. Không né tránh kết luận khi đã có đủ căn cứ pháp lý.
- Cách trích dẫn: viết tên văn bản đúng như trong dấu ngoặc vuông của context (ví dụ: [Luật Dược, số 105/2016/QH13]).
  KHÔNG thêm phần trong ngoặc đơn theo sau, chỉ viết mình tên văn bản trong [], không dùng [Source N].
- Chỉ nói "không có đủ thông tin" khi context thực sự không đề cập đến vấn đề được hỏi.
- Trả lời bằng tiếng Việt, ngắn gọn, mạch lạc."""


USER_PROMPT_TEMPLATE = """Context (các đoạn trích từ tài liệu, mỗi đoạn được gán nhãn [Tên văn bản]):

{context}

Câu hỏi: {question}

Hướng dẫn cấu trúc câu trả lời (theo đúng thứ tự này):
1. Mở đầu: "Căn cứ vào [điều khoản cụ thể] [Tên văn bản] có hiệu lực từ ngày [ngày hiệu lực] thì [chủ đề câu hỏi] được quy định như sau:" – nếu không xác định được ngày hiệu lực từ context, bỏ phần "có hiệu lực từ ngày ...".
2. In đậm tiêu đề điều khoản: "**Điều X. [Tên điều khoản]**", sau đó dán nguyên văn điều khoản liên quan từ context dưới dạng blockquote bằng cách thêm "> " vào đầu mỗi dòng. Mỗi khoản trên một dòng riêng bắt đầu bằng "> ", giữ đúng số thứ tự khoản.
3. Nếu có văn bản sửa đổi/bổ sung: trình bày rõ quy định cũ và mới, áp dụng quy định hiện hành.
4. Kết luận: bắt đầu bằng "Như vậy," – tóm tắt các điều kiện/nội dung chính dưới dạng danh sách gạch đầu dòng.
5. Nếu người dùng có thể cần thêm thông tin liên quan (ví dụ nội dung của điều khác được dẫn chiếu), gợi ý ngắn gọn ở cuối.

Quy tắc trích dẫn: viết [Tên văn bản] đúng theo nhãn trong context. KHÔNG thêm phần ngoặc đơn nào sau tên văn bản. Không dùng [Source N]."""


DRUG_SYSTEM_PROMPT = """Bạn là trợ lý dược học, trả lời các câu hỏi về thuốc và thông tin y tế dựa trên context được cung cấp.

Nguyên tắc:
- Đọc KỸ câu hỏi để xác định PHẠM VI trả lời trước khi đọc context.
- **Trả lời đúng và đủ câu hỏi — không hơn, không kém.** Chỉ dùng bố cục nhiều mục khi người dùng hỏi tổng quát về một thuốc ("thuốc X là gì", "cho tôi thông tin về X", "tổng quan về X").
- Khi câu hỏi hẹp (liều dùng, tác dụng phụ, chống chỉ định, tương tác, cơ chế...): chỉ trả lời đúng phần đó. Không thêm các mục khác không được hỏi.
- Giữ nguyên các con số, tên hoạt chất, hàm lượng, tần suất, điều kiện dùng thuốc đúng như context — không đơn giản hóa dữ kiện chuyên môn trong phạm vi câu hỏi.
- Nếu context nêu nhiều trường hợp, đối tượng hoặc liều khác nhau cho cùng một chủ đề được hỏi, trình bày riêng từng trường hợp.
- Nếu context có thông tin mâu thuẫn, nêu rõ từng trường hợp thay vì tự hòa giải.
- Nếu thuốc là thuốc kê đơn, nhắc người dùng cần tư vấn bác sĩ/dược sĩ.
- KHÔNG nhắc đến "context", "theo context", "từ context" hay bất kỳ nguồn dữ liệu nào trong câu trả lời.
- KHÔNG thêm tên tài liệu, mã nguồn hay nhãn trích dẫn vào cuối câu.
- Chỉ nói "không có đủ thông tin" khi context thực sự không đề cập đến vấn đề được hỏi.
- Trình bày bằng Markdown rõ ràng. Không lạm dụng in đậm toàn câu; chỉ in đậm tên mục hoặc cảnh báo quan trọng.
- Trả lời bằng tiếng Việt."""


DRUG_USER_PROMPT_TEMPLATE = """Context:

{context}

Câu hỏi: {question}

Hướng dẫn trả lời:
- Xác định phạm vi câu hỏi:
    1) Câu hỏi tổng quát ("thuốc X là gì", "thông tin về X", "tổng quan về X"): trả lời theo cấu trúc nhiều mục (chỉ hiển thị mục có dữ liệu): **Giới thiệu**, **Cơ chế tác dụng**, **Chỉ định**, **Liều dùng & Dạng bào chế**, **Tác dụng phụ**, **Chống chỉ định**, **Lưu ý**.
    2) Câu hỏi về một khía cạnh cụ thể (liều dùng, tác dụng phụ, chống chỉ định, tương tác, cơ chế...): chỉ trả lời đúng khía cạnh đó. KHÔNG thêm các mục khác.
    3) Câu hỏi nguyên tắc/tư vấn chung: trả lời trực tiếp, giữ đầy đủ điều kiện, ngoại lệ và cảnh báo liên quan.
- Không rút gọn dữ kiện chuyên môn (liều, tần suất, hàm lượng, điều kiện) trong phạm vi câu hỏi.
- Không nhắc đến "context" hay nguồn trong câu trả lời.
"""


PRICE_SYSTEM_PROMPT = """Bạn là chuyên viên quản lý tồn kho và thông tin thuốc ERP.
- Trả lời ĐÚNG VÀO câu hỏi — chỉ trình bày thông tin người dùng hỏi, không liệt kê thêm các trường không liên quan.
- Ví dụ: nếu hỏi tồn kho thì chỉ trả lời số lượng tồn kho; nếu hỏi giá thì chỉ trả lời giá.
- Không mở đầu bằng "Dưới đây là thông tin chi tiết về..." hay tương tự khi câu hỏi chỉ cần một con số.
- Trình bày bằng Markdown ngắn gọn, rõ ràng. Giữ nguyên bảng Markdown nếu context có sẵn.
- Trả lời bằng tiếng Việt."""


PRICE_USER_PROMPT_TEMPLATE = """Context (kết quả truy vấn ERP/kho):

{context}

Câu hỏi: {question}

Trả lời trực tiếp câu hỏi, chỉ dùng thông tin liên quan từ context. Không trình bày các trường thông tin không được hỏi. Giữ nguyên bảng Markdown nếu có."""


COMBINED_SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về thuốc, y tế và pháp lý. Context có thể chứa thông tin từ nhiều nguồn: văn bản pháp lý (luật, nghị định, thông tư, văn bản sửa đổi), thông tin y khoa, và báo cáo dữ liệu trực tiếp từ kho ERP.

Nguyên tắc:
- Đọc KỸ toàn bộ context, kể cả các văn bản sửa đổi, bổ sung.
- Khi context chứa cả văn bản gốc lẫn văn bản sửa đổi, **ưu tiên áp dụng quy định mới nhất**. Giải thích rõ sự thay đổi.
- Trả lời ĐẦY ĐỦ mọi phần câu hỏi, đưa ra **kết luận rõ ràng, dứt khoát** khi đã có đủ căn cứ.
- Cách trích dận: viết tên văn bản trong [] đúng theo nhãn trong context. KHÔNG thêm ngoặc đơn nào sau tên văn bản. Không dùng [Source N].
- Nếu câu hỏi kết hợp pháp lý và kho: tách biệt thành 2 phần rõ rệt. Phối hợp nhịp nhàng giữa bảng Markdown dữ liệu kho và câu trả lời tư vấn pháp lý. Dữ liệu kho không được lược bớt đi định dạng.
- Trả lời bằng tiếng Việt. Không bịa thông tin."""


COMBINED_USER_PROMPT_TEMPLATE = """Context (có thể gồm văn bản pháp lý, thông tin thuốc, và báo cáo tồn kho ERP; mỗi đoạn gán nhãn rõ ràng):

{context}

Câu hỏi: {question}

Hướng dẫn: Với câu hỏi pháp lý, trả lời theo cấu trúc: (1) "Căn cứ vào [điều khoản] [Tên văn bản] có hiệu lực từ ngày [ngày] thì ... được quy định như sau:" → (2) **Điều X. Tiêu đề** + nguyên văn điều khoản dưới dạng blockquote (thêm "> " vào đầu mỗi dòng khoản, mỗi khoản xuống dòng riêng) → (3) "Như vậy," + danh sách gạch đầu dòng tóm tắt. Cung cấp dữ liệu tồn kho bằng bảng Markdown y như Context phản hồi."""


def _clean_source_name(source: str) -> str:
    """Strip common file extensions and trailing whitespace from a source filename."""
    name = (source or "Unknown").strip()
    # Remove common doc extensions (case-insensitive)
    name = re.sub(r'\s*\.(?:docx?|pdf|xlsx?|txt|csv)\s*$', '', name, flags=re.IGNORECASE)
    return name.strip()


def _get_encoder(model: str):
    if not tiktoken:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoder) -> int:
    if not text:
        return 0
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    # Fallback heuristic ~ 4 chars/token
    return max(1, len(text) // 4)


def _truncate_to_tokens(text: str, max_tokens: int, encoder) -> str:
    if not text or max_tokens <= 0:
        return ""
    if _count_tokens(text, encoder) <= max_tokens:
        return text
    if encoder:
        try:
            ids = encoder.encode(text)
            cut = ids[:max_tokens]
            out = encoder.decode(cut).strip()
            return out + " ..."
        except Exception:
            pass
    approx_chars = max_tokens * 4
    return text[:approx_chars].rstrip() + " ..."


def _context_rank(ctx: Dict[str, Any]) -> int:
    rank = ctx.get("rank")
    try:
        return int(rank)
    except Exception:
        return 10**6


def _format_context_item(
    ctx: Dict[str, Any],
    include_labels: bool,
    encoder,
    per_item_soft_cap_tokens: int,
) -> str:
    source = _clean_source_name(ctx.get("source", "Unknown"))
    content = (ctx.get("content") or "").strip()
    rank = _context_rank(ctx)

    body_parts: list[str] = []
    if content:
        body_parts.append(_truncate_to_tokens(content, per_item_soft_cap_tokens, encoder))

    body = "\n".join([p for p in body_parts if p]).strip()
    if include_labels:
        return f"[{source}] (rank={rank})\n{body}".strip()
    return body


def _build_diverse_ordered_context(
    context_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prioritise top rank per source first, then second-best per source, etc."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ctx in context_list:
        source = _clean_source_name(ctx.get("source", "Unknown"))
        grouped[source].append(ctx)
    for source in grouped:
        grouped[source].sort(key=_context_rank)

    source_order = sorted(grouped.keys(), key=lambda s: _context_rank(grouped[s][0]))
    queues = {s: deque(grouped[s]) for s in source_order}

    out: List[Dict[str, Any]] = []
    while True:
        progressed = False
        for s in source_order:
            if queues[s]:
                out.append(queues[s].popleft())
                progressed = True
        if not progressed:
            break
    return out


def _select_context_by_token_budget(
    context_list: List[Dict[str, Any]],
    total_context_budget_tokens: int,
    include_labels: bool,
    encoder,
    per_item_soft_cap_tokens: int,
) -> str:
    ordered = _build_diverse_ordered_context(context_list)
    blocks: List[str] = []
    used = 0
    sep = "\n\n---\n\n"
    sep_tokens = _count_tokens(sep, encoder)

    for ctx in ordered:
        block = _format_context_item(
            ctx=ctx,
            include_labels=include_labels,
            encoder=encoder,
            per_item_soft_cap_tokens=per_item_soft_cap_tokens,
        )
        if not block:
            continue
        block_tokens = _count_tokens(block, encoder)
        extra = block_tokens if not blocks else block_tokens + sep_tokens

        if used + extra <= total_context_budget_tokens:
            blocks.append(block)
            used += extra
            continue

        remaining = total_context_budget_tokens - used - (sep_tokens if blocks else 0)
        if remaining >= 40:
            shortened = _truncate_to_tokens(block, remaining, encoder)
            if shortened:
                blocks.append(shortened)
            break
        break

    if blocks:
        return sep.join(blocks)

    # Fallback: always keep at least the best-ranked context in clipped form.
    if ordered:
        first = _format_context_item(
            ctx=ordered[0],
            include_labels=include_labels,
            encoder=encoder,
            per_item_soft_cap_tokens=max(80, per_item_soft_cap_tokens),
        )
        if first:
            return _truncate_to_tokens(first, max(80, total_context_budget_tokens), encoder)
    return ""


def _extract_text_content(resp: Any) -> str:
    """Extract robust text from LangChain/OpenAI response objects."""
    def _collect_text(value: Any) -> list[str]:
        out: list[str] = []
        if value is None:
            return out
        if isinstance(value, str):
            text = value.strip()
            if text:
                out.append(text)
            return out
        if isinstance(value, list):
            for item in value:
                out.extend(_collect_text(item))
            return out
        if isinstance(value, dict):
            # Common OpenAI/LangChain keys where text may live.
            for key in ("text", "output_text", "content", "value"):
                if key in value:
                    out.extend(_collect_text(value.get(key)))
            # Traverse nested structures conservatively.
            for nested_key in ("message", "output", "choices", "delta"):
                if nested_key in value:
                    out.extend(_collect_text(value.get(nested_key)))
            return out
        # Objects from LangChain blocks may expose text/content attrs.
        out.extend(_collect_text(getattr(value, "text", None)))
        out.extend(_collect_text(getattr(value, "content", None)))
        return out

    content = getattr(resp, "content", None)
    text_parts = _collect_text(content)
    if not text_parts:
        # Some LangChain/OpenAI adapters keep text in metadata/additional_kwargs.
        text_parts.extend(_collect_text(getattr(resp, "additional_kwargs", None)))
        text_parts.extend(_collect_text(getattr(resp, "response_metadata", None)))
    if not text_parts and hasattr(resp, "text"):
        try:
            text_attr = getattr(resp, "text")
            text_parts.extend(_collect_text(text_attr))
        except Exception:
            pass
    return "\n".join([p for p in text_parts if p]).strip()


def _friendly_llm_error(_err: Exception) -> str:
    return "Xin lỗi, mình chưa tạo được câu trả lời đầy đủ từ dữ liệu hiện có."


def _extract_refusal(resp: Any) -> str:
    try:
        addl = getattr(resp, "additional_kwargs", {}) or {}
        refusal = addl.get("refusal")
        if refusal is None:
            return ""
        if isinstance(refusal, str):
            return refusal.strip()
        return str(refusal).strip()
    except Exception:
        return ""


def _extract_usage(resp: Any) -> Dict[str, int]:
    """Extract prompt_tokens and completion_tokens from LangChain OpenAI response."""
    out = {"prompt_tokens": 0, "completion_tokens": 0}
    try:
        meta = getattr(resp, "response_metadata", None) or {}
        usage = meta.get("usage_metadata") or meta.get("token_usage") or meta.get("usage")
        if isinstance(usage, dict):
            out["prompt_tokens"] = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
            out["completion_tokens"] = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    except Exception:
        pass
    return out


def _build_chat_openai_client(
    model_name: str,
    api_key: str,
    temperature: float,
    max_output_tokens: int,
) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
    }
    if (model_name or "").lower().startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_output_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_output_tokens
    return ChatOpenAI(**kwargs)


def _generate_with_openai(
    query: str,
    context_list: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    history_summary: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Generate answer from context using OpenAI.
    Returns (content, usage_dict with prompt_tokens, completion_tokens).
    """
    settings = get_settings()
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if not settings.openai_api_key:
        return "Lỗi: Chưa cấu hình OPENAI_API_KEY.", empty_usage

    if not context_list:
        return "Tôi không có đủ thông tin cụ thể để trả lời câu hỏi này. (Không tìm thấy ngữ cảnh phù hợp trong cơ sở tài liệu.)", empty_usage

    template = user_template or USER_PROMPT_TEMPLATE
    model_name = settings.llm_model
    encoder = _get_encoder(model_name)
    include_labels = (system_prompt != DRUG_SYSTEM_PROMPT)
    max_output_tokens = settings.llm_max_output_tokens_default
    if (system_prompt or SYSTEM_PROMPT) in (SYSTEM_PROMPT, DRUG_SYSTEM_PROMPT, COMBINED_SYSTEM_PROMPT):
        max_output_tokens = settings.llm_max_output_tokens_legal

    llm_total_budget = max(settings.llm_total_budget_tokens, max_output_tokens + 1500)
    input_budget = max(1000, llm_total_budget - max_output_tokens)

    system_msg = system_prompt or SYSTEM_PROMPT
    user_template_without_context = template.format(context="", question=query)
    fixed_tokens = _count_tokens(system_msg, encoder) + _count_tokens(user_template_without_context, encoder)
    available_dynamic_tokens = max(400, input_budget - fixed_tokens)

    history_budget = int(available_dynamic_tokens * settings.llm_history_budget_ratio)
    context_budget = int(available_dynamic_tokens * settings.llm_context_budget_ratio)
    slack = max(120, available_dynamic_tokens - history_budget - context_budget)

    # Keep summary + recent turns under history budget.
    history_messages: List[Dict[str, str]] = []
    history_used = 0
    if history_summary:
        summary_text = _truncate_to_tokens(history_summary, max(40, history_budget // 2), encoder)
        if summary_text:
            history_messages.append({
                "role": "system",
                "content": f"Tóm tắt hội thoại trước đó: {summary_text}",
            })
            history_used += _count_tokens(history_messages[-1]["content"], encoder)

    raw_history = []
    if history:
        for msg in history:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                raw_history.append({"role": role, "content": content})
    for msg in raw_history[-8:]:
        content_tokens = _count_tokens(msg["content"], encoder)
        if history_used + content_tokens <= history_budget:
            history_messages.append(msg)
            history_used += content_tokens
        else:
            remaining = history_budget - history_used
            if remaining >= 30:
                clipped = _truncate_to_tokens(msg["content"], remaining, encoder)
                if clipped:
                    history_messages.append({"role": msg["role"], "content": clipped})
            break

    per_item_soft_cap = max(120, min(settings.llm_context_chunk_soft_cap_tokens, context_budget // max(1, len(context_list))))
    context_block = _select_context_by_token_budget(
        context_list=context_list,
        total_context_budget_tokens=max(120, context_budget + slack // 2),
        include_labels=include_labels,
        encoder=encoder,
        per_item_soft_cap_tokens=per_item_soft_cap,
    )
    user_msg = template.format(context=context_block, question=query)

    # Final guardrail: if still over input budget, progressively shrink context then history.
    def _message_tokens(msgs: List[Dict[str, str]]) -> int:
        # Approximate chat overhead with +4 tokens/message
        return sum(_count_tokens(m.get("content", ""), encoder) + 4 for m in msgs)

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_msg}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_msg})
    if _message_tokens(messages) > input_budget:
        shrink_budget = max(80, int(context_budget * 0.75))
        context_block = _select_context_by_token_budget(
            context_list=context_list,
            total_context_budget_tokens=shrink_budget,
            include_labels=include_labels,
            encoder=encoder,
            per_item_soft_cap_tokens=max(100, per_item_soft_cap - 40),
        )
        user_msg = template.format(context=context_block, question=query)
        messages = [{"role": "system", "content": system_msg}] + history_messages + [{"role": "user", "content": user_msg}]
    while len(messages) > 2 and _message_tokens(messages) > input_budget:
        # Drop oldest non-system history message first.
        del messages[1]

    llm = _build_chat_openai_client(
        model_name=model_name,
        api_key=settings.openai_api_key,
        temperature=0.2,
        max_output_tokens=max_output_tokens,
    )

    fallback_model_name = (getattr(settings, "llm_fallback_model", "") or "").strip()
    use_fallback = bool(fallback_model_name and fallback_model_name != model_name)

    try:
        resp = llm.invoke(messages)
        content = _extract_text_content(resp)
        refusal_reason = _extract_refusal(resp)
        if not content and refusal_reason:
            logger.warning(
                "LLM refusal detected. model=%s query=%r refusal=%r finish_reason=%r",
                model_name,
                (query or "")[:200],
                refusal_reason[:400],
                (getattr(resp, "response_metadata", {}) or {}).get("finish_reason"),
            )
        if not content:
            # Safety net: ask once more with a minimal direct instruction.
            rescue_messages = list(messages)
            rescue_messages.append({
                "role": "system",
                "content": "Nếu chưa chắc chắn, vẫn phải trả lời ngắn gọn theo context hiện có; tuyệt đối không để trống.",
            })
            try:
                rescue = llm.invoke(rescue_messages)
                content = _extract_text_content(rescue)
                if not content and _extract_refusal(rescue):
                    logger.warning(
                        "LLM refusal on rescue. model=%s query=%r refusal=%r",
                        model_name,
                        (query or "")[:200],
                        _extract_refusal(rescue)[:400],
                    )
            except Exception as rescue_err:
                logger.exception(
                    "LLM rescue invoke failed. model=%s query=%r context_items=%d",
                    model_name,
                    (query or "")[:200],
                    len(context_list or []),
                )
                return _friendly_llm_error(rescue_err), empty_usage
        if not content and use_fallback:
            try:
                fallback_llm = _build_chat_openai_client(
                    model_name=fallback_model_name,
                    api_key=settings.openai_api_key,
                    temperature=0.2,
                    max_output_tokens=max_output_tokens,
                )
                fallback_resp = fallback_llm.invoke(messages)
                content = _extract_text_content(fallback_resp)
                if content:
                    logger.warning(
                        "Recovered by fallback model. primary=%s fallback=%s query=%r",
                        model_name,
                        fallback_model_name,
                        (query or "")[:200],
                    )
                    usage = _extract_usage(fallback_resp)
                    return content, usage
                logger.warning(
                    "Fallback model also returned empty. primary=%s fallback=%s query=%r fallback_refusal=%r",
                    model_name,
                    fallback_model_name,
                    (query or "")[:200],
                    _extract_refusal(fallback_resp)[:400],
                )
            except Exception:
                logger.exception(
                    "Fallback model invoke failed. primary=%s fallback=%s query=%r",
                    model_name,
                    fallback_model_name,
                    (query or "")[:200],
                )
        if not content:
            logger.warning(
                "LLM returned empty content after rescue. model=%s query=%r prompt_tokens_est=%d context_items=%d content_repr=%r addl_keys=%s meta_keys=%s",
                model_name,
                (query or "")[:200],
                _message_tokens(messages),
                len(context_list or []),
                repr(getattr(resp, "content", None))[:500],
                list((getattr(resp, "additional_kwargs", {}) or {}).keys()),
                list((getattr(resp, "response_metadata", {}) or {}).keys()),
            )
            content = _friendly_llm_error(ValueError("empty_content"))
        usage = _extract_usage(resp)
        return content, usage
    except Exception as e:
        logger.exception(
            "LLM invoke failed. model=%s query=%r context_items=%d",
            model_name,
            (query or "")[:200],
            len(context_list or []),
        )
        return _friendly_llm_error(e), empty_usage


def generate_answer(
    query: str,
    context_list: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    history_summary: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
) -> Tuple[str, Dict[str, int]]:
    """Return (grounded answer with citations, usage_dict)."""
    return _generate_with_openai(
        query, context_list, history=history, history_summary=history_summary,
        system_prompt=system_prompt, user_template=user_template,
    )
