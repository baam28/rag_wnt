from typing import Any, Optional, List, Dict

from langchain_openai import ChatOpenAI

from config import get_settings


SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi dựa trên ngữ cảnh (context) được cung cấp.
- Chỉ dựa vào thông tin trong context để trả lời. Mỗi tuyên bố thực tế phải có trích dẫn nguồn dạng [Source N] (N là số thứ tự nguồn).
- Nếu context không chứa thông tin đủ để trả lời câu hỏi, bạn phải nói rõ: "Tôi không có đủ thông tin cụ thể để trả lời câu hỏi này."
- Trả lời bằng cùng ngôn ngữ với câu hỏi (ưu tiên tiếng Việt nếu câu hỏi bằng tiếng Việt).
- Không bịa thông tin. Nếu không chắc chắn, hãy nói không đủ thông tin."""


USER_PROMPT_TEMPLATE = """Context (các đoạn trích từ tài liệu):

{context}

Câu hỏi: {question}

Hãy trả lời dựa trên context trên. Gắn [Source N] cho mỗi nguồn bạn dùng. Nếu không đủ thông tin, hãy nói "Tôi không có đủ thông tin cụ thể để trả lời câu hỏi này." """


PRICE_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu giá thuốc tại Việt Nam.
- Khi context chứa kết quả tra cứu giá thuốc, KHÔNG liệt kê từng mục giá trong câu trả lời. Chỉ tóm tắt ngắn: số loại thuốc tìm được, khoảng giá (từ X đến Y), và nhắc người dùng xem bảng giá bên dưới để xem chi tiết từng thuốc và link.
- Nếu thuốc là thuốc kê đơn (Rx), hãy nói rõ và khuyên người dùng liên hệ nhà thuốc.
- KHÔNG nhắc lại lưu ý về giá thay đổi hay xác nhận với nhà thuốc/dược sĩ trong câu trả lời; lưu ý đó đã hiển thị ở bảng giá bên dưới.
- Nếu có thêm thông tin từ tài liệu nội bộ (liều dùng, chỉ định, v.v.), hãy bổ sung.
- Trả lời bằng tiếng Việt."""


PRICE_USER_PROMPT_TEMPLATE = """Context (bao gồm kết quả tra cứu giá và tài liệu liên quan):

{context}

Câu hỏi: {question}

Hãy trả lời ngắn gọn: tóm tắt số loại thuốc và khoảng giá, nhắc xem bảng bên dưới để xem chi tiết và link. KHÔNG liệt kê từng thuốc/giá trong câu trả lời. Gắn [Source N] nếu dùng nguồn. Không nhắc lại lưu ý về giá (đã có ở bảng bên dưới)."""


COMBINED_SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về thuốc, y tế và pháp lý. Context có thể chứa thông tin từ nhiều nguồn khác nhau: văn bản pháp lý (chỉ thị, nghị định, thông tư), thông tin dược lý (tác dụng, liều dùng), và kết quả tra cứu giá thuốc.
- Trả lời ĐẦY ĐỦ mọi phần câu hỏi của người dùng bằng cách tổng hợp đúng các phần tương ứng từ context (ví dụ: gộp thông tin pháp lý và thông tin thuốc nếu câu hỏi chạm vào cả hai).
- Giải thích rõ ràng và mạch lạc, chuyển ý mượt mà giữa các khía cạnh khác nhau.
- Phần giá (nếu có): chỉ tóm tắt (số loại thuốc, khoảng giá) và nhắc xem bảng giá bên dưới; KHÔNG liệt kê từng mục giá. KHÔNG nhắc lại lưu ý về giá.
- Mỗi tuyên bố thực tế phải có trích dẫn [Source N]. Trả lời bằng tiếng Việt. Không bịa thông tin."""


COMBINED_USER_PROMPT_TEMPLATE = """Context (có thể gồm thông tin thuốc/hoạt chất và giá thuốc):

{context}

Câu hỏi: {question}

Hãy tổng hợp câu trả lời bao phủ MỌI khía cạnh người dùng hỏi. Dùng đúng nguồn cho từng phần (pháp lý, thông tin thuốc, giá cả). Với giá chỉ tóm tắt và hướng dẫn xem bảng; không liệt kê giá. Gắn [Source N] cho mỗi nguồn dùng. Nếu thiếu dữ liệu cho một phần, hãy nói rõ."""


def build_context_block(context_list: List[Dict[str, Any]]) -> str:
    """Format retrieved context with [Source N] labels."""
    blocks = []
    for i, ctx in enumerate(context_list, 1):
        content = ctx.get("content", "").strip()
        source = ctx.get("source", "Unknown")
        blocks.append(f"[Source {i}]\n{content}\n(Nguồn: {source})")
    return "\n\n---\n\n".join(blocks)


def _generate_with_openai(
    query: str,
    context_list: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
) -> str:
    """
    Generate answer from context using OpenAI.
    history: optional list of {"role": "user"/"assistant", "content": "..."} for chat continuity.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return "Lỗi: Chưa cấu hình OPENAI_API_KEY."

    if not context_list:
        return "Tôi không có đủ thông tin cụ thể để trả lời câu hỏi này. (Không tìm thấy ngữ cảnh phù hợp trong cơ sở tài liệu.)"

    context_block = build_context_block(context_list)
    template = user_template or USER_PROMPT_TEMPLATE
    user_msg = template.format(context=context_block, question=query)
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
    ]
    if history:
        for msg in history[-8:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = llm.invoke(messages)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"Lỗi khi tạo câu trả lời: {e}"


def generate_answer(
    query: str,
    context_list: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
) -> str:
    """Return a grounded answer with citations (OpenAI), aware of prior chat history."""
    return _generate_with_openai(
        query, context_list, history=history,
        system_prompt=system_prompt, user_template=user_template,
    )
