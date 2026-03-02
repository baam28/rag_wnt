"""RAG Chatbot API: retrieval, grounded answers with citations."""
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from config import get_settings
from retriever import retrieve


# --- Prompts (answer generation) ---

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi dựa trên ngữ cảnh (context) được cung cấp.
- Chỉ dựa vào thông tin trong context để trả lời. Mỗi tuyên bố thực tế phải có trích dẫn nguồn dạng [Source N] (N là số thứ tự nguồn).
- Nếu context không chứa thông tin đủ để trả lời câu hỏi, bạn phải nói rõ: "Tôi không có đủ thông tin cụ thể để trả lời câu hỏi này."
- Trả lời bằng cùng ngôn ngữ với câu hỏi (ưu tiên tiếng Việt nếu câu hỏi bằng tiếng Việt).
- Không bịa thông tin. Nếu không chắc chắn, hãy nói không đủ thông tin."""

USER_PROMPT_TEMPLATE = """Context (các đoạn trích từ tài liệu):

{context}

Câu hỏi: {question}

Hãy trả lời dựa trên context trên. Gắn [Source N] cho mỗi nguồn bạn dùng. Nếu không đủ thông tin, hãy nói "Tôi không có đủ thông tin cụ thể để trả lời câu hỏi này." """


def build_context_block(context_list: list[dict[str, Any]]) -> str:
    """Format retrieved context with [Source N] labels."""
    blocks = []
    for i, ctx in enumerate(context_list, 1):
        content = ctx.get("content", "").strip()
        source = ctx.get("source", "Unknown")
        blocks.append(f"[Source {i}]\n{content}\n(Nguồn: {source})")
    return "\n\n---\n\n".join(blocks)


def _generate_with_openai(query: str, context_list: list[dict[str, Any]]) -> str:
    """Generate answer from context using OpenAI."""
    settings = get_settings()
    if not settings.openai_api_key:
        return "Lỗi: Chưa cấu hình OPENAI_API_KEY."

    if not context_list:
        return "Tôi không có đủ thông tin cụ thể để trả lời câu hỏi này. (Không tìm thấy ngữ cảnh phù hợp trong cơ sở tài liệu.)"

    context_block = build_context_block(context_list)
    user_msg = USER_PROMPT_TEMPLATE.format(context=context_block, question=query)
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )
    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"Lỗi khi tạo câu trả lời: {e}"


def generate_answer(query: str, context_list: list[dict[str, Any]]) -> str:
    """Return a grounded answer with citations (OpenAI)."""
    return _generate_with_openai(query, context_list)


# --- FastAPI app ---

app = FastAPI(
    title="RAG Chatbot API",
    description="Semantic RAG with reasoning & retrieval",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    has_context: bool
    collection_name: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Retrieve context and return grounded answer with citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    try:
        context_list = retrieve(req.question)
        answer = generate_answer(req.question, context_list)
        sources = [
            {
                "rank": c.get("rank"),
                "source": c.get("source"),
                "summary": c.get("summary", "")[:200],
                "collection_name": c.get("collection_name"),
            }
            for c in context_list
        ]
        return AskResponse(
            answer=answer,
            sources=sources,
            has_context=len(context_list) > 0,
            collection_name=context_list[0].get("collection_name") if context_list else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "RAG Chatbot API. Use POST /ask with body: { \"question\": \"...\" }. "
        "Collection is chosen automatically by the retriever.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
