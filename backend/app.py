"""RAG Chatbot API: retrieval, grounded answers with citations."""
import asyncio
from typing import Any, Optional
from pathlib import Path
import shutil
import json
import time
import threading

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from qdrant_client.http import models as qmodels

from config import get_settings
from retriever import retrieve, get_qdrant_client
from ingest import ingest_file
from drug_price_tool import get_vietnam_drug_price, detect_price_query


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


def build_context_block(context_list: list[dict[str, Any]]) -> str:
    """Format retrieved context with [Source N] labels."""
    blocks = []
    for i, ctx in enumerate(context_list, 1):
        content = ctx.get("content", "").strip()
        source = ctx.get("source", "Unknown")
        blocks.append(f"[Source {i}]\n{content}\n(Nguồn: {source})")
    return "\n\n---\n\n".join(blocks)


def _generate_with_openai(
    query: str,
    context_list: list[dict[str, Any]],
    history: Optional[list[dict[str, str]]] = None,
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

    messages: list[dict[str, str]] = [
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
    context_list: list[dict[str, Any]],
    history: Optional[list[dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
) -> str:
    """Return a grounded answer with citations (OpenAI), aware of prior chat history."""
    return _generate_with_openai(
        query, context_list, history=history,
        system_prompt=system_prompt, user_template=user_template,
    )


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


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    has_context: bool
    collection_name: Optional[str] = None
    price_data: Optional[dict[str, Any]] = None


class IngestResponse(BaseModel):
    file: str
    collection_name: str
    num_parents: int
    num_children: int
    total_chunks_in_db: int


class CollectionInfo(BaseModel):
    name: str


class DocumentInfo(BaseModel):
    source: str
    parent_count: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Retrieve context and return grounded answer with citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    try:
        is_price, drug_name = detect_price_query(req.question)
        price_data: Optional[dict[str, Any]] = None

        if is_price and drug_name:
            try:
                price_data = get_vietnam_drug_price(drug_name, question=req.question)
            except Exception:
                price_data = None

        try:
            context_list = retrieve(req.question)
        except Exception:
            context_list = []

        has_price_context = False
        if price_data and (
            price_data.get("prices")
            or price_data.get("price_range")
            or price_data.get("is_prescription")
        ):
            price_ctx = _format_price_as_context(price_data)
            context_list = [price_ctx] + context_list
            has_price_context = True

        history_payload = [
            {"role": m.role, "content": m.content}
            for m in (req.history or [])
            if m.content and m.role in ("user", "assistant")
        ]
        answer = generate_answer(
            req.question,
            context_list,
            history=history_payload,
            system_prompt=PRICE_SYSTEM_PROMPT if has_price_context else None,
            user_template=PRICE_USER_PROMPT_TEMPLATE if has_price_context else None,
        )
        sources = [
            {
                "rank": c.get("rank"),
                "source": c.get("source"),
                "summary": c.get("summary", "")[:200],
                "content": c.get("content", "")[:1000],
                "collection_name": c.get("collection_name"),
                "page": c.get("page"),
            }
            for c in context_list
        ]
        return AskResponse(
            answer=answer,
            sources=sources,
            has_context=len(context_list) > 0,
            collection_name=context_list[0].get("collection_name") if context_list else None,
            price_data=price_data,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            lines.append(f"Có {len(drugs)} loại thuốc liên quan. Chi tiết từng thuốc và link xem ở bảng giá bên dưới.")
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

    # Do not add disclaimer to context; it is shown once in the price list UI below.

    return {
        "content": "\n".join(lines),
        "source": "Tra cứu giá thuốc trực tuyến",
        "summary": price_data.get("price_range", f"Giá thuốc {drug}"),
        "collection_name": None,
        "rank": 0,
        "page": None,
    }


@app.post("/ingest-file", response_model=IngestResponse)
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    collection_name: str = Form("rag_chatbot"),
):
    """Ingest a single uploaded file into the vector store."""
    settings = get_settings()
    upload_dir = settings.upload_dir
    if not isinstance(upload_dir, Path):
        upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / file.filename
    try:
        contents = await file.read()
        with open(target_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ingest_file(
                target_path,
                collection_name=collection_name,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return IngestResponse(
        file=result.get("file", str(target_path)),
        collection_name=result.get("collection_name", collection_name),
        num_parents=result.get("num_parents", 0),
        num_children=result.get("num_children", 0),
        total_chunks_in_db=result.get("total_chunks_in_db", 0),
    )


@app.post("/db/clear")
def clear_db():
    """Clear the entire Qdrant database directory (persist_dir)."""
    settings = get_settings()
    db_path = settings.persist_dir
    if not isinstance(db_path, Path):
        db_path = Path(db_path)
    try:
        if db_path.exists():
            shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"Cleared database at '{db_path}'. You can ingest again."}


@app.get("/admin/collections", response_model=list[CollectionInfo])
def list_collections():
    """List all Qdrant collections."""
    settings = get_settings()
    client = get_qdrant_client(settings.persist_dir)
    try:
        resp = client.get_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return [CollectionInfo(name=c.name) for c in resp.collections]


@app.get("/admin/docs", response_model=list[DocumentInfo])
def list_documents(collection_name: str = Query(..., alias="collection_name")):
    """
    List logical documents in a collection, grouped by `source`.
    Uses the {collection}_parents.json file to count parent chunks per source.
    """
    settings = get_settings()
    parents_path = settings.persist_dir / f"{collection_name}_parents.json"
    if not parents_path.exists():
        return []
    try:
        with open(parents_path, "r", encoding="utf-8") as f:
            parents = __import__("json").load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    counts: dict[str, int] = {}
    for meta in parents.values():
        src = meta.get("source", "Unknown")
        counts[src] = counts.get(src, 0) + 1

    return [DocumentInfo(source=s, parent_count=n) for s, n in counts.items()]


class DeleteDocumentRequest(BaseModel):
    collection_name: str
    source: str


@app.delete("/admin/docs")
def delete_document(req: DeleteDocumentRequest):
    """
    Delete all chunks and parent metadata for a given document (`source`) in a collection.
    """
    settings = get_settings()
    client = get_qdrant_client(settings.persist_dir)

    # Delete points from Qdrant filtered by payload.source
    try:
        next_offset = None
        all_ids: list[int] = []
        while True:
            scroll_result, next_offset = client.scroll(
                collection_name=req.collection_name,
                limit=1000,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source",
                            match=qmodels.MatchValue(value=req.source),
                        )
                    ]
                ),
            )
            if not scroll_result:
                break
            for p in scroll_result:
                all_ids.append(p.id)
            if next_offset is None:
                break
        if all_ids:
            client.delete(
                collection_name=req.collection_name,
                points_selector=qmodels.PointIdsList(points=all_ids),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete points: {e}")

    # Update parents metadata file
    parents_path = settings.persist_dir / f"{req.collection_name}_parents.json"
    if parents_path.exists():
        try:
            with open(parents_path, "r", encoding="utf-8") as f:
                parents = json.load(f)
            parents = {
                pid: meta
                for pid, meta in parents.items()
                if meta.get("source") != req.source
            }
            with open(parents_path, "w", encoding="utf-8") as f:
                json.dump(parents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update parents: {e}")

    return {"message": f"Deleted document '{req.source}' from collection '{req.collection_name}'."}


@app.delete("/admin/collections/{collection_name}")
def delete_collection(collection_name: str):
    """
    Delete an entire collection and its associated metadata files.
    """
    settings = get_settings()
    client = get_qdrant_client(settings.persist_dir)
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Remove parents + sparse vocab files if present
    parents_path = settings.persist_dir / f"{collection_name}_parents.json"
    vocab_path = settings.persist_dir / f"{collection_name}_sparse_vocab.json"
    for p in (parents_path, vocab_path):
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    return {"message": f"Deleted collection '{collection_name}' and related metadata."}


# --- Drug price lookup ---

class DrugPriceRequest(BaseModel):
    drug_name: str


@app.post("/drug-price")
def drug_price(req: DrugPriceRequest):
    """Look up real-time retail prices for a medicine in Vietnam."""
    if not req.drug_name.strip():
        raise HTTPException(status_code=400, detail="drug_name is required")
    try:
        result = get_vietnam_drug_price(req.drug_name.strip())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Feedback storage ---

_FEEDBACK_FILE = Path(__file__).resolve().parent.parent / "feedback.json"
_feedback_lock = threading.Lock()


def _load_feedback() -> list[dict]:
    if _FEEDBACK_FILE.exists():
        try:
            with open(_FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_feedback(data: list[dict]):
    with open(_FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str  # "up" or "down"
    comment: Optional[str] = None
    session_id: Optional[str] = None


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": req.question[:500],
        "answer": req.answer,
        "rating": req.rating,
        "comment": (req.comment or "")[:500],
        "session_id": req.session_id or "",
    }
    with _feedback_lock:
        data = _load_feedback()
        data.append(entry)
        _save_feedback(data)
    return {"status": "ok"}


@app.get("/admin/feedback")
def get_feedback():
    data = _load_feedback()
    total = len(data)
    up = sum(1 for d in data if d.get("rating") == "up")
    down = sum(1 for d in data if d.get("rating") == "down")
    down_entries = [d for d in data if d.get("rating") == "down"]
    down_entries.reverse()
    return {
        "total": total,
        "up": up,
        "down": down,
        "down_entries": down_entries[:100],
        "all_entries": list(reversed(data))[:200],
    }


@app.get("/")
def root():
    return {
        "message": "RAG Chatbot API. Use POST /ask with body: { \"question\": \"...\" }. "
        "UI: GET /app/ to open the frontend.",
    }


# Serve uploaded documents at /uploads/{filename}
_upload_dir = Path(__file__).resolve().parent.parent / "uploads"
_upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(_upload_dir)), name="uploads")

# Serve frontend at /app (project root / frontend/)
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
