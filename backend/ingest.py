"""Ingestion: load PDF/DOCX, semantic chunking, parent-child hierarchy, metadata (summary, target_question)."""
import base64
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import tiktoken
from docling.document_converter import DocumentConverter  # type: ignore[import-untyped]
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import get_settings
from utils import fix_position_ids as _fix_position_ids, get_qdrant_client, tokenize_for_sparse as _tokenize_for_sparse

_ingest_logger = logging.getLogger(__name__)

TABLE_START = "<!--TABLE_START-->"
TABLE_END = "<!--TABLE_END-->"


# --- Sparse vector helpers (for Qdrant native sparse) ---


def _build_vocab(texts: list[str], max_vocab_size: int = 100_000) -> dict[str, int]:
    """Build token -> index vocabulary from corpus. Returns dict and saves nothing."""
    from collections import Counter
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(_tokenize_for_sparse(t))
    vocab = {}
    for token, _ in counter.most_common(max_vocab_size):
        vocab[token] = len(vocab)
    return vocab


def _text_to_sparse_vector(
    text: str,
    vocab: dict[str, int],
    use_tf: bool = True,
) -> tuple[list[int], list[float]]:
    """Convert text to (indices, values) for Qdrant SparseVector. Uses 1+log(tf)."""
    from math import log
    tokens = _tokenize_for_sparse(text)
    if not tokens:
        return [], []
    tf: dict[int, float] = {}
    for t in tokens:
        idx = vocab.get(t)
        if idx is None:
            continue
        tf[idx] = tf.get(idx, 0) + 1
    if use_tf:
        # Sublinear: 1 + log(tf)
        for k in tf:
            tf[k] = 1.0 + log(tf[k])
    indices = sorted(tf.keys())
    values = [float(tf[i]) for i in indices]
    return indices, values


# --- Tokenization & text splitting ---

def get_encoding():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text))


def chunk_by_tokens(text: str, max_tokens: int, overlap: int = 50) -> list[str]:
    """Split text into chunks by token count with overlap."""
    enc = get_encoding()
    tokens = enc.encode(text)
    out = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        out.append(enc.decode(tokens[start:end]))
        start = end - overlap if end < len(tokens) else len(tokens)
    return out


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, keeping TABLE_START...TABLE_END blocks atomic."""
    table_pattern = re.compile(
        rf"({re.escape(TABLE_START)}.*?{re.escape(TABLE_END)})",
        re.DOTALL,
    )
    segments = table_pattern.split(text)
    parts: list[str] = []
    for seg in segments:
        seg_stripped = seg.strip()
        if not seg_stripped:
            continue
        if seg_stripped.startswith(TABLE_START):
            parts.append(seg_stripped)
        else:
            for p in re.split(r"\n\s*\n+", seg):
                p = p.strip()
                if p:
                    parts.append(p)
    if not parts and text.strip():
        parts = [text.strip()]
    return parts


def _split_sentences(text: str) -> list[str]:
    """Simple rule-based sentence splitter (Vietnamese and English)."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _split_words(text: str) -> list[str]:
    return text.split()


def split_by_paragraphs(text: str, max_paragraphs: int, overlap: int = 0) -> list[str]:
    paras = _split_paragraphs(text)
    if not paras:
        return []
    max_p = max(1, max_paragraphs)
    chunks: list[str] = []
    start = 0
    while start < len(paras):
        end = min(start + max_p, len(paras))
        chunks.append("\n\n".join(paras[start:end]))
        if end >= len(paras):
            break
        start = max(0, end - max(overlap, 0))
    return chunks


def split_by_sentences(text: str, max_sentences: int, overlap: int = 1) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    max_s = max(1, max_sentences)
    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        end = min(start + max_s, len(sentences))
        chunks.append(" ".join(sentences[start:end]))
        if end >= len(sentences):
            break
        start = max(0, end - max(overlap, 0))
    return chunks


def split_by_words(text: str, max_words: int, overlap: int = 30) -> list[str]:
    words = _split_words(text)
    if not words:
        return []
    max_w = max(1, max_words)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_w, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(0, end - max(overlap, 0))
    return chunks


def _estimate_doc_stats(documents: list[Document]) -> dict[str, float]:
    """Compute simple statistics to drive auto chunking decisions."""
    full_text_parts: list[str] = []
    for d in documents:
        if d and getattr(d, "page_content", None):
            full_text_parts.append(str(d.page_content))
    full_text = "\n\n".join(full_text_parts).strip()
    if not full_text:
        return {
            "num_paragraphs": 0,
            "num_sentences": 0,
            "num_words": 0,
            "avg_words_per_paragraph": 0.0,
            "avg_words_per_sentence": 0.0,
        }

    paragraphs = _split_paragraphs(full_text)
    sentences = _split_sentences(full_text)
    words = _split_words(full_text)

    num_paragraphs = len(paragraphs)
    num_sentences = len(sentences)
    num_words = len(words)

    avg_words_per_paragraph = num_words / max(1, num_paragraphs)
    avg_words_per_sentence = num_words / max(1, num_sentences)

    return {
        "num_paragraphs": num_paragraphs,
        "num_sentences": num_sentences,
        "num_words": num_words,
        "avg_words_per_paragraph": avg_words_per_paragraph,
        "avg_words_per_sentence": avg_words_per_sentence,
    }


def auto_select_chunk_strategy(documents: list[Document]) -> str:
    """
    Heuristic to choose a chunking strategy automatically:
    - Prefer paragraph-based when documents are clearly structured into many paragraphs.
    - Prefer sentence-based for long, dense prose.
    - Fall back to semantic token-based when structure is unclear.
    """
    stats = _estimate_doc_stats(documents)
    num_paragraphs = stats["num_paragraphs"]
    num_sentences = stats["num_sentences"]
    avg_words_per_paragraph = stats["avg_words_per_paragraph"]
    avg_words_per_sentence = stats["avg_words_per_sentence"]

    if num_paragraphs >= 10 and 40 <= avg_words_per_paragraph <= 200:
        return "paragraph"
    if num_sentences >= 20 and 10 <= avg_words_per_sentence <= 60:
        return "sentence"
    return "semantic_tokens"


def _load_pdf_docling(path: Path, settings=None) -> list[Document]:
    """
    Load PDF via Docling and convert it to rich Markdown text for RAG.

    - Uses Docling's DocumentConverter to parse the PDF.
    - Exports the full document to Markdown (headings, lists, tables).
    - Ignores images (no GPT-4 Vision).
    """
    if settings is None:
        settings = get_settings()

    converter = DocumentConverter()
    try:
        # Docling auto-detects the format from the file path
        result = converter.convert(str(path))
    except Exception:
        # On failure, return empty and let caller handle the error
        return []

    doc = getattr(result, "document", None)
    if doc is None:
        return []

    try:
        markdown = doc.export_to_markdown()
    except Exception:
        return []

    markdown = (markdown or "").strip()
    if not markdown:
        return []

    return [
        Document(
            page_content=markdown,
            metadata={
                "source": path.name,
                "file_path": str(path),
                # Treat entire Docling output as a single logical page/section
                "page": 0,
            },
        )
    ]


def _load_docx(path: Path) -> list[Document]:
    """Load DOCX via python-docx (no extra system deps)."""
    try:
        from docx import Document as DocxDocument
    except ImportError:
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(str(path))
            return loader.load()
        except Exception:
            raise ImportError("Install python-docx: pip install python-docx")
    doc = DocxDocument(path)
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    text = "\n\n".join(parts) or ""
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": path.name, "file_path": str(path)})]


def load_document(file_path: Path, settings=None) -> list[Document]:
    """Load a single PDF or DOCX file into LangChain documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = _load_pdf_docling(path, settings=settings)
    elif suffix in (".docx", ".doc"):
        docs = _load_docx(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .pdf or .docx")

    for d in docs:
        d.metadata.setdefault("source", str(path.name))
        d.metadata["file_path"] = str(path)
    return docs


# --- Parent and child chunking ---

_PRESPLIT_THRESHOLD_TOKENS = 50_000
_PRESPLIT_SECTION_TOKENS = 10_000


def _presplit_large_doc(doc: Document) -> list[Document]:
    """Split a very large document into sections of ~_PRESPLIT_SECTION_TOKENS
    on paragraph boundaries so SemanticChunker doesn't choke."""
    total = count_tokens(doc.page_content)
    if total <= _PRESPLIT_THRESHOLD_TOKENS:
        return [doc]

    paras = _split_paragraphs(doc.page_content)
    sections: list[Document] = []
    buf: list[str] = []
    buf_tokens = 0
    for p in paras:
        n = count_tokens(p)
        if buf_tokens + n > _PRESPLIT_SECTION_TOKENS and buf:
            sections.append(
                Document(page_content="\n\n".join(buf), metadata=dict(doc.metadata))
            )
            buf = [p]
            buf_tokens = n
        else:
            buf.append(p)
            buf_tokens += n
    if buf:
        sections.append(
            Document(page_content="\n\n".join(buf), metadata=dict(doc.metadata))
        )
    return sections


def semantic_parent_chunks(
    documents: list[Document],
    parent_target_tokens: int,
    embeddings: Any,
) -> list[Document]:
    """
    Split markdown documents by their headers, then recursively character split
    any remaining overly long sections to fit the target token limit.
    This replaces the expensive SemanticChunker.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_target_tokens * 4,  # Approx 4 chars per token
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    parents: list[Document] = []
    for doc in documents:
        text = doc.page_content or ""
        md_splits = markdown_splitter.split_text(text)
        
        for md_split in md_splits:
            # Reattach original file-level metadata to the markdown sections
            merged_metadata = {**doc.metadata, **md_split.metadata}
            md_split.metadata = merged_metadata
            
            # If the md section is still too big, recursive character split it
            if count_tokens(md_split.page_content) > parent_target_tokens:
                sub_splits = char_splitter.split_documents([md_split])
                parents.extend(sub_splits)
            else:
                parents.append(md_split)

    return parents


def paragraph_parent_chunks(
    documents: list[Document],
    parent_paragraphs: int,
) -> list[Document]:
    """Group documents into parent chunks by paragraphs."""
    parents: list[Document] = []
    for doc in documents:
        text = doc.page_content or ""
        paras = _split_paragraphs(text)
        if not paras:
            continue
        current: list[str] = []
        for p in paras:
            current.append(p)
            if len(current) >= max(1, parent_paragraphs):
                parents.append(
                    Document(
                        page_content="\n\n".join(current),
                        metadata=dict(doc.metadata),
                    )
                )
                current = []
        if current:
            parents.append(
                Document(
                    page_content="\n\n".join(current),
                    metadata=dict(doc.metadata),
                )
            )
    return parents


def sentence_parent_chunks(
    documents: list[Document],
    parent_sentences: int,
) -> list[Document]:
    """Group documents into parent chunks by sentence count."""
    parents: list[Document] = []
    for doc in documents:
        text = doc.page_content or ""
        sentences = _split_sentences(text)
        if not sentences:
            continue
        current: list[str] = []
        for s in sentences:
            current.append(s)
            if len(current) >= max(1, parent_sentences):
                parents.append(
                    Document(
                        page_content=" ".join(current),
                        metadata=dict(doc.metadata),
                    )
                )
                current = []
        if current:
            parents.append(
                Document(
                    page_content=" ".join(current),
                    metadata=dict(doc.metadata),
                )
            )
    return parents


def word_parent_chunks(
    documents: list[Document],
    parent_words: int,
) -> list[Document]:
    """Group documents into parent chunks by word count."""
    parents: list[Document] = []
    for doc in documents:
        text = doc.page_content or ""
        words = _split_words(text)
        if not words:
            continue
        max_w = max(1, parent_words)
        start = 0
        while start < len(words):
            end = min(start + max_w, len(words))
            chunk_text = " ".join(words[start:end])
            parents.append(
                Document(
                    page_content=chunk_text,
                    metadata=dict(doc.metadata),
                )
            )
            start = end
    return parents


def build_parent_chunks(
    documents: list[Document],
    settings,
    embeddings: Any,
) -> list[Document]:
    """
    Build parent chunks according to configured strategy:
    - semantic_tokens (default)
    - paragraph
    - sentence
    - word
    - auto (choose best heuristic based on document statistics)
    """
    strategy = getattr(settings, "chunk_strategy", "semantic_tokens") or "semantic_tokens"
    strategy = strategy.lower()

    if strategy == "auto":
        # Use whole-document statistics to pick a concrete strategy
        strategy = auto_select_chunk_strategy(documents)

    if strategy == "paragraph":
        return paragraph_parent_chunks(
            documents,
            parent_paragraphs=getattr(settings, "parent_chunk_paragraphs", 4),
        )
    if strategy == "sentence":
        return sentence_parent_chunks(
            documents,
            parent_sentences=getattr(settings, "parent_chunk_sentences", 8),
        )
    if strategy == "word":
        return word_parent_chunks(
            documents,
            parent_words=getattr(settings, "parent_chunk_words", 400),
        )

    # Fallback to existing semantic + token-based approach
    return semantic_parent_chunks(
        documents,
        parent_target_tokens=getattr(settings, "parent_chunk_tokens", 700),
        embeddings=embeddings,
    )


def parent_to_children_dynamic(parent: Document, settings, *, effective_strategy: Optional[str] = None) -> list[Document]:
    """
    Split a parent chunk into child chunks according to the configured strategy.
    - semantic_tokens: token-based windows
    - paragraph: paragraphs per chunk
    - sentence: sentences per chunk
    - word: words per chunk
    """
    text = parent.page_content or ""
    strategy = effective_strategy or getattr(settings, "chunk_strategy", "semantic_tokens") or "semantic_tokens"
    strategy = strategy.lower()

    if strategy == "auto":
        strategy = auto_select_chunk_strategy([parent])

    if strategy == "paragraph":
        chunks = split_by_paragraphs(
            text,
            max_paragraphs=getattr(settings, "child_chunk_paragraphs", 1),
            overlap=0,
        )
    elif strategy == "sentence":
        chunks = split_by_sentences(
            text,
            max_sentences=getattr(settings, "child_chunk_sentences", 3),
            overlap=1,
        )
    elif strategy == "word":
        chunks = split_by_words(
            text,
            max_words=getattr(settings, "child_chunk_words", 120),
            overlap=30,
        )
    else:
        chunks = chunk_by_tokens(
            text,
            max_tokens=getattr(settings, "child_chunk_tokens", 150),
            overlap=30,
        )

    children: list[Document] = []
    for i, c in enumerate(chunks):
        content = c.strip()
        if not content:
            continue
        children.append(
            Document(
                page_content=content,
                metadata={
                    **parent.metadata,
                    "chunk_index": i,
                    "parent_content": text,
                },
            )
        )
    return children


def add_parent_metadata(parent: Document, llm: ChatOpenAI) -> tuple[str, str]:
    """Generate summary and target_question for a parent chunk via LLM."""
    prompt = """Bạn là trợ lý tóm tắt. Cho đoạn văn sau (có thể bằng tiếng Việt hoặc tiếng Anh), hãy:
1. Tóm tắt ngắn gọn nội dung chính (2-3 câu) bằng cùng ngôn ngữ với tài liệu.
2. Viết một câu hỏi mẫu mà đoạn văn này có thể trả lời (target question).

Đoạn văn:
---
{text}
---

Trả lời theo đúng format sau (giữ nguyên nhãn):
SUMMARY: <tóm tắt>
TARGET_QUESTION: <câu hỏi mẫu>"""
    msg = prompt.format(text=parent.page_content[:4000])
    try:
        resp = llm.invoke(msg)
        content = resp.content if hasattr(resp, "content") else str(resp)
        summary = ""
        target_question = ""
        for line in content.strip().split("\n"):
            if line.upper().startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()
            elif line.upper().startswith("TARGET_QUESTION:"):
                target_question = line.split(":", 1)[1].strip()
        return summary or content[:200], target_question or "Nội dung đoạn văn là gì?"
    except Exception:
        _ingest_logger.warning(
            "LLM failed to generate summary/target_question for chunk (source: %s). "
            "Falling back to text truncation.",
            parent.metadata.get("source", "unknown"),
            exc_info=True,
        )
        return parent.page_content[:200], "Nội dung đoạn văn là gì?"


# --- Collection name & Qdrant client ---

def sanitize_collection_name(name: str) -> str:
    """Normalize collection name: 3-63 chars, alphanumeric/underscore/hyphen."""
    if not name or not str(name).strip():
        return "rag_chatbot"
    s = str(name).strip()
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "rag_chatbot"
    if len(s) < 3:
        s = s + "xx"[: 3 - len(s)]
    if not s[0].isalnum():
        s = "c" + s
    if not s[-1].isalnum():
        s = s + "1"
    return s[:63]



def _ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """Create collection if missing; recreate if schema is incompatible or corrupted."""
    try:
        info = client.get_collection(collection_name)
        dense_cfg = None
        vectors_cfg = getattr(info.config.params, "vectors", None)
        if isinstance(vectors_cfg, dict):
            dense_cfg = vectors_cfg.get("dense")
        else:
            dense_cfg = getattr(vectors_cfg, "dense", None)
        if dense_cfg and getattr(dense_cfg, "size", None) == vector_size:
            return
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    except Exception:
        ...

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(on_disk=False),
            ),
        },
    )


def _noop_progress(step: str, msg: str, current: int = 0, total: int = 0) -> None:
    pass


# --- Parallel embedding ---

def _parallel_embed(
    embeddings,
    texts: list[str],
    batch_size: int = 64,
    max_workers: int = 4,
    progress_fn: Optional[Callable[[str, str, int, int], None]] = None,
) -> list[list[float]]:
    """Embed texts in parallel batches using a thread pool."""
    if not texts:
        return []

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    total_batches = len(batches)

    if total_batches <= 1:
        if progress_fn:
            progress_fn("embed", f"Embedding {len(texts)} chunks...", 0, 1)
        result = embeddings.embed_documents(texts)
        if progress_fn:
            progress_fn("embed", f"Embedded {len(result)} chunks", 1, 1)
        return result

    all_embeddings: list[Optional[list[list[float]]]] = [None] * total_batches
    completed = 0

    if progress_fn:
        progress_fn("embed", f"Embedding {len(texts)} chunks in {total_batches} batches...", 0, total_batches)

    def _embed_batch(idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
        import time as _time
        for attempt in range(5):
            try:
                return idx, embeddings.embed_documents(batch)
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 2 ** attempt
                    _time.sleep(wait)
                    continue
                raise
        return idx, embeddings.embed_documents(batch)

    with ThreadPoolExecutor(max_workers=min(max_workers, total_batches)) as pool:
        futures = {
            pool.submit(_embed_batch, i, batch): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            idx, batch_result = future.result()
            all_embeddings[idx] = batch_result
            completed += 1
            if progress_fn:
                progress_fn("embed", f"Batch {completed}/{total_batches} done", completed, total_batches)

    result: list[list[float]] = []
    for batch_result in all_embeddings:
        result.extend(batch_result)
    return result


# --- Full file ingestion ---

def ingest_file(
    file_path: Path,
    *,
    collection_name: str = "rag_chatbot",
    on_progress: Optional[Callable[[str, str, int, int], None]] = None,
    skip_summary: bool = False,
) -> dict[str, Any]:
    """
    Load file, build parent/child chunks, add metadata, store in Qdrant.
    Returns dict with num_parents, num_children, etc. on_progress(step, msg, current, total) is optional.
    If skip_summary is True, do not call LLM for summary/target_question (faster, uses truncation instead).
    """
    progress = on_progress or _noop_progress
    collection_name = sanitize_collection_name(collection_name)
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required. Set it in .env or environment.")

    file_path = Path(file_path)
    progress("load", "Loading file...", 0, 1)
    documents = load_document(file_path, settings=settings)
    if not documents:
        return {"error": "No content extracted", "file": str(file_path)}
    progress("load", f"Loaded {len(documents)} page(s)", 1, 1)

    if settings.embedding_model.startswith("text-embedding"):
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"trust_remote_code": True, "device": "cpu"},
        )
        _fix_position_ids(embeddings.client)
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )
    configured_strategy = getattr(settings, "chunk_strategy", "semantic_tokens") or "semantic_tokens"
    configured_strategy = configured_strategy.lower()
    if configured_strategy == "auto":
        effective_strategy = auto_select_chunk_strategy(documents)
        progress("semantic", f"Auto chunking (strategy: {effective_strategy})...", 0, 0)
    else:
        effective_strategy = configured_strategy
        progress("semantic", f"Chunking ({effective_strategy})...", 0, 0)

    parents = build_parent_chunks(
        documents=documents,
        settings=settings,
        embeddings=embeddings,
    )
    progress("semantic", f"Created {len(parents)} parent chunks", len(parents), len(parents))

    all_children: list[Document] = []
    parent_meta: dict[str, dict] = {}

    for i, parent in enumerate(parents):
        progress("metadata", f"Summary for parent {i + 1}/{len(parents)}...", i + 1, len(parents))
        parent_id = hashlib.sha256(
            (parent.page_content + str(i)).encode()
        ).hexdigest()[:16]
        if skip_summary:
            summary = (parent.page_content[:200] + "...") if len(parent.page_content) > 200 else parent.page_content
            target_question = "Nội dung đoạn văn là gì?"
        else:
            summary, target_question = add_parent_metadata(parent, llm)

        parent_meta[parent_id] = {
            "content": parent.page_content,
            "summary": summary,
            "target_question": target_question,
            "source": parent.metadata.get("source", file_path.name),
        }

        children = parent_to_children_dynamic(parent, settings, effective_strategy=effective_strategy)
        for j, child in enumerate(children):
            child.metadata["parent_id"] = parent_id
            child.metadata["summary"] = summary
            child.metadata["target_question"] = target_question
            child.metadata["parent_content"] = parent.page_content
            child.metadata["source"] = parent.metadata.get("source", file_path.name)
            all_children.append(child)

    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    client = get_qdrant_client(settings.persist_dir)

    # Merge parent metadata with existing (supports multi-file ingestion)
    parents_path = settings.persist_dir / f"{collection_name}_parents.json"
    existing_parents: dict = {}
    if parents_path.exists():
        try:
            with open(parents_path, "r", encoding="utf-8") as f:
                existing_parents = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_parents = {}
    existing_parents.update(parent_meta)
    with open(parents_path, "w", encoding="utf-8") as f:
        json.dump(existing_parents, f, ensure_ascii=False, indent=2)

    child_texts = [c.page_content for c in all_children]
    child_metadatas = []
    for c in all_children:
        m = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in c.metadata.items()}
        if len(m.get("parent_content", "")) > 30000:
            m["parent_content"] = m["parent_content"][:30000] + "..."
        child_metadatas.append(m)

    child_embeddings = _parallel_embed(
        embeddings,
        child_texts,
        batch_size=getattr(settings, "embed_batch_size", 64),
        max_workers=getattr(settings, "embed_max_workers", 4),
        progress_fn=progress,
    )

    vector_size = len(child_embeddings[0]) if child_embeddings else 0
    if vector_size == 0:
        return {
            "file": str(file_path),
            "collection_name": collection_name,
            "num_parents": len(parents),
            "num_children": 0,
            "total_chunks_in_db": 0,
        }

    # --- Build / merge sparse vocab & vectors (Python-side sparse index) ---
    progress("sparse", "Building sparse vocab and vectors...", 0, 1)
    vocab_path = settings.persist_dir / f"{collection_name}_sparse_vocab.json"
    existing_vocab: dict[str, int] = {}
    if vocab_path.exists():
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                existing_vocab = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_vocab = {}

    new_vocab = _build_vocab(child_texts)
    merged_vocab = dict(existing_vocab)
    next_idx = max(merged_vocab.values(), default=-1) + 1
    for token in new_vocab:
        if token not in merged_vocab:
            merged_vocab[token] = next_idx
            next_idx += 1

    sparse_vectors: list[tuple[list[int], list[float]]] = [
        _text_to_sparse_vector(t, merged_vocab) for t in child_texts
    ]
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(merged_vocab, f, ensure_ascii=False)
    progress("sparse", f"Built sparse vocab ({len(merged_vocab)} terms)", 1, 1)

    _ensure_collection(client, collection_name, vector_size)

    # Determine ID offset so new points don't overwrite existing ones
    try:
        id_offset = client.count(collection_name=collection_name, exact=True).count
    except Exception:
        id_offset = 0

    # Build Qdrant points (dense + sparse vectors)
    points: list[qmodels.PointStruct] = []
    for i in range(len(child_texts)):
        indices, values = sparse_vectors[i]
        point_kwargs: dict[str, Any] = {
            "id": id_offset + i,
            "vector": {"dense": child_embeddings[i]},
            "payload": {**child_metadatas[i], "text": child_texts[i]},
        }
        if indices:
            sparse_vec = qmodels.SparseVector(indices=indices, values=values)
            point_kwargs["vector"]["sparse"] = sparse_vec
        points.append(qmodels.PointStruct(**point_kwargs))

    progress("qdrant", f"Writing {len(points)} chunks to Qdrant...", 0, 1)
    client.upsert(collection_name=collection_name, points=points)

    total_in_db = client.count(collection_name=collection_name, exact=True).count
    progress("done", f"Done: {len(parents)} parents, {len(all_children)} children", 1, 1)

    return {
        "file": str(file_path),
        "collection_name": collection_name,
        "num_parents": len(parents),
        "num_children": len(all_children),
        "total_chunks_in_db": total_in_db,
    }


def ingest_directory(
    dir_path: Path,
    collection_name: str = "rag_chatbot",
    on_progress: Optional[Callable[[str, str, int, int], None]] = None,
) -> list[dict[str, Any]]:
    """Ingest all PDF and DOCX files in a directory."""
    results = []
    for ext in ("*.pdf", "*.docx", "*.doc"):
        for path in Path(dir_path).rglob(ext):
            try:
                r = ingest_file(
                    path,
                    collection_name=collection_name,
                    on_progress=on_progress,
                )
                results.append(r)
            except Exception as e:
                results.append({"file": str(path), "error": str(e)})
    return results
