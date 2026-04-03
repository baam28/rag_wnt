"""Ingestion: load PDF/DOCX, semantic chunking, parent-child hierarchy, metadata (summary, target_question)."""
import base64
import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import tiktoken
from docling.document_converter import DocumentConverter 
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import get_settings
from legal_tokenizer import legal_tokenize as _legal_tokenize
from mongo_client import (
    insert_sparse_vocab_entries,
    load_sparse_bm25_stats,
    load_sparse_vocab_map,
    upsert_sparse_bm25_stats,
    upsert_vector_parents,
)
from utils import fix_position_ids as _fix_position_ids, get_qdrant_client, tokenize_for_sparse as _tokenize_for_sparse

_ingest_logger = logging.getLogger(__name__)

TABLE_START = "<!--TABLE_START-->"
TABLE_END = "<!--TABLE_END-->"


def _normalize_text_for_match(text: str) -> str:
    """Lowercase + remove Vietnamese diacritics for robust substring matching."""
    normalized = unicodedata.normalize("NFD", (text or "").lower())
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip()


def _split_sector_values(legal_sectors: str) -> list[str]:
    raw = (legal_sectors or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split("|") if part.strip()]


def _extract_pharma_tags(legal_sectors: str, keywords: list[str]) -> list[str]:
    sectors = _split_sector_values(legal_sectors)
    if not sectors:
        return []
    normalized_keywords = [_normalize_text_for_match(k) for k in keywords if (k or "").strip()]
    tags: list[str] = []
    for sector in sectors:
        normalized_sector = _normalize_text_for_match(sector)
        if any(k and (k in normalized_sector or normalized_sector in k) for k in normalized_keywords):
            tags.append(sector)
    return tags


def _parse_proxy_issuance_date(date_text: str, date_format: str) -> datetime | None:
    raw = (date_text or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, date_format)
    except ValueError:
        return None


def _proxy_group_key(meta: dict[str, Any]) -> str:
    document_number = str(meta.get("document_number") or "").strip()
    legal_type = _normalize_text_for_match(str(meta.get("legal_type") or ""))
    if document_number:
        return f"num::{legal_type}::{_normalize_text_for_match(document_number)}"
    title = _normalize_text_for_match(str(meta.get("title") or ""))
    return f"title::{legal_type}::{title}"


def _proxy_rank_tuple(
    *,
    is_vbhn: bool,
    issuance_date: datetime,
    has_issuing_authority: bool,
    has_url: bool,
) -> tuple[int, int, int, int]:
    return (
        1 if is_vbhn else 0,
        issuance_date.toordinal(),
        1 if has_issuing_authority else 0,
        1 if has_url else 0,
    )


def _sanitize_metadata_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        cleaned: list[Any] = []
        for item in value:
            cleaned.append(_sanitize_metadata_value(item))
        return cleaned
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _sanitize_text_for_api(text: str) -> str:
    """Normalize text to be safe for JSON APIs (remove NUL and invalid UTF-8 surrogates)."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.replace("\x00", "")
    cleaned = cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return "".join(
        ch for ch in cleaned
        if (ch in "\n\r\t") or (unicodedata.category(ch) not in {"Cc", "Cs"})
    )


def _load_existing_hf_ids(collection_name: str) -> set[str]:
    """Load existing hf_id values from a Qdrant collection to support resumable ingestion."""
    existing: set[str] = set()
    try:
        client = get_qdrant_client()
    except Exception:
        return existing

    try:
        client.get_collection(collection_name)
    except Exception:
        return existing

    offset = None
    try:
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=["hf_id"],
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                raw = payload.get("hf_id")
                if raw is None:
                    continue
                hf_id = str(raw).strip()
                if hf_id:
                    existing.add(hf_id)
            if next_offset is None:
                break
            offset = next_offset
    except Exception:
        return existing

    return existing


# --- Sparse vector helpers (for Qdrant native sparse) ---


def _build_vocab(
    texts: list[str],
    max_vocab_size: int = 100_000,
) -> tuple[dict[str, int], dict[str, float], float]:
    """Build vocabulary and BM25 statistics from corpus.

    Returns:
        vocab:    token → index mapping (top-N by frequency)
        idf_map:  token → Robertson IDF value
        avgdl:    average document length in tokens
    """
    from collections import Counter
    from math import log

    doc_token_lists: list[list[str]] = [_legal_tokenize(t) for t in texts]
    N = len(doc_token_lists)

    corpus_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()
    total_tokens = 0
    for token_list in doc_token_lists:
        corpus_counter.update(token_list)
        df_counter.update(set(token_list))
        total_tokens += len(token_list)

    avgdl = total_tokens / N if N > 0 else 1.0

    vocab: dict[str, int] = {}
    for token, _ in corpus_counter.most_common(max_vocab_size):
        vocab[token] = len(vocab)

    # Robertson IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    idf_map: dict[str, float] = {}
    for token in vocab:
        df = df_counter.get(token, 0)
        idf_map[token] = log((N - df + 0.5) / (df + 0.5) + 1.0)

    return vocab, idf_map, avgdl


def _text_to_sparse_vector(
    text: str,
    vocab: dict[str, int],
    idf_map: dict[str, float] | None = None,
    avgdl: float = 1.0,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[list[int], list[float]]:
    """Convert text to (indices, values) for Qdrant SparseVector using BM25.

    When ``idf_map`` is provided, applies the full Okapi BM25 formula with
    document-length normalization.  Falls back to ``1 + log(tf)`` when
    ``idf_map`` is None or empty.
    """
    from math import log
    tokens = _legal_tokenize(text)
    if not tokens:
        return [], []
    dl = len(tokens)
    tf_raw: dict[int, int] = {}
    for t in tokens:
        idx = vocab.get(t)
        if idx is None:
            continue
        tf_raw[idx] = tf_raw.get(idx, 0) + 1

    scores: dict[int, float] = {}
    if idf_map:
        idx_to_token = {v: k for k, v in vocab.items()}
        for idx, raw_tf in tf_raw.items():
            token = idx_to_token.get(idx, "")
            idf = idf_map.get(token, 1.0)
            tf_norm = (raw_tf * (k1 + 1)) / (raw_tf + k1 * (1 - b + b * dl / avgdl))
            scores[idx] = idf * tf_norm
    else:
        for idx, raw_tf in tf_raw.items():
            scores[idx] = 1.0 + log(raw_tf)

    indices = sorted(scores.keys())
    values = [float(scores[i]) for i in indices]
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


# --- Legal article parsing ---

# Matches Vietnamese and English article markers at the start of a line.
# Group 1: keyword (Điều/Article/Section), Group 2: number, Group 3: optional title text.
_ARTICLE_RE = re.compile(
    r"(?im)^(điều|article|section)\s+(\d+)[.:]?\s*(.*)",
)

# Matches chapter/part markers.
# Group 1: keyword, Group 2: Roman or Arabic numeral, Group 3: optional title text.
_CHAPTER_RE = re.compile(
    r"(?im)^(chương|chapter|part|phần)\s+([IVXLCDM]+|\d+)[.:]?\s*(.*)",
)

# Matches numbered clause lines inside an article, e.g. "1. " or "  2. "
_CLAUSE_RE = re.compile(r"(?m)^[ \t]*(\d+)\.\s+")


def _detect_legal_articles(text: str) -> bool:
    """Return True if *text* contains at least 5 article markers (Điều/Article/Section)."""
    return len(_ARTICLE_RE.findall(text)) >= 5


def _split_legal_articles(text: str) -> list[dict]:
    """Parse *text* into a list of article dicts.

    Each dict has:
    - ``article_number`` (str): e.g. "15"
    - ``article_title``  (str): heading text after "Điều 15."
    - ``chapter_number`` (str): current chapter numeral, e.g. "II"
    - ``chapter_title``  (str): current chapter heading text
    - ``body``           (str): full article text including its header line

    Text before the first article is returned as a preamble entry with
    ``article_number="0"``.
    """
    lines = text.splitlines(keepends=True)

    current_chapter_num = ""
    current_chapter_title = ""

    articles: list[dict] = []
    buf: list[str] = []
    current_article_num = "0"
    current_article_title = ""
    current_chapter_at_start = ""
    current_chapter_title_at_start = ""

    def _flush(buf, art_num, art_title, chap_num, chap_title):
        body = "".join(buf).strip()
        if body:
            articles.append({
                "article_number": art_num,
                "article_title": art_title,
                "chapter_number": chap_num,
                "chapter_title": chap_title,
                "body": body,
            })

    for line in lines:
        stripped = line.strip()

        chap_m = _CHAPTER_RE.match(stripped)
        if chap_m:
            _flush(buf, current_article_num, current_article_title,
                   current_chapter_at_start, current_chapter_title_at_start)
            buf = [line]
            current_chapter_num = chap_m.group(2).strip()
            current_chapter_title = chap_m.group(3).strip()
            current_article_num = "0"
            current_article_title = ""
            current_chapter_at_start = current_chapter_num
            current_chapter_title_at_start = current_chapter_title
            continue

        art_m = _ARTICLE_RE.match(stripped)
        if art_m:
            _flush(buf, current_article_num, current_article_title,
                   current_chapter_at_start, current_chapter_title_at_start)
            buf = [line]
            current_article_num = art_m.group(2).strip()
            current_article_title = art_m.group(3).strip()
            current_chapter_at_start = current_chapter_num
            current_chapter_title_at_start = current_chapter_title
            continue

        buf.append(line)

    _flush(buf, current_article_num, current_article_title,
           current_chapter_at_start, current_chapter_title_at_start)

    return articles


def _split_clauses(text: str) -> list[str]:
    """Split article text on numbered clause markers (e.g. '1. ', '2. ').

    Falls back to sentence splitting when fewer than 2 clauses are found.
    """
    matches = list(_CLAUSE_RE.finditer(text))
    if len(matches) < 2:
        return _split_sentences(text)

    chunks: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


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


def article_parent_chunks(
    documents: list[Document],
    parent_target_tokens: int,
    embeddings: Any,
) -> list[Document]:
    """Split legal documents into parent chunks, one per article (Điều/Article/Section).

    Each article is kept as a single parent chunk.  If an article exceeds
    *parent_target_tokens*, it is further split with RecursiveCharacterTextSplitter
    while preserving the article-level metadata on every sub-chunk.

    Falls back to :func:`semantic_parent_chunks` when no article markers are found.
    """
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_target_tokens * 4,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    parents: list[Document] = []
    for doc in documents:
        text = doc.page_content or ""
        if not _detect_legal_articles(text):
            parents.extend(
                semantic_parent_chunks([doc], parent_target_tokens, embeddings)
            )
            continue

        article_dicts = _split_legal_articles(text)
        for art in article_dicts:
            if art["article_number"] == "0" and len(art["body"].split()) < 30:
                continue
            article_meta = {
                **doc.metadata,
                "article_number": art["article_number"],
                "article_title": art["article_title"],
                "chapter_number": art["chapter_number"],
                "chapter_title": art["chapter_title"],
                "chunk_strategy": "article",
            }
            if count_tokens(art["body"]) > parent_target_tokens:
                article_doc = Document(page_content=art["body"], metadata=article_meta)
                sub_splits = char_splitter.split_documents([article_doc])
                parents.extend(sub_splits)
            else:
                parents.append(Document(page_content=art["body"], metadata=article_meta))

    return parents


def build_parent_chunks(
    documents: list[Document],
    settings,
    embeddings: Any,
) -> list[Document]:
    """Build parent chunks using article-based chunking.

    Each Điều/Article/Section becomes one parent chunk.  Documents without
    article markers fall back to semantic (header-based) splitting.
    """
    return article_parent_chunks(
        documents,
        parent_target_tokens=getattr(settings, "article_max_parent_tokens", 1200),
        embeddings=embeddings,
    )


def parent_to_children_dynamic(parent: Document, settings, *, effective_strategy: Optional[str] = None) -> list[Document]:
    """Split a parent chunk into child chunks by clause (Khoản).

    Splits on numbered clause markers (1. 2. …); falls back to sentences
    when fewer than 2 clauses are found.
    """
    text = parent.page_content or ""
    chunks = _split_clauses(text)

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

    texts = [_sanitize_text_for_api(t) for t in texts]
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    total_batches = len(batches)

    # OpenAI embeddings client is not safely parallelized in this flow and may emit malformed requests.
    # Force sequential batching for OpenAI models to avoid intermittent 400 invalid JSON errors.
    model_name = str(getattr(embeddings, "model", "") or "")
    if model_name.startswith("text-embedding"):
        max_workers = 1

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
                msg = str(e).lower()
                if "rate_limit" in msg or "429" in msg:
                    wait = 2 ** attempt
                    _time.sleep(wait)
                    continue
                if "parse the json body" in msg or "not valid json" in msg:
                    fixed_batch = [_sanitize_text_for_api(item) for item in batch]
                    _time.sleep(1)
                    try:
                        return idx, embeddings.embed_documents(fixed_batch)
                    except Exception:
                        raise
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


def _upsert_points_in_batches(
    client: QdrantClient,
    collection_name: str,
    points: list[qmodels.PointStruct],
    batch_size: int,
    progress_fn: Optional[Callable[[str, str, int, int], None]] = None,
) -> None:
    if not points:
        return
    effective_batch = max(1, int(batch_size or 128))
    total = len(points)
    batches = (total + effective_batch - 1) // effective_batch
    for i in range(batches):
        start = i * effective_batch
        end = min(start + effective_batch, total)
        if progress_fn:
            progress_fn("qdrant", f"Writing chunks {start + 1}-{end}/{total}...", i, batches)
        client.upsert(collection_name=collection_name, points=points[start:end])
    if progress_fn:
        progress_fn("qdrant", f"Writing chunks {total}/{total} complete", batches, batches)


# --- Full file ingestion ---

def ingest_documents(
    documents: list[Document],
    *,
    source_label: str,
    collection_name: str = "rag_chatbot",
    on_progress: Optional[Callable[[str, str, int, int], None]] = None,
    skip_summary: bool = False,
) -> dict[str, Any]:
    """Ingest pre-loaded documents into Qdrant using the standard parent-child pipeline."""
    progress = on_progress or _noop_progress
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required. Set it in .env or environment.")

    if not documents:
        return {"error": "No content extracted", "file": source_label}

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
    progress("semantic", "Chunking (article)...", 0, 0)
    effective_strategy = "article"

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
            "source": parent.metadata.get("source", source_label),
        }

        children = parent_to_children_dynamic(parent, settings, effective_strategy=effective_strategy)
        for child in children:
            # Preserve all parent metadata (including HF metadata like hf_id, title, legal_sectors, etc.)
            # then add chunking-specific metadata
            for key, value in parent.metadata.items():
                if key not in child.metadata or key == "source":
                    child.metadata[key] = value
            child.metadata["parent_id"] = parent_id
            child.metadata["summary"] = summary
            child.metadata["target_question"] = target_question
            all_children.append(child)

    client = get_qdrant_client()
    upsert_vector_parents(collection_name, parent_meta)

    child_texts = [_sanitize_text_for_api(c.page_content) for c in all_children]
    child_metadatas = []
    for c in all_children:
        m = {k: _sanitize_metadata_value(v) for k, v in c.metadata.items()}
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
            "file": source_label,
            "collection_name": collection_name,
            "num_parents": len(parents),
            "num_children": 0,
            "total_chunks_in_db": 0,
        }

    progress("sparse", "Building sparse vocab and vectors...", 0, 1)
    existing_vocab = load_sparse_vocab_map(collection_name)

    cfg = get_settings()
    new_vocab, idf_map, avgdl = _build_vocab(child_texts)
    merged_vocab = dict(existing_vocab)
    next_idx = max(merged_vocab.values(), default=-1) + 1
    added_vocab: dict[str, int] = {}
    for token in new_vocab:
        if token not in merged_vocab:
            merged_vocab[token] = next_idx
            added_vocab[token] = next_idx
            next_idx += 1

    sparse_vectors: list[tuple[list[int], list[float]]] = [
        _text_to_sparse_vector(t, merged_vocab, idf_map, avgdl, cfg.bm25_k1, cfg.bm25_b)
        for t in child_texts
    ]
    insert_sparse_vocab_entries(collection_name, added_vocab)
    upsert_sparse_bm25_stats(collection_name, avgdl, idf_map)
    progress("sparse", f"Built sparse vocab ({len(merged_vocab)} terms)", 1, 1)

    _ensure_collection(client, collection_name, vector_size)

    try:
        id_offset = client.count(collection_name=collection_name, exact=True).count
    except Exception:
        id_offset = 0

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
    _upsert_points_in_batches(
        client,
        collection_name=collection_name,
        points=points,
        batch_size=getattr(settings, "qdrant_upsert_batch_size", 128),
        progress_fn=progress,
    )

    total_in_db = client.count(collection_name=collection_name, exact=True).count
    progress("done", f"Done: {len(parents)} parents, {len(all_children)} children", 1, 1)

    return {
        "file": source_label,
        "collection_name": collection_name,
        "num_parents": len(parents),
        "num_children": len(all_children),
        "total_chunks_in_db": total_in_db,
    }

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
    file_path = Path(file_path)
    progress("load", "Loading file...", 0, 1)
    settings = get_settings()
    documents = load_document(file_path, settings=settings)
    return ingest_documents(
        documents,
        source_label=str(file_path),
        collection_name=collection_name,
        on_progress=on_progress,
        skip_summary=skip_summary,
    )


def ingest_hf_legal_dataset(
    *,
    collection_name: str,
    dataset_repo: str | None = None,
    metadata_config: str | None = None,
    content_config: str | None = None,
    split: str | None = None,
    only_pharma_related: bool = False,
    row_filter: Optional[Callable[[dict], bool]] = None,
    mode: str = "default",
    write_quarantine: bool = False,
    quarantine_collection_name: str | None = None,
    limit: int | None = None,
    start_offset: int = 0,
    batch_size: int | None = None,
    skip_summary: bool | None = None,
    on_progress: Optional[Callable[[str, str, int, int], None]] = None,
) -> dict[str, Any]:
    """Ingest legal documents from Hugging Face dataset with metadata payload preservation."""
    progress = on_progress or _noop_progress
    settings = get_settings()
    mode_normalized = (mode or "default").strip().lower()
    if mode_normalized not in {"default", "proxy_strict"}:
        raise ValueError("mode must be one of: default, proxy_strict")
    use_proxy_mode = mode_normalized == "proxy_strict"

    repo_id = dataset_repo or settings.hf_legal_dataset_repo
    metadata_name = metadata_config or settings.hf_legal_metadata_config
    content_name = content_config or settings.hf_legal_content_config
    split_name = split or settings.hf_legal_split
    batch_size = max(1, int(batch_size or settings.hf_legal_batch_size))
    skip_summary_effective = settings.hf_legal_skip_summary_default if skip_summary is None else bool(skip_summary)
    collection_name = sanitize_collection_name(collection_name)
    quarantine_collection_effective = ""
    if write_quarantine:
        quarantine_collection_effective = sanitize_collection_name(
            quarantine_collection_name or f"{collection_name}_quarantine"
        )

    existing_hf_ids = _load_existing_hf_ids(collection_name)
    if existing_hf_ids:
        progress("dataset", f"Found {len(existing_hf_ids)} existing hf_id entries; resume mode enabled", 0, 1)

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets package is required for Hugging Face ingestion. Install with pip install datasets") from e

    progress("dataset", f"Loading metadata split '{metadata_name}/{split_name}'...", 0, 1)
    metadata_rows = load_dataset(repo_id, metadata_name, split=split_name)
    progress("dataset", "Building metadata index...", 0, len(metadata_rows) if hasattr(metadata_rows, "__len__") else 1)

    metadata_by_id: dict[str, dict[str, Any]] = {}
    pharma_keywords = settings.hf_pharma_sector_keywords
    for idx, row in enumerate(metadata_rows):
        if idx < max(0, start_offset):
            continue
        doc_id = str(row.get("id", "")).strip()
        if not doc_id:
            continue
        legal_sectors = str(row.get("legal_sectors") or "")
        pharma_tags = _extract_pharma_tags(legal_sectors, pharma_keywords)
        metadata_by_id[doc_id] = {
            "hf_id": doc_id,
            "document_number": str(row.get("document_number") or ""),
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
            "legal_type": str(row.get("legal_type") or ""),
            "legal_sectors": legal_sectors,
            "issuing_authority": str(row.get("issuing_authority") or ""),
            "issuance_date": str(row.get("issuance_date") or ""),
            "signers": str(row.get("signers") or ""),
            "source_dataset": repo_id,
            "is_pharma_related": bool(pharma_tags),
            "pharma_tags": "|".join(pharma_tags),
            "source": str(row.get("title") or row.get("document_number") or f"hf_doc_{doc_id}"),
            "file_path": f"hf://{repo_id}/{doc_id}",
        }
        if idx % 5000 == 0 and idx > 0:
            progress("dataset", f"Indexed metadata rows: {idx}", idx, len(metadata_rows) if hasattr(metadata_rows, "__len__") else idx)

    proxy_selected_doc_ids: set[str] = set()
    proxy_reject_reason_by_doc_id: dict[str, str] = {}
    rejected_legal_type = 0
    rejected_bad_date = 0
    rejected_older_version = 0

    if use_proxy_mode:
        valid_legal_types = {
            _normalize_text_for_match(legal_type)
            for legal_type in settings.hf_proxy_legal_type_whitelist
            if (legal_type or "").strip()
        }
        vbhn_label = _normalize_text_for_match("Văn bản hợp nhất")
        best_by_group: dict[str, tuple[str, tuple[int, int, int, int]]] = {}
        candidate_doc_ids: list[str] = []

        for doc_id, meta in metadata_by_id.items():
            legal_type_normalized = _normalize_text_for_match(str(meta.get("legal_type") or ""))
            if legal_type_normalized not in valid_legal_types:
                proxy_reject_reason_by_doc_id[doc_id] = "rejected_legal_type"
                continue

            if row_filter is not None and not row_filter(meta):
                proxy_reject_reason_by_doc_id[doc_id] = "skipped_non_pharma"
                continue
            if row_filter is None and only_pharma_related and not meta.get("is_pharma_related"):
                proxy_reject_reason_by_doc_id[doc_id] = "skipped_non_pharma"
                continue

            issuance_date = _parse_proxy_issuance_date(str(meta.get("issuance_date") or ""), settings.hf_proxy_date_format)
            if issuance_date is None:
                proxy_reject_reason_by_doc_id[doc_id] = "rejected_bad_date"
                continue

            candidate_doc_ids.append(doc_id)
            group_key = _proxy_group_key(meta)
            rank = _proxy_rank_tuple(
                is_vbhn=(settings.hf_proxy_prefer_vbhn and legal_type_normalized == vbhn_label),
                issuance_date=issuance_date,
                has_issuing_authority=bool(str(meta.get("issuing_authority") or "").strip()),
                has_url=bool(str(meta.get("url") or "").strip()),
            )
            current = best_by_group.get(group_key)
            if current is None or rank > current[1]:
                best_by_group[group_key] = (doc_id, rank)

        proxy_selected_doc_ids = {doc_id for doc_id, _ in best_by_group.values()}
        for doc_id in candidate_doc_ids:
            if doc_id in proxy_selected_doc_ids:
                proxy_reject_reason_by_doc_id[doc_id] = "selected"
            else:
                proxy_reject_reason_by_doc_id[doc_id] = "rejected_older_version"

    progress("dataset", f"Loading content split '{content_name}/{split_name}'...", 0, 1)
    content_rows = load_dataset(repo_id, content_name, split=split_name)

    ingested_docs = 0
    skipped_no_metadata = 0
    skipped_empty_content = 0
    skipped_non_pharma = 0
    total_parents = 0
    total_children = 0
    total_chunks_in_db = 0
    batches_processed = 0
    docs_batch: list[Document] = []
    quarantine_ingested_docs = 0
    quarantine_batches_processed = 0
    quarantine_num_parents = 0
    quarantine_num_children = 0
    quarantine_total_chunks_in_db = 0
    quarantine_docs_batch: list[Document] = []

    for idx, row in enumerate(content_rows):
        if idx < max(0, start_offset):
            continue

        doc_id = str(row.get("id", "")).strip()
        if not doc_id:
            continue
        if doc_id in existing_hf_ids:
            continue
        meta = metadata_by_id.get(doc_id)
        if not meta:
            skipped_no_metadata += 1
            continue

        if use_proxy_mode:
            reason = proxy_reject_reason_by_doc_id.get(doc_id)
            if reason is None:
                reason = "rejected_legal_type"
            if reason != "selected":
                if reason == "skipped_non_pharma":
                    skipped_non_pharma += 1
                elif reason == "rejected_bad_date":
                    rejected_bad_date += 1
                elif reason == "rejected_older_version":
                    rejected_older_version += 1
                elif reason == "rejected_legal_type":
                    rejected_legal_type += 1
                else:
                    skipped_non_pharma += 1

                if quarantine_collection_effective:
                    rejected_content = str(row.get("content") or "").strip()
                    if rejected_content:
                        rejected_meta = {
                            **meta,
                            "validity_mode": "proxy",
                            "validity_confidence": "low",
                            "is_latest_in_group": False,
                            "proxy_reject_reason": reason,
                        }
                        quarantine_docs_batch.append(
                            Document(page_content=rejected_content, metadata=rejected_meta)
                        )
                        quarantine_ingested_docs += 1

                        if len(quarantine_docs_batch) >= batch_size:
                            quarantine_batches_processed += 1
                            progress(
                                "dataset",
                                f"Ingesting HF quarantine batch {quarantine_batches_processed} ({len(quarantine_docs_batch)} docs)...",
                                quarantine_ingested_docs,
                                quarantine_ingested_docs,
                            )
                            quarantine_result = ingest_documents(
                                quarantine_docs_batch,
                                source_label=f"{repo_id}:{split_name}:quarantine_batch_{quarantine_batches_processed}",
                                collection_name=quarantine_collection_effective,
                                skip_summary=skip_summary_effective,
                            )
                            if "error" in quarantine_result:
                                return quarantine_result
                            quarantine_num_parents += int(quarantine_result.get("num_parents", 0))
                            quarantine_num_children += int(quarantine_result.get("num_children", 0))
                            quarantine_total_chunks_in_db = int(
                                quarantine_result.get("total_chunks_in_db", quarantine_total_chunks_in_db)
                            )
                            quarantine_docs_batch = []
                continue
        else:
            if row_filter is not None:
                if not row_filter(meta):
                    skipped_non_pharma += 1
                    continue
            elif only_pharma_related and not meta.get("is_pharma_related"):
                skipped_non_pharma += 1
                continue

        content = str(row.get("content") or "").strip()
        if not content:
            skipped_empty_content += 1
            continue

        if use_proxy_mode:
            legal_type_normalized = _normalize_text_for_match(str(meta.get("legal_type") or ""))
            confidence = "medium" if legal_type_normalized == _normalize_text_for_match("Văn bản hợp nhất") else "low"
            meta = {
                **meta,
                "validity_mode": "proxy",
                "validity_confidence": confidence,
                "is_latest_in_group": True,
            }

        docs_batch.append(Document(page_content=content, metadata=dict(meta)))
        ingested_docs += 1

        if limit and ingested_docs >= limit:
            should_flush = True
        else:
            should_flush = len(docs_batch) >= batch_size

        if should_flush and docs_batch:
            batches_processed += 1
            progress("dataset", f"Ingesting HF batch {batches_processed} ({len(docs_batch)} docs)...", ingested_docs, limit or ingested_docs)
            result = ingest_documents(
                docs_batch,
                source_label=f"{repo_id}:{split_name}:batch_{batches_processed}",
                collection_name=collection_name,
                skip_summary=skip_summary_effective,
            )
            if "error" in result:
                return result
            total_parents += int(result.get("num_parents", 0))
            total_children += int(result.get("num_children", 0))
            total_chunks_in_db = int(result.get("total_chunks_in_db", total_chunks_in_db))
            docs_batch = []

        if limit and ingested_docs >= limit:
            break

    if docs_batch:
        batches_processed += 1
        progress("dataset", f"Ingesting final HF batch {batches_processed} ({len(docs_batch)} docs)...", ingested_docs, limit or ingested_docs)
        result = ingest_documents(
            docs_batch,
            source_label=f"{repo_id}:{split_name}:batch_{batches_processed}",
            collection_name=collection_name,
            skip_summary=skip_summary_effective,
        )
        if "error" in result:
            return result
        total_parents += int(result.get("num_parents", 0))
        total_children += int(result.get("num_children", 0))
        total_chunks_in_db = int(result.get("total_chunks_in_db", total_chunks_in_db))

    if quarantine_docs_batch:
        quarantine_batches_processed += 1
        progress(
            "dataset",
            f"Ingesting final HF quarantine batch {quarantine_batches_processed} ({len(quarantine_docs_batch)} docs)...",
            quarantine_ingested_docs,
            quarantine_ingested_docs,
        )
        quarantine_result = ingest_documents(
            quarantine_docs_batch,
            source_label=f"{repo_id}:{split_name}:quarantine_batch_{quarantine_batches_processed}",
            collection_name=quarantine_collection_effective,
            skip_summary=skip_summary_effective,
        )
        if "error" in quarantine_result:
            return quarantine_result
        quarantine_num_parents += int(quarantine_result.get("num_parents", 0))
        quarantine_num_children += int(quarantine_result.get("num_children", 0))
        quarantine_total_chunks_in_db = int(
            quarantine_result.get("total_chunks_in_db", quarantine_total_chunks_in_db)
        )

    progress("done", f"HF ingest complete: {ingested_docs} docs", ingested_docs, limit or ingested_docs)
    return {
        "dataset": repo_id,
        "mode": mode_normalized,
        "write_quarantine": bool(quarantine_collection_effective),
        "quarantine_collection_name": quarantine_collection_effective or None,
        "collection_name": collection_name,
        "ingested_docs": ingested_docs,
        "batches_processed": batches_processed,
        "num_parents": total_parents,
        "num_children": total_children,
        "total_chunks_in_db": total_chunks_in_db,
        "skip_summary": skip_summary_effective,
        "only_pharma_related": only_pharma_related,
        "skipped_no_metadata": skipped_no_metadata,
        "skipped_empty_content": skipped_empty_content,
        "skipped_non_pharma": skipped_non_pharma,
        "rejected_legal_type": rejected_legal_type,
        "rejected_bad_date": rejected_bad_date,
        "rejected_older_version": rejected_older_version,
        "quarantine_ingested_docs": quarantine_ingested_docs,
        "quarantine_batches_processed": quarantine_batches_processed,
        "quarantine_num_parents": quarantine_num_parents,
        "quarantine_num_children": quarantine_num_children,
        "quarantine_total_chunks_in_db": quarantine_total_chunks_in_db,
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
