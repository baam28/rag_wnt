"""
Vietnamese legal-domain tokenizer for BM25/sparse retrieval.

Expands compound legal references (Điều, Khoản, Điểm, document numbers)
into dedicated tokens alongside standard word tokens, improving sparse
retrieval recall for queries containing specific legal citations.

"""

import re
import unicodedata
from typing import Union

# ---------------------------------------------------------------------------
# Vietnamese legal reference patterns
# ---------------------------------------------------------------------------

# Article refs: "Điều 28", "Dieu 28", "điều 28" (with optional sub-clause)
_DIEU_RE = re.compile(
    r'\b(?:đi[eề]u|dieu)\s+(\d+)(?:\s+kho[aả]n\s+(\d+))?',
    re.IGNORECASE | re.UNICODE
)

# Clause refs: "Khoản 1", "khoản 2a"
_KHOAN_RE = re.compile(
    r'\bkho[aả]n\s+(\d+[a-z]?)\b',
    re.IGNORECASE | re.UNICODE
)

# Point refs: "Điểm a", "điểm b)"
_DIEM_RE = re.compile(
    r'\b(?:đi[eể]m|diem)\s+([a-zA-Z])\b',
    re.IGNORECASE | re.UNICODE
)

# Vietnamese document numbers:
# "54/2017/NĐ-CP", "05/2023/TT-BYT", "36/2023/QH15"
_DOC_NUM_RE = re.compile(
    r'\b(\d{1,3})/(\d{4})/(NĐ|ND|TT|QĐ|QD|CT|CV|QH|TTCP|BYT|BCT|BTC|BGDĐT|BGDDT|BKHCN|BCA|BTP|BLĐTBXH|BLDTBXH|BTNMT|BGTVT|BXD|BNN|BCĐ|BCD|VPCP)([-\w.]*)',
    re.IGNORECASE | re.UNICODE
)

# Year standalone (used in document numbers and legal citations)
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')

# Vietnamese legal type keywords (for token expansion)
_LEGAL_TYPE_RE = re.compile(
    r'\b(lu[aậ]t|ngh[iị]\s+đ[iị]nh|th[oô]ng\s+t[uư]|quy[eế]t\s+đ[iị]nh|ch[iỉ]\s+th[iị]|ngh[iị]\s+quy[eế]t|ph[aá]p\s+l[eệ]nh|c[oô]ng\s+v[aă]n|h[uư][oớ]ng\s+d[aẫ]n)\b',
    re.IGNORECASE | re.UNICODE
)

# Standard word tokenizer (2+ char words)
_WORD_RE = re.compile(r'(?u)\b\w\w+\b')

# ---------------------------------------------------------------------------
# Vietnamese stopwords (minimal functional set)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    'và', 'của', 'là', 'trong', 'các', 'có', 'được', 'này', 'đó',
    'với', 'cho', 'về', 'theo', 'tại', 'từ', 'đến', 'bởi', 'bởi vì',
    'khi', 'mà', 'như', 'nếu', 'thì', 'hay', 'hoặc', 'nhưng', 'vì',
    'để', 'do', 'qua', 'sau', 'trước', 'trên', 'dưới', 'ngoài',
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'is', 'at',
    'by', 'for', 'on', 'with', 'as', 'it', 'its', 'be', 'are',
})


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and NFC-normalize Unicode (preserves Vietnamese diacritics)."""
    return unicodedata.normalize('NFC', text.lower())


def _ascii_slug(text: str) -> str:
    """Convert a string to a safe ASCII slug for token construction.
    Replaces non-alphanumeric chars with underscores, strips leading/trailing."""
    slug = re.sub(r'[^a-z0-9]+', '_', _normalize(text))
    return slug.strip('_')


# ---------------------------------------------------------------------------
# Legal reference expansion
# ---------------------------------------------------------------------------

def _expand_legal_refs(text: str) -> list[str]:
    """Extract expanded tokens from Vietnamese legal references."""
    extra: list[str] = []

    # Article refs — also handle embedded clause (Điều 28 khoản 1)
    for m in _DIEU_RE.finditer(text):
        art = m.group(1)
        clause = m.group(2)
        extra.append(f'dieu_{art}')
        if clause:
            extra.append(f'dieu_{art}_khoan_{clause}')
            extra.append(f'khoan_{clause}')

    # Standalone clause refs
    for m in _KHOAN_RE.finditer(text):
        extra.append(f'khoan_{m.group(1).lower()}')

    # Point refs
    for m in _DIEM_RE.finditer(text):
        extra.append(f'diem_{m.group(1).lower()}')

    # Document numbers  e.g. "54/2017/NĐ-CP" → "nd_cp_54_2017", "2017"
    for m in _DOC_NUM_RE.finditer(text):
        num, year, doc_type, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        type_slug = _ascii_slug(doc_type + suffix)
        extra.append(f'{type_slug}_{num}_{year}')
        extra.append(year)

    # Legal type keywords → normalized single tokens
    for m in _LEGAL_TYPE_RE.finditer(text):
        extra.append(_ascii_slug(m.group(0)))

    return extra


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def legal_tokenize(text: str) -> list[str]:
    """Tokenize text with Vietnamese legal reference expansion for BM25.

    Returns word tokens (lowercased, stopwords removed) plus expanded
    legal reference tokens for Điều/Khoản/Điểm/document-number patterns.
    """
    normalized = _normalize(text)
    words = _WORD_RE.findall(normalized)
    tokens = [w for w in words if w not in _STOPWORDS]
    tokens.extend(_expand_legal_refs(text))
    return tokens


def legal_tokenize_query(query: str) -> list[str]:
    """Tokenize a search query with legal reference expansion.

    Identical to legal_tokenize — queries receive the same expansion so
    that a query for 'Điều 28 khoản 1' matches indexed documents.
    """
    return legal_tokenize(query)


def legal_tokenize_corpus(texts: list[str]) -> list[list[str]]:
    """Tokenize a corpus for BM25 indexing.

    Returns list-of-lists compatible with bm25s.BM25.index().
    """
    return [legal_tokenize(t) for t in texts]


def legal_tokenize_queries(queries: Union[str, list[str]]) -> list[list[str]]:
    """Tokenize one or more queries for BM25 retrieval.

    Returns list-of-lists compatible with bm25s.BM25.retrieve().
    """
    if isinstance(queries, str):
        queries = [queries]
    return [legal_tokenize_query(q) for q in queries]


