"""Automatic resolution of Vietnamese legal cross-references in LLM answers.

After the first generate_answer() call, the /ask handler passes the answer text
here. We detect patterns like "quy định tại điểm a, b khoản 1 Điều 13 của <doc>"
and retrieve the actual content of those referenced clauses so the answer can be
re-generated with the full information.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Pattern A — canonical: "quy định tại [điểm X, Y] [khoản N] Điều M [của <doc>]"
_PAT_FULL = re.compile(
    r"quy\s+định\s+tại\s+"
    r"(?:(?:các\s+)?điểm\s+([\w,\s/àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵýỷỹ"
    r"ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸHOẶCVÀ]+?)\s+)?"
    r"(?:khoản\s+(\d+)\s+)?"
    r"điều\s+(\d+)"
    r"(?:\s+(?:của|thuộc)\s+([^\.,;\n]{5,80}))?",
    re.IGNORECASE | re.UNICODE,
)

# Pattern B — "theo [quy định tại] Điều M [khoản N]"
_PAT_THEO = re.compile(
    r"theo\s+(?:quy\s+định\s+(?:tại\s+)?)?"
    r"(?:khoản\s+(\d+)\s+)?điều\s+(\d+)"
    r"(?:\s+khoản\s+(\d+))?",
    re.IGNORECASE | re.UNICODE,
)

# Pattern C — "căn cứ [theo] [quy định tại] Điều M"
_PAT_CAN_CU = re.compile(
    r"căn\s+cứ\s+(?:theo\s+)?(?:quy\s+định\s+tại\s+)?điều\s+(\d+)",
    re.IGNORECASE | re.UNICODE,
)

# Pattern D — parenthetical "(Điều M [khoản N])"
_PAT_PAREN = re.compile(
    r"\(\s*(?:xem\s+)?điều\s+(\d+)(?:\s+khoản\s+(\d+))?\s*\)",
    re.IGNORECASE | re.UNICODE,
)

# Pattern E — bare citation: "[các] [điểm X, Y] [khoản N] Điều M"
# Catches e.g. "điểm a, b hoặc d khoản 1 Điều 13" without a leading trigger word.
_VIET_CHARS = (
    r"àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵýỷỹ"
    r"ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ"
)
_PAT_BARE = re.compile(
    r"(?:(?:các\s+)?điểm\s+([\w,\s/hoặcvà" + _VIET_CHARS + r"]+?)\s+)?"
    r"khoản\s+(\d+)\s+điều\s+(\d+)"
    r"(?:\s+(?:của|thuộc)\s+([^\.,;\n]{5,80}))?",
    re.IGNORECASE | re.UNICODE,
)

# Noise words to strip from doc hints (dates, version markers, etc.)
_DOC_NOISE = re.compile(
    r"\b(?:ngày|số|ban\s+hành|có\s+hiệu\s+lực|hợp\s+nhất\s+Luật|hợp\s+nhất\s+Nghị)\b.*$",
    re.IGNORECASE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_points(raw: str | None) -> list[str]:
    """Extract single-letter point labels from a raw string like 'a, b hoặc d'."""
    if not raw:
        return []
    tokens = re.split(r"[,\s]+(?:và|hoặc)?\s*|[,\s]+", raw.strip())
    return [t.strip() for t in tokens if re.match(r"^[a-zđ]$", t.strip(), re.IGNORECASE)]


def _clean_doc_hint(raw: str | None) -> str:
    """Strip noise tail from a document name hint."""
    if not raw:
        return ""
    cleaned = _DOC_NOISE.sub("", raw.strip()).strip().rstrip(".,;:")
    return cleaned


def _build_query(article: str, clause: str | None, points: list[str], doc_hint: str) -> str:
    """Build a retrieval query string for a detected cross-reference."""
    parts: list[str] = []
    if points:
        parts.append(f"điểm {', '.join(points)}")
    if clause:
        parts.append(f"khoản {clause}")
    parts.append(f"Điều {article}")
    if doc_hint:
        parts.append(doc_hint)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_cross_references(text: str) -> list[dict[str, Any]]:
    """Detect Vietnamese legal cross-references in *text*.

    Returns a list of dicts with keys:
        raw, article, clause, points, doc_hint, query
    De-duplicated by (article, clause) key. Capped at crossref_max_refs.
    """
    from config import get_settings
    settings = get_settings()

    refs: dict[tuple, dict] = {}  # (article, clause) → ref dict

    # Pattern A — most specific, may capture points + clause + doc hint
    for m in _PAT_FULL.finditer(text):
        raw_points, clause, article, doc_hint_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        if not article:
            continue
        art_int = int(article)
        if not (1 <= art_int <= 300):
            continue
        points = _parse_points(raw_points)
        clause_str = clause or ""
        doc_hint = _clean_doc_hint(doc_hint_raw)
        key = (article, clause_str)
        if key not in refs:
            refs[key] = {
                "raw": m.group(0),
                "article": article,
                "clause": clause_str,
                "points": points,
                "doc_hint": doc_hint,
                "query": _build_query(article, clause_str or None, points, doc_hint),
            }

    # Pattern B — "theo Điều M [khoản N]"
    for m in _PAT_THEO.finditer(text):
        # groups: (khoản_before, điều, khoản_after)
        clause_before, article, clause_after = m.group(1), m.group(2), m.group(3)
        if not article:
            continue
        art_int = int(article)
        if not (1 <= art_int <= 300):
            continue
        clause_str = (clause_before or clause_after or "")
        key = (article, clause_str)
        if key not in refs:
            refs[key] = {
                "raw": m.group(0),
                "article": article,
                "clause": clause_str,
                "points": [],
                "doc_hint": "",
                "query": _build_query(article, clause_str or None, [], ""),
            }

    # Pattern C — "căn cứ Điều M"
    for m in _PAT_CAN_CU.finditer(text):
        article = m.group(1)
        if not article:
            continue
        art_int = int(article)
        if not (1 <= art_int <= 300):
            continue
        key = (article, "")
        if key not in refs:
            refs[key] = {
                "raw": m.group(0),
                "article": article,
                "clause": "",
                "points": [],
                "doc_hint": "",
                "query": _build_query(article, None, [], ""),
            }

    # Pattern D — parenthetical "(Điều M [khoản N])"
    for m in _PAT_PAREN.finditer(text):
        article, clause = m.group(1), m.group(2)
        if not article:
            continue
        art_int = int(article)
        if not (1 <= art_int <= 300):
            continue
        clause_str = clause or ""
        key = (article, clause_str)
        if key not in refs:
            refs[key] = {
                "raw": m.group(0),
                "article": article,
                "clause": clause_str,
                "points": [],
                "doc_hint": "",
                "query": _build_query(article, clause_str or None, [], ""),
            }

    # Pattern E — bare citation "điểm X khoản N Điều M" (no trigger word)
    for m in _PAT_BARE.finditer(text):
        raw_points, clause, article, doc_hint_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        if not article:
            continue
        art_int = int(article)
        if not (1 <= art_int <= 300):
            continue
        points = _parse_points(raw_points)
        clause_str = clause or ""
        doc_hint = _clean_doc_hint(doc_hint_raw)
        key = (article, clause_str)
        if key not in refs:
            refs[key] = {
                "raw": m.group(0),
                "article": article,
                "clause": clause_str,
                "points": points,
                "doc_hint": doc_hint,
                "query": _build_query(article, clause_str or None, points, doc_hint),
            }

    all_refs = list(refs.values())
    # Sort by number of mentions in the text before applying the cap so that the most
    # prominent references survive when more than crossref_max_refs are detected.
    if len(all_refs) > settings.crossref_max_refs:
        all_refs.sort(
            key=lambda r: -len(re.findall(rf"[Đđ]iều\s+{re.escape(r['article'])}\b", text, re.UNICODE))
        )
    result = all_refs[: settings.crossref_max_refs]
    if result:
        logger.debug("detect_cross_references: found %d ref(s): %s", len(result), [r["query"] for r in result])
    return result


def resolve_cross_references(
    answer: str,
    legal_collection_name: str,
) -> list[dict[str, Any]]:
    """Detect cross-references in *answer* and retrieve their content.

    Returns a (possibly empty) list of new context dicts that are not already
    present in the original contexts — deduplication is done by the caller via
    _deduplicate_contexts().
    """
    from retriever import retrieve

    refs = detect_cross_references(answer)
    if not refs:
        return []

    queries = [r["query"] for r in refs]
    logger.info("resolve_cross_references: resolving %d ref(s): %s", len(queries), queries)

    all_docs: list[dict[str, Any]] = []
    seen_fingerprints: set[tuple] = set()

    def _fetch(query: str) -> list[dict[str, Any]]:
        try:
            return retrieve(query, [legal_collection_name])
        except Exception:
            logger.exception("resolve_cross_references: retrieve failed for query=%r", query)
            return []

    max_workers = min(len(queries), 3)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch, q) for q in queries]
        for future in as_completed(futures):
            for doc in future.result():
                fp = (doc.get("source", ""), (doc.get("content") or "")[:80])
                if fp not in seen_fingerprints:
                    seen_fingerprints.add(fp)
                    all_docs.append(doc)

    logger.info("resolve_cross_references: retrieved %d unique new chunk(s)", len(all_docs))
    return all_docs
