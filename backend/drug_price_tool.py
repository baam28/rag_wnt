"""Tool: fetch real-time medicine prices from Nhà thuốc Long Châu.

Ethical scraping compliance
---------------------------
robots.txt (https://www.nhathuoclongchau.com.vn/robots.txt):
    User-agent: *
    Disallow:
All paths are explicitly permitted for all crawlers.

We additionally follow these best practices:
  - Identify ourselves with an honest User-Agent string (not disguised as a browser).
  - Send a single HTTP request per user query (no recursive crawling).
  - Respect the server with a reasonable timeout and no retry flood.
  - Cache results in memory for PRICE_CACHE_TTL seconds to avoid hammering
    the server when users ask about the same drug multiple times.
  - Never store, republish, or sell the scraped data; it is used only to
    answer an in-session user query in real time.

Data source: Long Châu SSR __NEXT_DATA__ JSON embedded in the search page.
"""

import json
import logging
import re
import time
import threading
from collections import defaultdict
from typing import Any, Tuple
from urllib.parse import quote_plus

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Price-query detection (used by the supervisor / router to auto-route)
# ---------------------------------------------------------------------------

_PRICE_KEYWORDS = [
    "giá thuốc", "giá bán", "giá của", "bao nhiêu tiền",
    "thuốc .+ giá", "giá .+ bao nhiêu", "mua .+ bao nhiêu",
    "tra cứu giá", "price of",
    r"giá\s+.+",
    r".+\s+giá\s*",
    r"bao nhiêu\s+.+",
    r".+\s+bao nhiêu",
]
_PRICE_RE = re.compile("|".join(_PRICE_KEYWORDS), re.IGNORECASE)

_STRIP_PHRASES = [
    "cho tôi biết", "cho biết", "tra cứu giá", "tra cứu",
    "giá thuốc", "giá bán", "giá của", "giá",
    "bao nhiêu tiền", "là bao nhiêu", "bao nhiêu",
    "thuốc", "mua", "ở đâu", "hiện tại", "bây giờ",
    "tìm", "xem", "kiểm tra",
]


def detect_price_query(question: str) -> Tuple[bool, str]:
    """Return (is_price_query, extracted_drug_name)."""
    q = question.strip()
    if not _PRICE_RE.search(q):
        return False, ""
    drug = q.lower()
    drug = re.sub(r"[?!.,;:]+", " ", drug)
    for phrase in sorted(_STRIP_PHRASES, key=len, reverse=True):
        drug = re.sub(re.escape(phrase), " ", drug)
    drug = " ".join(drug.split()).strip()
    if len(drug) < 2:
        return True, q.strip()
    return True, drug


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LONGCHAU_BASE = "https://nhathuoclongchau.com.vn"

DISCLAIMER = (
    "Lưu ý: Giá thuốc có thể thay đổi tùy theo thời điểm, địa điểm và nhà thuốc. "
    "Thông tin trên chỉ mang tính tham khảo. "
    "Vui lòng liên hệ nhà thuốc hoặc dược sĩ để được tư vấn chính xác về giá và cách sử dụng thuốc."
)

PRESCRIPTION_MESSAGE = (
    "Thuốc này thuộc loại thuốc kê đơn (Rx). Giá thuốc kê đơn được quản lý "
    "và thường không được niêm yết công khai trên các trang bán lẻ. "
    "Vui lòng liên hệ nhà thuốc hoặc bác sĩ để biết giá chính xác."
)

_RX_PATTERN = re.compile(
    r"thuốc\s*kê\s*đơn|cần\s*toa|bán\s*theo\s*đơn|prescription",
    re.IGNORECASE,
)

# Honest User-Agent: identifies this as an automated tool, not a browser.
# Long Châu's robots.txt allows all crawlers; we still want to be transparent.
_HTTP_HEADERS = {
    "User-Agent": (
        "RAGChatbot-PriceLookup/1.0 "
        "(educational healthcare assistant; single-request per query; "
        "contact: see project repo)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

_NON_DRUG_QUERY_KEYWORDS = ("trà", "sữa rửa mặt", "tẩy trang", "mỹ phẩm", "cleansing")
_DRUG_CATEGORY_INDICATOR = "thuốc"

# ---------------------------------------------------------------------------
# In-memory price cache (avoids re-scraping the same drug within a session)
# ---------------------------------------------------------------------------

PRICE_CACHE_TTL = 300  # seconds (5 minutes)

_price_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_price_cache_lock = threading.Lock()


def _cache_get(key: str) -> dict[str, Any] | None:
    with _price_cache_lock:
        entry = _price_cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > PRICE_CACHE_TTL:
            del _price_cache[key]
            return None
        return value


def _cache_set(key: str, value: dict[str, Any]) -> None:
    with _price_cache_lock:
        _price_cache[key] = (time.monotonic(), value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_vnd(raw_price: int) -> str:
    return f"{raw_price:,.0f} VND".replace(",", ".")


def _product_category_str(p: dict[str, Any]) -> str:
    cat = p.get("category") or p.get("categoryName") or p.get("categories") or ""
    if isinstance(cat, dict):
        name = cat.get("name") or cat.get("title") or ""
        path = cat.get("path") or cat.get("breadcrumb") or ""
        if isinstance(path, list):
            path = " ".join(str(x) for x in path)
        return f"{name} {path}".strip()
    if isinstance(cat, list):
        return " ".join(str(x) for x in cat)
    return str(cat)


def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    norm = _normalize(text)
    return norm.split() if norm else []


def _is_relevant_product(query: str, product_name: str) -> bool:
    """Heuristic: keep only products whose names are close to the query."""
    q_tokens = _tokens(query)
    name_tokens = _tokens(product_name)
    if not q_tokens or not name_tokens:
        return True
    if len(q_tokens) >= 2:
        match_count = sum(
            1 for qt in q_tokens
            if any(qt in nt or nt in qt for nt in name_tokens)
        )
        return match_count >= min(len(q_tokens), 2)
    q = q_tokens[0]
    if any(q in nt or nt in q for nt in name_tokens):
        return True
    if len(q) >= 5:
        stem = q[:4]
        return any(stem in nt for nt in name_tokens)
    return False


def _title_case(name: str) -> str:
    words = name.split()
    out = []
    for w in words:
        if re.match(r"^\d", w):
            out.append(w.lower())
        elif w.isupper() and len(w) > 1:
            out.append(w.capitalize())
        else:
            out.append(w)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Long Châu scraper
# ---------------------------------------------------------------------------

def _scrape_longchau(drug_name: str, drug_only: bool = True) -> list[dict[str, Any]]:
    """Scrape Long Châu search results via SSR __NEXT_DATA__ JSON.

    Makes a single GET request to the search page and parses the embedded
    JSON — no session cookies, no JavaScript execution, no recursive links.

    Long Châu robots.txt:  User-agent: * / Disallow: (all paths permitted).
    """
    url = f"{_LONGCHAU_BASE}/tim-kiem?s={quote_plus(drug_name)}"
    try:
        resp = httpx.get(url, headers=_HTTP_HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning(
                "Long Châu search returned HTTP %d for query '%s'.",
                resp.status_code, drug_name,
            )
            return []
    except httpx.HTTPError:
        logger.warning("Long Châu HTTP error while searching for '%s'.", drug_name, exc_info=True)
        return []

    match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        resp.text,
        re.DOTALL,
    )
    if not match:
        logger.warning("Long Châu: __NEXT_DATA__ not found in response for '%s'.", drug_name)
        return []

    try:
        nd = json.loads(match.group(1))
    except json.JSONDecodeError:
        logger.warning("Long Châu: failed to parse __NEXT_DATA__ JSON for '%s'.", drug_name)
        return []

    products = (
        nd.get("props", {})
        .get("pageProps", {})
        .get("initProducts", {})
        .get("products", [])
    )

    if drug_only:
        products = [
            p for p in products
            if _DRUG_CATEGORY_INDICATOR in _product_category_str(p).lower()
        ]

    products = [
        p for p in products
        if _is_relevant_product(drug_name, p.get("name") or p.get("webName") or "")
    ]

    results = []
    for p in products:
        name = p.get("name") or p.get("webName") or ""
        slug = p.get("slug", "")
        is_rx = bool(p.get("isPrescription")) or bool(
            _RX_PATTERN.search(name + " " + str(p.get("category", "")))
        )
        source_url = f"{_LONGCHAU_BASE}/{slug}" if slug else url

        price_entries: list[dict[str, Any]] = []
        multi = p.get("prices") or p.get("priceList")
        if not isinstance(multi, list) and isinstance(p.get("price"), list):
            multi = p.get("price")
        if isinstance(multi, list) and len(multi) > 0:
            for entry in multi:
                if not isinstance(entry, dict):
                    continue
                disc = entry.get("discount") or {}
                final = disc.get("finalPrice") or entry.get("price") or 0
                unit = entry.get("measureUnitName") or entry.get("measureUnit") or ""
                if final and isinstance(final, (int, float)) and final > 0:
                    price_entries.append({
                        "final": int(final),
                        "unit": unit,
                        "is_inventory": entry.get("isInventory", True),
                    })
        if not price_entries:
            price_info = p.get("price", {})
            if isinstance(price_info, dict):
                disc = price_info.get("discount") or {}
                final = disc.get("finalPrice") or price_info.get("price") or 0
                unit = price_info.get("measureUnitName") or ""
                is_inventory = price_info.get("isInventory", True)
                if final and isinstance(final, (int, float)) and final > 0:
                    price_entries.append({
                        "final": int(final),
                        "unit": unit,
                        "is_inventory": is_inventory,
                    })

        if not name or not price_entries:
            continue

        for pe in price_entries:
            results.append({
                "drug_name": _title_case(name),
                "price": _format_vnd(pe["final"]),
                "price_raw": pe["final"],
                "unit": pe["unit"],
                "source_name": "Nhà thuốc Long Châu",
                "source_url": source_url,
                "is_prescription": is_rx,
                "in_stock": pe["is_inventory"],
            })

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _empty_result(drug_name: str) -> dict[str, Any]:
    return {
        "drug_name": drug_name,
        "prices": [],
        "drugs": [],
        "price_range": "Không tìm thấy giá",
        "is_prescription": False,
        "notes": "",
    }


def _drug_only_from_query(drug_name: str, question: str | None) -> bool:
    if not question:
        return True
    q_lower = question.lower()
    drug_lower = (drug_name or "").lower()
    if _DRUG_CATEGORY_INDICATOR in q_lower:
        return True
    for kw in _NON_DRUG_QUERY_KEYWORDS:
        if kw in drug_lower or kw in q_lower:
            return False
    return True


def get_vietnam_drug_price(drug_name: str, question: str | None = None) -> dict[str, Any]:
    """Fetch real-time medicine prices from Nhà thuốc Long Châu.

    Robots.txt compliance: Long Châu's robots.txt (User-agent: * / Disallow:)
    explicitly permits all automated access.  We send a single request per
    query and cache results for PRICE_CACHE_TTL seconds.

    Returns a dict with:
      drug_name, prices[], drugs[], price_range, is_prescription, notes,
      disclaimer, source_urls
    """
    cache_key = f"{drug_name.lower()}|{bool(question and 'thuốc' in question.lower())}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("Price cache hit for '%s'.", drug_name)
        return cached

    drug_only = _drug_only_from_query(drug_name, question)
    products = _scrape_longchau(drug_name, drug_only=drug_only)

    if not products:
        result = _empty_result(drug_name)
        result["disclaimer"] = DISCLAIMER
        result["source_urls"] = []
        _cache_set(cache_key, result)
        return result

    all_rx = all(p["is_prescription"] for p in products)
    if all_rx:
        result = {
            "drug_name": drug_name,
            "prices": [],
            "drugs": [],
            "price_range": "Thuốc kê đơn – giá không niêm yết công khai",
            "is_prescription": True,
            "notes": PRESCRIPTION_MESSAGE,
            "disclaimer": DISCLAIMER,
            "source_urls": [p["source_url"] for p in products],
        }
        _cache_set(cache_key, result)
        return result

    prices_list = []
    raw_vals = []
    source_urls = []
    for p in products:
        prices_list.append({
            "price": p["price"],
            "price_raw": p["price_raw"],
            "unit": p["unit"],
            "source_name": p["source_name"],
            "source_url": p["source_url"],
            "drug_name": p["drug_name"],
            "in_stock": p["in_stock"],
        })
        raw_vals.append(p["price_raw"])
        source_urls.append(p["source_url"])

    by_drug: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prices_list:
        by_drug[item["drug_name"]].append(item)
    drugs = []
    for name, options in by_drug.items():
        options_sorted = sorted(options, key=lambda x: x["price_raw"])
        cheapest = options_sorted[0]
        drugs.append({
            "drug_name": name,
            "options": [
                {
                    "unit": o["unit"],
                    "price": o["price"],
                    "price_raw": o["price_raw"],
                    "source_name": o["source_name"],
                    "source_url": o["source_url"],
                }
                for o in options_sorted
            ],
            "cheapest": {
                "unit": cheapest["unit"],
                "price": cheapest["price"],
                "source_name": cheapest["source_name"],
                "source_url": cheapest["source_url"],
            },
        })

    low = _format_vnd(min(raw_vals))
    high = _format_vnd(max(raw_vals))
    common_unit = products[0]["unit"] or "đơn vị"
    price_range = (
        f"{low} – {high} / {common_unit}" if low != high
        else f"{low} / {common_unit}"
    )

    variants = [p["drug_name"] for p in products]
    notes = (
        f"Tìm thấy {len(products)} sản phẩm: {', '.join(variants)}"
        if len(products) > 1 else ""
    )

    result = {
        "drug_name": drug_name,
        "prices": prices_list,
        "drugs": drugs,
        "price_range": price_range,
        "is_prescription": False,
        "notes": notes,
        "disclaimer": DISCLAIMER,
        "source_urls": list(set(source_urls)),
    }
    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# LangChain Tool wrapper
# ---------------------------------------------------------------------------

@tool
def vietnam_drug_price_tool(drug_name: str) -> str:
    """Look up current retail prices for a medicine at Nhà thuốc Long Châu.
    Input: the medicine name in Vietnamese or its generic/brand name."""

    result = get_vietnam_drug_price(drug_name)

    lines = [f"Giá thuốc: {result.get('drug_name', drug_name)}"]

    if result.get("is_prescription"):
        lines.append(f"\n{result.get('notes', PRESCRIPTION_MESSAGE)}")
    else:
        lines.append(f"Khoảng giá: {result.get('price_range', 'Không tìm thấy')}")
        for p in result.get("prices", []):
            name = p.get("source_name", "")
            price = p.get("price", "N/A")
            unit = p.get("unit", "")
            lines.append(f"  - {name}: {price}/{unit}" if unit else f"  - {name}: {price}")
        if result.get("notes"):
            lines.append(f"\nGhi chú: {result['notes']}")

    urls = result.get("source_urls", [])
    if urls:
        lines.append("\nNguồn:")
        for u in urls[:5]:
            lines.append(f"  {u}")

    lines.append(f"\n{DISCLAIMER}")
    return "\n".join(lines)
