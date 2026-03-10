"""Tool: fetch real-time medicine prices from major Vietnamese pharmacy chains.

Strategy: scrape product data directly from pharmacy websites (Long Châu SSR)
instead of relying on search engines, which are unreliable for Vietnamese queries.
"""

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Tuple
from urllib.parse import quote_plus

import httpx
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Price-query detection (used by /ask to auto-route)
# ---------------------------------------------------------------------------

_PRICE_KEYWORDS = [
    "giá thuốc", "giá bán", "giá của", "bao nhiêu tiền",
    "thuốc .+ giá", "giá .+ bao nhiêu", "mua .+ bao nhiêu",
    "tra cứu giá", "price of",
    # Broader: trigger without "thuốc" (e.g. "giá panadol", "panadol giá bao nhiêu")
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
_ANKHANG_BASE = "https://www.nhathuocankhang.com"
_PHARMACITY_BASE = "https://www.pharmacity.vn"

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

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

# When search term contains one of these, do not restrict to drug category (user may want tea, washer, etc.)
_NON_DRUG_QUERY_KEYWORDS = ("trà", "sữa rửa mặt", "tẩy trang", "mỹ phẩm", "cleansing")
_DRUG_CATEGORY_INDICATOR = "thuốc"


# ---------------------------------------------------------------------------
# Direct pharmacy scraping
# ---------------------------------------------------------------------------

def _format_vnd(raw_price: int) -> str:
    """Format integer price (e.g. 1800) to '1.800 VND'."""
    return f"{raw_price:,.0f} VND".replace(",", ".")


def _product_category_str(p: dict[str, Any]) -> str:
    """Return a single string from product category (may be object or path) for allowlist check."""
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
    """Lowercase and strip punctuation/extra whitespace for relevance checks."""
    if not text:
        return ""
    text = text.lower()
    # Replace non-word characters with space
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    norm = _normalize(text)
    return norm.split() if norm else []


def _is_relevant_product(query: str, product_name: str) -> bool:
    """Heuristic to decide if a Long Châu product is relevant to the query."""
    q_tokens = _tokens(query)
    name_tokens = _tokens(product_name)
    if not q_tokens or not name_tokens:
        # If we can't confidently decide, keep the product rather than drop everything.
        return True

    # Brand-like queries with 2+ tokens: require most/all tokens to appear.
    if len(q_tokens) >= 2:
        match_count = 0
        for qt in q_tokens:
            for nt in name_tokens:
                if qt in nt or nt in qt:
                    match_count += 1
                    break
        # Require at least 2 matching tokens or all tokens if fewer.
        required = min(len(q_tokens), 2)
        return match_count >= required

    # Single-token queries (e.g. "panadol", "crila", "magnesium")
    q = q_tokens[0]
    # Allow direct substring or token-level match.
    for nt in name_tokens:
        if q in nt or nt in q:
            return True
    # Allow simple stem-based match to include obvious generics (e.g. Panactol for Panadol).
    if len(q) >= 5:
        stem = q[:4]
        for nt in name_tokens:
            if stem in nt:
                return True
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


def _scrape_longchau(drug_name: str, drug_only: bool = True) -> list[dict[str, Any]]:
    """Scrape Long Châu search results via SSR __NEXT_DATA__.
    When drug_only is True, only include products whose category indicates medicine (e.g. thuốc).
    """
    url = f"{_LONGCHAU_BASE}/tim-kiem?s={quote_plus(drug_name)}"
    try:
        resp = httpx.get(url, headers=_HTTP_HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return []
    except httpx.HTTPError:
        return []

    match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        resp.text,
        re.DOTALL,
    )
    if not match:
        return []

    try:
        nd = json.loads(match.group(1))
    except json.JSONDecodeError:
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

    # Apply relevance filtering so we only keep products whose names are close
    # to the query (brand and closely related generics), after category filter.
    products = [
        p
        for p in products
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
                if final and (isinstance(final, (int, float)) and final > 0):
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
                if final and (isinstance(final, (int, float)) and final > 0):
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


def _scrape_ankhang(drug_name: str, drug_only: bool = True) -> list[dict[str, Any]]:
    """Scrape Nhà thuốc An Khang search results (best-effort, HTML only).

    NOTE: This is a minimal implementation; the HTML structure may change.
    On failure, it returns an empty list so other stores can still be used.
    """
    search_url = f"{_ANKHANG_BASE}/tim-kiem?kw={quote_plus(drug_name)}"
    try:
        resp = httpx.get(search_url, headers=_HTTP_HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return []
    except httpx.HTTPError:
        return []

    html = resp.text
    results: list[dict[str, Any]] = []

    # Very loose heuristic: look for product blocks containing the query and a price-like pattern.
    # This avoids depending on exact CSS classes.
    pattern = re.compile(
        r'<a[^>]+href="(?P<href>/[^"]+)"[^>]*>(?P<name>[^<]+)</a>(?P<tail>.{0,400}?)',
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(html):
        name = re.sub(r"\\s+", " ", m.group("name")).strip()
        if not name:
            continue
        # Ensure this looks related to the query
        if drug_name.lower() not in name.lower():
            continue
        tail = m.group("tail")
        price_match = re.search(r"([0-9][0-9\\.]{3,})\\s*[₫đ]", tail)
        if not price_match:
            continue
        price_str = price_match.group(1).replace(".", "")
        if not price_str.isdigit():
            continue
        price_raw = int(price_str)
        if price_raw <= 0:
            continue
        unit = ""
        unit_match = re.search(r"(/\\s*(Hộp|Vỉ|Viên|Ống|Gói))", tail, re.IGNORECASE)
        if unit_match:
            unit = unit_match.group(2)
        href = m.group("href")
        if not href.startswith("http"):
            href = f"{_ANKHANG_BASE}{href}"
        results.append(
            {
                "drug_name": _title_case(name),
                "price": _format_vnd(price_raw),
                "price_raw": price_raw,
                "unit": unit,
                "source_name": "Nhà thuốc An Khang",
                "source_url": href,
                "is_prescription": False,
                "in_stock": True,
            }
        )

    return results


def _scrape_pharmacity(drug_name: str, drug_only: bool = True) -> list[dict[str, Any]]:
    """Scrape Pharmacity search results (best-effort, HTML only).

    NOTE: Pharmacity is highly dynamic; this parser may return an empty list
    if the site relies heavily on client-side rendering.
    """
    search_url = f"{_PHARMACITY_BASE}/search?query={quote_plus(drug_name)}"
    try:
        resp = httpx.get(search_url, headers=_HTTP_HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return []
    except httpx.HTTPError:
        return []

    html = resp.text
    results: list[dict[str, Any]] = []

    # Heuristic: look for product links under /san-pham/ or /p/ with a nearby price.
    pattern = re.compile(
        r'<a[^>]+href="(?P<href>/(?:san-pham|p)/[^"]+)"[^>]*>(?P<name>[^<]+)</a>(?P<tail>.{0,400}?)',
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(html):
        name = re.sub(r"\\s+", " ", m.group("name")).strip()
        if not name:
            continue
        if drug_name.lower() not in name.lower():
            continue
        tail = m.group("tail")
        price_match = re.search(r"([0-9][0-9\\.]{3,})\\s*[₫đ]", tail)
        if not price_match:
            continue
        price_str = price_match.group(1).replace(".", "")
        if not price_str.isdigit():
            continue
        price_raw = int(price_str)
        if price_raw <= 0:
            continue
        unit = ""
        unit_match = re.search(r"(/\\s*(Hộp|Vỉ|Viên|Ống|Gói))", tail, re.IGNORECASE)
        if unit_match:
            unit = unit_match.group(2)
        href = m.group("href")
        url = f"{_PHARMACITY_BASE}{href}"
        results.append(
            {
                "drug_name": _title_case(name),
                "price": _format_vnd(price_raw),
                "price_raw": price_raw,
                "unit": unit,
                "source_name": "Nhà thuốc Pharmacity",
                "source_url": url,
                "is_prescription": False,
                "in_stock": True,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Main public function
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
    """True if we should restrict results to drug category (allowlist by thuốc)."""
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
    """
    Fetch real-time medicine prices from Vietnamese pharmacy chains.

    When question is provided and indicates drug intent (e.g. contains "thuốc"),
    only products in a medicine category are returned. When the user asks for
    tea, washer, etc., all results are returned.

    Returns a dict with:
      drug_name, prices[], drugs[], price_range, is_prescription, notes,
      disclaimer, source_urls
    """
    drug_only = _drug_only_from_query(drug_name, question)

    # For now, only use Long Châu as the data source. Other scrapers are
    # defined but not invoked here.
    products: list[dict[str, Any]] = _scrape_longchau(drug_name, drug_only=drug_only)

    if not products:
        result = _empty_result(drug_name)
        result["disclaimer"] = DISCLAIMER
        result["source_urls"] = []
        return result

    all_rx = all(p["is_prescription"] for p in products)
    if all_rx:
        return {
            "drug_name": drug_name,
            "prices": [],
            "drugs": [],
            "price_range": "Thuốc kê đơn – giá không niêm yết công khai",
            "is_prescription": True,
            "notes": PRESCRIPTION_MESSAGE,
            "disclaimer": DISCLAIMER,
            "source_urls": [p["source_url"] for p in products],
        }

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

    # Group by drug_name, sort by price_raw per group, build drugs (options + cheapest)
    by_drug: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prices_list:
        by_drug[item["drug_name"]].append(item)
    drugs = []
    for name, options in by_drug.items():
        options_sorted = sorted(options, key=lambda x: x["price_raw"])
        opts = [
            {
                "unit": o["unit"],
                "price": o["price"],
                "price_raw": o["price_raw"],
                "source_name": o["source_name"],
                "source_url": o["source_url"],
            }
            for o in options_sorted
        ]
        cheapest = options_sorted[0]
        drugs.append({
            "drug_name": name,
            "options": opts,
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

    return {
        "drug_name": drug_name,
        "prices": prices_list,
        "drugs": drugs,
        "price_range": price_range,
        "is_prescription": False,
        "notes": notes,
        "disclaimer": DISCLAIMER,
        "source_urls": list(set(source_urls)),
    }


# ---------------------------------------------------------------------------
# LangChain Tool wrapper
# ---------------------------------------------------------------------------

@tool
def vietnam_drug_price_tool(drug_name: str) -> str:
    """Look up current retail prices for a medicine at major Vietnamese
    pharmacy chains (Long Châu).
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
            # Link is already shown in the UI (↗), so avoid duplicating it here
            if unit:
                lines.append(f"  - {name}: {price}/{unit}")
            else:
                lines.append(f"  - {name}: {price}")
        if result.get("notes"):
            lines.append(f"\nGhi chú: {result['notes']}")

    urls = result.get("source_urls", [])
    if urls:
        lines.append("\nNguồn:")
        for u in urls[:5]:
            lines.append(f"  {u}")

    lines.append(f"\n{DISCLAIMER}")
    return "\n".join(lines)
