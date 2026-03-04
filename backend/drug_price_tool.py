"""Tool: fetch real-time medicine prices from major Vietnamese pharmacy chains.

Strategy: scrape product data directly from pharmacy websites (Long Châu SSR)
instead of relying on search engines, which are unreliable for Vietnamese queries.
"""

import json
import re
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

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}


# ---------------------------------------------------------------------------
# Direct pharmacy scraping
# ---------------------------------------------------------------------------

def _format_vnd(raw_price: int) -> str:
    """Format integer price (e.g. 1800) to '1.800 VND'."""
    return f"{raw_price:,.0f} VND".replace(",", ".")


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


def _scrape_longchau(drug_name: str) -> list[dict[str, Any]]:
    """Scrape Long Châu search results via SSR __NEXT_DATA__."""
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

    results = []
    for p in products:
        name = p.get("name") or p.get("webName") or ""
        slug = p.get("slug", "")
        price_info = p.get("price", {})
        final = (
            price_info.get("discount", {}).get("finalPrice")
            or price_info.get("price")
            or 0
        )
        unit = price_info.get("measureUnitName", "")
        is_rx = bool(p.get("isPrescription")) or bool(
            _RX_PATTERN.search(name + " " + str(p.get("category", "")))
        )
        is_inventory = price_info.get("isInventory", True)

        if not name or final == 0:
            continue

        results.append({
            "drug_name": _title_case(name),
            "price": _format_vnd(final),
            "price_raw": final,
            "unit": unit,
            "source_name": "Nhà thuốc Long Châu",
            "source_url": f"{_LONGCHAU_BASE}/{slug}" if slug else url,
            "is_prescription": is_rx,
            "in_stock": is_inventory,
        })

    return results


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def _empty_result(drug_name: str) -> dict[str, Any]:
    return {
        "drug_name": drug_name,
        "prices": [],
        "price_range": "Không tìm thấy giá",
        "is_prescription": False,
        "notes": "",
    }


def get_vietnam_drug_price(drug_name: str) -> dict[str, Any]:
    """
    Fetch real-time medicine prices from Vietnamese pharmacy chains.

    Returns a dict with:
      drug_name, prices[], price_range, is_prescription, notes,
      disclaimer, source_urls
    """
    products = _scrape_longchau(drug_name)

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
            "unit": p["unit"],
            "source_name": p["source_name"],
            "source_url": p["source_url"],
            "drug_name": p["drug_name"],
            "in_stock": p["in_stock"],
        })
        raw_vals.append(p["price_raw"])
        source_urls.append(p["source_url"])

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
