#!/usr/bin/env python3
"""
Long Chau Drug Price Scraper (v4)
----------------------------------
Scrapes all drugs from nhathuoclongchau.com.vn using the A-Z index.

Discovery:  GET tra-cuu-thuoc-a-z?alphabet=X&page=N  → __NEXT_DATA__ has slug list
Product:    GET each slug URL                         → __NEXT_DATA__ has full product data

No Playwright needed — all data is server-rendered in __NEXT_DATA__.

Field mapping from product __NEXT_DATA__:
  drug_name           ← product.webName
  active_ingredient   ← product.ingredient[].name + shortDescription
  dosage              ← extracted mg/ml amounts from ingredient shortDescriptions
  dosage_form         ← product.dosageForm  (Dạng bào chế)
  therapeutic_class   ← product.indications[].name  (Chỉ định)
  price / unit        ← product.prices[] — prefer "Hộp" level
  package_size        ← product.specification
  company             ← product.producer
  country_of_origin   ← product.brandOrigin
  prescription        ← product.prescription (bool)
  drug_code           ← product.sku

Usage:
    python scripts/scrape_longchau.py                 # Full crawl (~5800 drugs)
    python scripts/scrape_longchau.py --limit 20      # Quick test
    python scripts/scrape_longchau.py --resume        # Resume from progress file
"""

import argparse
import json
import math
import os
import re
import random
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import psycopg

# ─── Config ────────────────────────────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5433/pharmanet")
PROGRESS_FILE = Path(__file__).parent / "scrape_progress.json"
BASE_URL = "https://nhathuoclongchau.com.vn"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9",
}

# All alphabet values used by the A-Z index
AZ_ALPHABETS = [
    "3", "5",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]
ITEMS_PER_PAGE = 20

# ─── Progress ──────────────────────────────────────────────────────────────────
def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"scraped_urls": [], "total_saved": 0}

def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

# ─── HTTP helper ───────────────────────────────────────────────────────────────
def fetch_next_data(url: str) -> Optional[dict]:
    """Fetch a URL and return the parsed __NEXT_DATA__ pageProps, or None on error."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            content = r.read().decode("utf-8")
    except Exception as e:
        print(f"  ⚠️  HTTP error {url}: {e}")
        return None
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', content, re.DOTALL)
    if not m:
        print(f"  ⚠️  No __NEXT_DATA__ at {url}")
        return None
    try:
        return json.loads(m.group(1))["props"]["pageProps"]
    except Exception as e:
        print(f"  ⚠️  JSON parse error at {url}: {e}")
        return None

# ─── Database ──────────────────────────────────────────────────────────────────
def upsert_drug(conn, data: dict):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO drug_list
                (drug_id, drug_name, active_ingredient, dosage, dosage_form,
                 prescription_class, therapeutic_class)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (drug_id) DO UPDATE
            SET drug_name          = EXCLUDED.drug_name,
                active_ingredient  = EXCLUDED.active_ingredient,
                dosage             = EXCLUDED.dosage,
                dosage_form        = EXCLUDED.dosage_form,
                prescription_class = EXCLUDED.prescription_class,
                therapeutic_class  = EXCLUDED.therapeutic_class,
                updated_at         = NOW()
        """, (
            data["drug_code"],
            data["drug_name"],
            data["active_ingredient"],
            data["dosage"],
            data["dosage_form"],
            "Thuốc kê đơn (Rx)" if data["prescription_required"] else "Thuốc không kê đơn (OTC)",
            data["therapeutic_category"],
        ))

        stock_quantity = 0 if random.random() < 0.05 else random.randint(10, 500)
        expiry_date = datetime.now().date() + timedelta(days=random.randint(180, 1000))

        cur.execute("""
            INSERT INTO drug_inventory
                (drug_id, selling_unit, packaging_size, retail_price, stock_quantity, expiry_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (drug_id) DO UPDATE
            SET selling_unit   = EXCLUDED.selling_unit,
                packaging_size = EXCLUDED.packaging_size,
                retail_price   = EXCLUDED.retail_price,
                updated_at     = NOW()
        """, (
            data["drug_code"],
            data["price_per_unit"],
            data["package_size"],
            data["price"],
            stock_quantity,
            expiry_date,
        ))
    conn.commit()

# ─── A-Z Index: collect all product URLs ───────────────────────────────────────
def collect_all_product_urls() -> list:
    """
    Iterate every alphabet letter, paginate through each letter's pages (20/page),
    and collect all product slugs as full URLs.
    Uses plain HTTP — __NEXT_DATA__ is server-rendered, no JS needed.
    """
    all_urls: list = []
    seen: set = set()

    for alphabet in AZ_ALPHABETS:
        print(f"\n  📖 Alphabet '{alphabet}'")

        # Page 1 — also tells us totalCount
        page_props = fetch_next_data(
            f"{BASE_URL}/thuoc/tra-cuu-thuoc-a-z?alphabet={alphabet}&page=1"
        )
        if not page_props:
            print(f"    ⚠️  Skipping.")
            continue

        list_drugs = page_props.get("listDrugs", {})
        total_count = list_drugs.get("totalCount", 0)
        total_pages = math.ceil(total_count / ITEMS_PER_PAGE) if total_count else 1
        print(f"    {total_count} drugs, {total_pages} pages")

        pages_data = [list_drugs]

        for page_num in range(2, total_pages + 1):
            pp = fetch_next_data(
                f"{BASE_URL}/thuoc/tra-cuu-thuoc-a-z?alphabet={alphabet}&page={page_num}"
            )
            if pp:
                pages_data.append(pp.get("listDrugs", {}))
            else:
                print(f"    ⚠️  Failed page {page_num}, skipping.")

        for ld in pages_data:
            for item in ld.get("items", []):
                slug = item.get("slug", "")
                if slug:
                    full_url = f"{BASE_URL}/{slug}"
                    if full_url not in seen:
                        seen.add(full_url)
                        all_urls.append(full_url)

        print(f"    → {len(seen)} unique URLs so far")

    return all_urls

# ─── Product page: extract data from __NEXT_DATA__ ─────────────────────────────
def fetch_product_data(url: str) -> Optional[dict]:
    """
    Fetch a product page and extract all drug data from __NEXT_DATA__.
    No browser/JS needed — Next.js SSR embeds the full product object.
    """
    page_props = fetch_next_data(url)
    if not page_props:
        return None

    product = page_props.get("product")
    if not product:
        return None

    drug_name = (product.get("webName") or "").strip()
    if not drug_name:
        return None

    # ── Active ingredient ──────────────────────────────────────────────────────
    ingredients = product.get("ingredient") or []
    ingredient_parts = []
    for ing in ingredients:
        name = (ing.get("name") or "").strip()
        amount = (ing.get("shortDescription") or "").strip()
        if name:
            ingredient_parts.append(f"{name} {amount}".strip())
    active_ingredient = ", ".join(ingredient_parts)

    # ── Dosage: amounts from ingredient shortDescriptions ─────────────────────
    dosage_matches = re.findall(
        r'\d+(?:[.,]\d+)?\s*(?:mg|g|ml|mcg|iu|UI|%)',
        active_ingredient, re.IGNORECASE
    )
    dosage = ", ".join(dict.fromkeys(dosage_matches))  # deduplicate, preserve order

    # ── Dosage form (Dạng bào chế) ────────────────────────────────────────────
    dosage_form = (product.get("dosageForm") or "").strip()

    # ── Therapeutic category from Chỉ định ────────────────────────────────────
    # 1. Clean indications list
    indications = product.get("indications") or []
    indication_names = [i.get("name", "") for i in indications if i.get("name")]
    therapeutic_category = ", ".join(indication_names)

    # 2. Fall back: deepest category name (e.g. "Thuốc tránh thai")
    if not therapeutic_category:
        categories = product.get("categories") or []
        if categories:
            deepest = max(categories, key=lambda c: c.get("level", 0))
            therapeutic_category = deepest.get("name", "")

    # 3. Fall back: first sentence of Chỉ định from usage HTML
    if not therapeutic_category:
        usage_html = product.get("usage") or ""
        usage_text = re.sub(r"<[^>]+>", " ", usage_html).strip()
        usage_text = re.sub(r"\s+", " ", usage_text)
        m = re.search(r"Chỉ định\s+(.+?)(?:\.|$)", usage_text)
        if m:
            therapeutic_category = m.group(1).strip()

    # ── Price: prefer "Hộp" (level 1), fall back to isSellDefault, then first ─
    prices = product.get("prices") or []
    chosen_price = None
    for p in prices:
        if p.get("measureUnitName") in ("Hộp", "Lọ", "Túi"):
            chosen_price = p
            break
    if not chosen_price:
        for p in prices:
            if p.get("isSellDefault"):
                chosen_price = p
                break
    if not chosen_price and prices:
        chosen_price = prices[0]

    price = float(chosen_price["price"]) if chosen_price and chosen_price.get("price") else 0.0
    price_per_unit = (chosen_price.get("measureUnitName") or "") if chosen_price else ""

    # ── Package size (Quy cách / specification) ────────────────────────────────
    package_size = (product.get("specification") or "").strip()

    # ── Manufacturer ──────────────────────────────────────────────────────────
    company = (product.get("producer") or product.get("brand") or "").strip()

    # ── Country of origin ─────────────────────────────────────────────────────
    country_of_origin = (product.get("brandOrigin") or product.get("manufactor") or "").strip()

    # ── Prescription required ─────────────────────────────────────────────────
    prescription_required = bool(product.get("prescription"))

    # ── Drug code: Số đăng ký (registration number) ───────────────────────────
    reg_num = (product.get("registNum") or "").strip()
    sku = (product.get("sku") or "").strip()
    drug_code = reg_num if reg_num else (f"LONGCHAU-{sku}" if sku else f"LONGCHAU-{url.split('/')[-1][:30]}")

    return {
        "drug_code":          drug_code[:100],
        "drug_name":          drug_name[:500],
        "active_ingredient":  active_ingredient[:500],
        "dosage":             dosage[:100],
        "dosage_form":        dosage_form[:100],
        "therapeutic_category": therapeutic_category[:200],
        "prescription_required": prescription_required,
        "company":            company[:200],
        "country_of_origin":  country_of_origin[:100],
        "package_size":       package_size[:200],
        "price":              price,
        "price_per_unit":     price_per_unit[:50],
        "url":                url,
    }

# ─── Main ───────────────────────────────────────────────────────────────────────
def main(limit: Optional[int], resume: bool):
    progress = load_progress() if resume else {"scraped_urls": [], "total_saved": 0}
    already_scraped = set(progress["scraped_urls"])
    total_saved = progress["total_saved"]

    print(f"\n🚀 Long Chau Drug Scraper v4 (A-Z, pure HTTP)")
    print(f"   Mode : {'Resume' if resume else 'Fresh Start'}")
    print(f"   Limit: {limit or 'All'}")
    print(f"   Already scraped: {len(already_scraped)}")
    print(f"─" * 55)

    # ── Step 1: collect all product URLs from A-Z index ───────────────────────
    print("\n📋 Collecting product URLs from A-Z index...")
    if resume and progress.get("all_urls"):
        all_urls = progress["all_urls"]
        print(f"   Loaded {len(all_urls)} URLs from saved progress.")
    else:
        all_urls = collect_all_product_urls()
        progress["all_urls"] = all_urls
        save_progress(progress)
        print(f"\n   ✅ Collected {len(all_urls)} unique product URLs.")

    pending = [u for u in all_urls if u not in already_scraped]
    print(f"   Pending: {len(pending)} (skipping {len(all_urls) - len(pending)} already scraped)\n")

    # ── Step 2: fetch each product page and save ──────────────────────────────
    conn = psycopg.connect(DB_URL)
    print("✅ Connected to database.\n")

    try:
        for url in pending:
            if limit is not None and total_saved >= limit:
                print(f"\n⏹️  Limit of {limit} reached. Stopping.")
                break

            slug = url.split("/")[-1][:60]
            print(f"  🔬 {slug}")
            drug_data = fetch_product_data(url)

            if drug_data:
                try:
                    upsert_drug(conn, drug_data)
                    total_saved += 1
                    already_scraped.add(url)
                    unit_label = f"/{drug_data['price_per_unit']}" if drug_data["price_per_unit"] else ""
                    ing_label = drug_data["active_ingredient"][:50] or "N/A"
                    print(f"     ✅ {drug_data['drug_name'][:50]}")
                    print(f"        💊 {ing_label}")
                    print(f"        💰 {drug_data['price']:,.0f}đ{unit_label} | 📦 {drug_data['package_size'][:40]}")
                    if drug_data["therapeutic_category"]:
                        print(f"        🏷️  {drug_data['therapeutic_category'][:60]}")
                except Exception as e:
                    print(f"     ❌ DB Error: {e}")
                    already_scraped.add(url)
            else:
                already_scraped.add(url)
                print(f"     ⚠️  Skipped (no data)")

            progress["scraped_urls"] = list(already_scraped)
            progress["total_saved"] = total_saved
            save_progress(progress)
    finally:
        conn.close()

    print(f"\n✅ Done! Total drugs saved: {total_saved}")
    print(f"   Progress: {PROGRESS_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape drug prices from Long Chau (v4 - A-Z, pure HTTP)")
    parser.add_argument("--limit",  type=int, default=None, help="Max products to save")
    parser.add_argument("--resume", action="store_true",    help="Resume from progress file")
    args = parser.parse_args()

    main(limit=args.limit, resume=args.resume)
