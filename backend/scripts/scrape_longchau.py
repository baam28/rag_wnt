#!/usr/bin/env python3
"""
Long Chau Drug Price Scraper (v2)
----------------------------------
Scrapes drug name, ingredient (Thành phần), price (per-box), package size,
and manufacturer from nhathuoclongchau.com.vn.

Key fixes over v1:
- Reads "Thành phần" section for correct active ingredient
- Reads "Quy cách" for package size
- Selects "Hộp" (box) price unit before reading price
- Stores price_per_unit label alongside the price

Usage:
    python scripts/scrape_longchau.py                   # Full crawl
    python scripts/scrape_longchau.py --limit 20        # Quick test
    python scripts/scrape_longchau.py --resume          # Resume from progress file
    python scripts/scrape_longchau.py --category thuoc-giam-dau-ha-sot-khang-viem/thuoc-giam-dau-ha-sot
"""

import asyncio
import argparse
import json
import os
import re
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import psycopg
from playwright.async_api import async_playwright, Page, TimeoutError as PlaywrightTimeout

# ─── Config ────────────────────────────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5433/pharmanet")
PROGRESS_FILE = Path(__file__).parent / "scrape_progress.json"
BASE_URL = "https://nhathuoclongchau.com.vn"
REQUEST_DELAY = 0.1  # seconds between requests (faster)

# Drug category slugs - leaf-level sub-categories for proper pagination
DRUG_CATEGORIES = [
    "thuoc-giam-dau-ha-sot-khang-viem/thuoc-giam-dau-ha-sot",
    "thuoc-giam-dau-ha-sot-khang-viem/thuoc-giam-dau-khang-viem",
    "thuoc-giam-dau-ha-sot-khang-viem/thuoc-khang-viem",
    "thuoc-khang-sinh-khang-nam",
    "thuoc-ho-hap",
    "thuoc-tieu-hoa-and-gan-mat",
    "thuoc-tim-mach-and-mau",
    "thuoc-than-kinh/thuoc-than-kinh",
    "thuoc-bo-and-vitamin",
    "co-xuong-khop",
    "thuoc-mat-tai-mui-hong",
    "thuoc-da-lieu",
    "thuoc-di-ung",
    "thuoc-tiet-nieu-sinh-duc",
    "thuoc-tri-tieu-duong",
    "thuoc-dieu-tri-ung-thu/thuoc-dieu-tri-ung-thu",
    "mieng-dan-cao-xoa-dau",
    "thuoc-tiem-chich-and-dich-truyen",
]

# ─── Progress ──────────────────────────────────────────────────────────────────
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"scraped_urls": [], "total_saved": 0}

def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

# ─── Database ──────────────────────────────────────────────────────────────────
def upsert_drug(conn, data: dict):
    with conn.cursor() as cur:
        # 1. UPSERT drug_list
        cur.execute("""
            INSERT INTO drug_list 
                (drug_id, drug_name, active_ingredient, dosage, dosage_form,
                 prescription_class, therapeutic_class)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (drug_id) DO UPDATE 
            SET drug_name             = EXCLUDED.drug_name,
                active_ingredient     = EXCLUDED.active_ingredient,
                dosage                = EXCLUDED.dosage,
                dosage_form           = EXCLUDED.dosage_form,
                prescription_class    = EXCLUDED.prescription_class,
                therapeutic_class     = EXCLUDED.therapeutic_class,
                updated_at            = NOW()
        """, (
            data["drug_code"],
            data["drug_name"],
            data["active_ingredient"],
            data["dosage"],
            data["dosage_form"],
            "Thuốc kê đơn (Rx)" if data["prescription_required"] else "Thuốc không kê đơn (OTC)",
            data["therapeutic_category"],
        ))

        # 2. UPSERT drug_inventory
        # Only set random stock over writes if it's new (ON CONFLICT DO UPDATE doesn't touch stock/expiry)
        stock_quantity = 0 if random.random() < 0.05 else random.randint(10, 500)
        expiry_date = datetime.now().date() + timedelta(days=random.randint(180, 1000))
        
        cur.execute("""
            INSERT INTO drug_inventory 
                (drug_id, selling_unit, packaging_size, retail_price, stock_quantity, expiry_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (drug_id) DO UPDATE 
            SET selling_unit          = EXCLUDED.selling_unit,
                packaging_size        = EXCLUDED.packaging_size,
                retail_price          = EXCLUDED.retail_price,
                updated_at            = NOW()
        """, (
            data["drug_code"],
            data["price_per_unit"],
            data["package_size"],
            data["price"],
            stock_quantity,
            expiry_date,
        ))
    conn.commit()

# ─── Category Link Collector ────────────────────────────────────────────────────
async def get_product_links_from_category(page: Page, category_slug: str) -> list:
    """Paginate through a category and collect all product URLs."""
    all_links: set = set()
    page_num = 1

    while True:
        url = f"{BASE_URL}/thuoc/{category_slug}?page={page_num}"
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            await asyncio.sleep(1)  # Wait for React to render

            hrefs = await page.evaluate(
                "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.href)"
            )
            page_links = set(
                h for h in hrefs
                if re.search(r'/thuoc/[a-z0-9-]+-\d+\.html$', h)
            )

            if not page_links:
                print(f"  → No products at page {page_num}, stopping.")
                break

            new_links = page_links - all_links
            if not new_links:
                print(f"  → Page {page_num}: no new products, done.")
                break

            all_links |= page_links
            print(f"  → Page {page_num}: {len(new_links)} new products (running total: {len(all_links)})")
            page_num += 1
            await asyncio.sleep(REQUEST_DELAY)

        except PlaywrightTimeout:
            print(f"  ⚠️  Timeout on page {page_num}, stopping pagination.")
            break
        except Exception as e:
            print(f"  ⚠️  Error on {url}: {e}")
            break

    return list(all_links)

# ─── Product Page Scraper ───────────────────────────────────────────────────────
async def scrape_product_page(page: Page, url: str) -> Optional[dict]:
    """Visit a product page and extract drug data accurately."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(1)  # Allow JS to render product info

        # ── Drug name ──────────────────────────────────────────────────────────
        drug_name = ""
        h1 = await page.query_selector("h1")
        if h1:
            drug_name = (await h1.text_content() or "").strip()
        if not drug_name:
            return None

        # ── Package unit selector: prefer "Hộp" (box), else take default ──────
        price = 0.0
        price_per_unit = ""

        unit_buttons = await page.query_selector_all(
            "button, [role='button'], [class*='unit'], [class*='Unit']"
        )
        hop_clicked = False
        for btn in unit_buttons:
            btn_text = (await btn.text_content() or "").strip()
            if btn_text in ("Hộp", "Hộp ", "Lọ", "Túi"):
                try:
                    await btn.click()
                    await asyncio.sleep(0.1)
                    hop_clicked = True
                    price_per_unit = btn_text.strip()
                    break
                except Exception:
                    pass

        # ── Price extraction via JS ────────────────────────────────────────────
        price_raw = await page.evaluate("""
            () => {
                const allText = document.body.innerText;
                const match = allText.match(/([\d]{1,3}(?:[.,][\d]{3})*)\s*đ\s*\/\s*(\S+)/);
                if (match) return { amount: match[1], unit: match[2] };

                const og = document.querySelector('meta[property="product:price:amount"]');
                if (og) return { amount: og.content, unit: 'Hộp' };
                return null;
            }
        """)

        if price_raw and price_raw.get("amount"):
            amount_str = re.sub(r'[^\d]', '', price_raw["amount"])
            if amount_str and len(amount_str) <= 10:
                price = float(amount_str)
            if not price_per_unit:
                price_per_unit = price_raw.get("unit", "")

        # ── Active ingredient from "Thành phần" section ──────────────────────
        active_ingredient = await page.evaluate("""
            () => {
                const allElements = document.querySelectorAll('*');
                for (const el of allElements) {
                    if (el.children.length > 0) continue;
                    const txt = el.textContent.trim();
                    if (txt === 'Thành phần') {
                        let next = el.nextElementSibling || el.parentElement?.nextElementSibling;
                        if (next) return next.textContent.trim().slice(0, 300);
                    }
                }
                const body = document.body.innerText;
                const m = body.match(/Thành phần[:\\s]*([^\\n]{3,200})/);
                if (m) return m[1].trim().slice(0, 300);
                return '';
            }
        """)

        # ── Package size from "Quy cách" section ─────────────────────────────
        package_size = await page.evaluate("""
            () => {
                const body = document.body.innerText;
                const m = body.match(/Quy cách[:\\s]*([^\\n]{3,100})/);
                if (m) return m[1].trim().slice(0, 100);
                return '';
            }
        """)

        # ── Manufacturer/Company ────────────────────────────────────────────────
        company = await page.evaluate("""
            () => {
                const body = document.body.innerText;
                const patterns = [
                    /Thương hiệu[:\\s]*([^\\n]{2,100})/,
                    /Nhà sản xuất[:\\s]*([^\\n]{2,100})/,
                    /Hãng sản xuất[:\\s]*([^\\n]{2,100})/,
                ];
                for (const pat of patterns) {
                    const m = body.match(pat);
                    if (m) return m[1].trim().slice(0, 200);
                }
                return '';
            }
        """)

        # ── Dosage form ───────────────────────────────────────────────────────
        dosage_form = await page.evaluate("""
            () => {
                const body = document.body.innerText;
                const forms = [
                    'Viên nén', 'Viên nang', 'Viên sủi', 'Viên nhai',
                    'Dung dịch', 'Hỗn dịch', 'Bột pha', 'Bột', 'Gel',
                    'Kem', 'Thuốc mỡ', 'Thuốc nhỏ', 'Cao dán', 'Miếng dán',
                    'Siro', 'Xịt', 'Dầu', 'Cồn', 'Dung dịch tiêm', 'Viên'
                ];
                const m = body.match(/Dạng bào chế[:\\s]*([^\\n]{3,80})/);
                if (m) return m[1].trim().slice(0, 100);
                const title = document.querySelector('h1')?.textContent || '';
                for (const f of forms) {
                    if (title.startsWith(f)) return f;
                }
                return '';
            }
        """)

        # ── Therapeutic category ───────────────────────────────────────────────
        therapeutic_category = await page.evaluate("""
            () => {
                const breadcrumbs = Array.from(document.querySelectorAll('[class*="breadcrumb"] a, nav a'))
                    .map(a => a.textContent.trim())
                    .filter(t => t && t !== 'Trang chủ' && t !== 'Thuốc');
                if (breadcrumbs.length > 0) return breadcrumbs[0].slice(0, 200);
                const body = document.body.innerText;
                const m = body.match(/(?:Nhóm thuốc|Phân loại)[:\\s]*([^\\n]{3,100})/);
                if (m) return m[1].trim().slice(0, 200);
                return '';
            }
        """)

        # ── Prescription required ─────────────────────────────────────────────
        prescription_required = await page.evaluate("""
            () => {
                const body = document.body.innerText;
                return body.includes('Thuốc kê đơn') || body.includes('Kê đơn');
            }
        """)

        # ── Country of origin ─────────────────────────────────────────────────
        country_of_origin = await page.evaluate("""
            () => {
                const body = document.body.innerText;
                const patterns = [
                    /Nước sản xuất[:\\s]*([^\\n]{2,50})/,
                    /Xuất xứ[:\\s]*([^\\n]{2,50})/,
                ];
                for (const pat of patterns) {
                    const m = body.match(pat);
                    if (m) return m[1].trim().slice(0, 100);
                }
                return '';
            }
        """)

        # ── Drug code from URL ─────────────────────────────────────────────────
        code_match = re.search(r'-(\d+)\.html$', url)
        drug_code = f"LONGCHAU-{code_match.group(1)}" if code_match else url[-20:]

        dosage_matches = re.findall(r'\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|iu|UI)', active_ingredient, re.IGNORECASE)
        dosage = ", ".join(dosage_matches) if dosage_matches else ""

        return {
            "drug_code": drug_code,
            "drug_name": drug_name[:500],
            "active_ingredient": active_ingredient[:500],
            "dosage": dosage[:100],
            "dosage_form": dosage_form[:100],
            "therapeutic_category": therapeutic_category[:200],
            "prescription_required": bool(prescription_required),
            "company": company[:200],
            "country_of_origin": country_of_origin[:100],
            "package_size": package_size[:200],
            "price": price,
            "price_per_unit": price_per_unit[:50],
            "url": url,
        }

    except PlaywrightTimeout:
        print(f"    ⚠️  Timeout: {url}")
        return None
    except Exception as e:
        print(f"    ⚠️  Error: {e}")
        return None

# ─── Main ───────────────────────────────────────────────────────────────────────
async def main(limit: Optional[int], resume: bool, category: Optional[str]):
    progress = load_progress() if resume else {"scraped_urls": [], "total_saved": 0}
    already_scraped = set(progress["scraped_urls"])
    total_saved = progress["total_saved"]

    categories = [category] if category else DRUG_CATEGORIES

    print(f"\n🚀 Long Chau Drug Scraper v2")
    print(f"   Fixes: correct ingredient (Thành phần), box price, package size")
    print(f"   Mode : {'Resume' if resume else 'Fresh Start'}")
    print(f"   Limit: {limit or 'All'}")
    print(f"   Categories: {len(categories)}")
    print(f"   Already scraped: {len(already_scraped)}")
    print(f"─" * 55)

    conn = psycopg.connect(DB_URL)
    print("✅ Connected to database.\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
            locale="vi-VN",
            extra_http_headers={"Accept-Language": "vi-VN,vi;q=0.9"}
        )
        # Block heavy media to speed up
        await context.route(
            "**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf,mp4,mp3}",
            lambda route: route.abort()
        )

        page = await context.new_page()

        for cat_slug in categories:
            print(f"\n📂 Category: {cat_slug}")
            product_links = await get_product_links_from_category(page, cat_slug)
            pending = [u for u in product_links if u not in already_scraped]
            print(f"   {len(product_links)} total | {len(pending)} new to scrape")

            for url in pending:
                if limit is not None and total_saved >= limit:
                    print(f"\n⏹️  Limit of {limit} reached. Stopping.")
                    break

                slug = url.split('/')[-1][:60]
                print(f"  🔬 {slug}")
                drug_data = await scrape_product_page(page, url)

                if drug_data and drug_data["drug_name"]:
                    try:
                        upsert_drug(conn, drug_data)
                        total_saved += 1
                        already_scraped.add(url)
                        unit_label = f"/{drug_data['price_per_unit']}" if drug_data['price_per_unit'] else ""
                        ingredient_label = drug_data['active_ingredient'][:50] if drug_data['active_ingredient'] else "N/A"
                        print(f"     ✅ {drug_data['drug_name'][:45]}")
                        print(f"        💊 {ingredient_label}")
                        print(f"        💰 {drug_data['price']:,.0f}đ{unit_label} | 📦 {drug_data['package_size'][:40]}")
                    except Exception as e:
                        print(f"     ❌ DB Error: {e}")
                else:
                    already_scraped.add(url)  # Don't retry failed pages
                    print(f"     ⚠️  Skipped (no data)")

                # Save after every product
                progress["scraped_urls"] = list(already_scraped)
                progress["total_saved"] = total_saved
                save_progress(progress)

            if limit is not None and total_saved >= limit:
                break

        await browser.close()

    conn.close()
    print(f"\n✅ Done! Total drugs saved: {total_saved}")
    print(f"   Progress: {PROGRESS_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape drug prices from Long Chau (v2)")
    parser.add_argument("--limit",    type=int, default=None, help="Max products to save")
    parser.add_argument("--resume",   action="store_true",    help="Resume from progress file")
    parser.add_argument("--category", type=str, default=None, help="Scrape a single category slug")
    args = parser.parse_args()

    asyncio.run(main(limit=args.limit, resume=args.resume, category=args.category))
