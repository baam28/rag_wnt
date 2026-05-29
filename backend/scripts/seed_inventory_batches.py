#!/usr/bin/env python3
"""Seed a few realistic multi-batch rows in drug_inventory.

This script adds extra batches for a small set of existing products so the same
product can appear with different batch_number, batch_date, and expiry_date.
"""

from __future__ import annotations

import os
import random
from datetime import date, timedelta

import psycopg

DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5433/pharmanet")


def main() -> None:
    random.seed(42)
    today = date.today()

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.drug_id, COALESCE(i.selling_unit, ''), COALESCE(i.packaging_size, ''),
                       COALESCE(i.retail_price, 0), COALESCE(i.stock_quantity, 0)
                FROM drug_inventory i
                ORDER BY i.drug_id
                LIMIT 8
                """
            )
            base_rows = cur.fetchall()

            if not base_rows:
                print("No inventory rows found. Run scraper first.")
                return

            inserted = 0
            for drug_id, selling_unit, packaging_size, retail_price, stock_quantity in base_rows:
                for idx in range(2, 4):
                    batch_number = f"{str(drug_id)[:8].upper()}-B{idx:03d}"
                    batch_date = today - timedelta(days=random.randint(30, 240))
                    expiry_date = batch_date + timedelta(days=random.randint(365, 900))
                    batch_stock = max(0, int(stock_quantity) + random.randint(-20, 80))

                    cur.execute(
                        """
                        INSERT INTO drug_inventory
                            (drug_id, batch_number, selling_unit, packaging_size, retail_price,
                             stock_quantity, batch_date, expiry_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (drug_id, batch_number) DO UPDATE
                        SET selling_unit = EXCLUDED.selling_unit,
                            packaging_size = EXCLUDED.packaging_size,
                            retail_price = EXCLUDED.retail_price,
                            stock_quantity = EXCLUDED.stock_quantity,
                            batch_date = EXCLUDED.batch_date,
                            expiry_date = EXCLUDED.expiry_date,
                            updated_at = NOW()
                        """,
                        (
                            drug_id,
                            batch_number,
                            selling_unit,
                            packaging_size,
                            retail_price,
                            batch_stock,
                            batch_date,
                            expiry_date,
                        ),
                    )
                    inserted += 1

        conn.commit()

    print(f"Seeded/updated {inserted} extra batch rows across {len(base_rows)} products.")


if __name__ == "__main__":
    main()
