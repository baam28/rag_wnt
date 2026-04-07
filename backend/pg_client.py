"""PostgreSQL singleton connection pool + all DB helper functions.

Uses psycopg (v3) with psycopg_pool for synchronous operations (the FastAPI
endpoints are sync).
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config import get_settings

# ---------------------------------------------------------------------------
# Singleton connection pool
# ---------------------------------------------------------------------------

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """Return a thread-safe singleton synchronous ConnectionPool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                settings = get_settings()
                _pool = ConnectionPool(
                    conninfo=settings.database_url,
                    min_size=1,
                    max_size=10,
                    open=True,
                )
    return _pool


def get_conn() -> psycopg.Connection:
    """Borrow a connection from the pool (context manager friendly)."""
    return get_pool().connection()


def _q(sql: str, params=None) -> list[dict]:
    """Execute a SELECT and return all rows as list[dict]."""
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def _exec(sql: str, params=None) -> int:
    """Execute INSERT/UPDATE/DELETE; return rowcount."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount


def _exec_returning(sql: str, params=None):
    """Execute INSERT ... RETURNING id; return the first column of the first row."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None


def _to_int_or_none(value) -> int | None:
    """Safely convert a value to int; return None if not possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_users_collection():
    """Return the users proxy object."""
    return _UsersProxy()


class _UsersProxy:
    """Wrapper around the users table."""

    def find_one(self, query: dict) -> dict | None:
        if "username" in query:
            rows = _q("SELECT * FROM users WHERE username = %s", (query["username"],))
        elif "email" in query:
            rows = _q("SELECT * FROM users WHERE email = %s", (query["email"],))
        elif "_id" in query:
            rows = _q("SELECT * FROM users WHERE id = %s", (int(query["_id"]),))
        else:
            return None
        return _user_row_to_doc(rows[0]) if rows else None

    def find(self, query: dict | None = None) -> list[dict]:
        rows = _q("SELECT * FROM users ORDER BY created_at DESC")
        return [_user_row_to_doc(r) for r in rows]

    def insert_one(self, doc: dict):
        new_id = _exec_returning(
            """INSERT INTO users (username, email, password_hash, is_admin, created_at, last_login_at)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (
                doc.get("username"),
                doc.get("email"),
                doc.get("password_hash"),
                bool(doc.get("is_admin", False)),
                doc.get("created_at", datetime.now(timezone.utc)),
                doc.get("last_login_at", datetime.now(timezone.utc)),
            ),
        )
        doc["_id"] = str(new_id)
        return type("InsertResult", (), {"inserted_id": new_id})()

    def update_one(self, query: dict, update: dict, upsert: bool = False):
        user_id = _resolve_user_id(query)
        if user_id is None:
            return
        set_data = update.get("$set", {})
        for col, val in set_data.items():
            _exec(f"UPDATE users SET {col} = %s WHERE id = %s", (val, user_id))

    def delete_one(self, query: dict):
        user_id = _resolve_user_id(query)
        if user_id is None:
            return type("DelResult", (), {"deleted_count": 0})()
        cnt = _exec("DELETE FROM users WHERE id = %s", (user_id,))
        return type("DelResult", (), {"deleted_count": cnt})()

    def count_documents(self, query: dict) -> int:
        rows = _q("SELECT COUNT(*) AS n FROM users")
        return int(rows[0]["n"]) if rows else 0

    def aggregate(self, pipeline: list) -> list[dict]:
        # Only the analytics aggregation is used — group by created_at day
        rows = _q(
            "SELECT TO_CHAR(created_at, 'YYYY-MM-DD') AS day, COUNT(*) AS count"
            " FROM users GROUP BY day"
        )
        return [{"_id": r["day"], "count": int(r["count"])} for r in rows]


def _user_row_to_doc(row: dict) -> dict:
    """Convert a users table row to a dict."""
    doc = dict(row)
    doc["_id"] = str(row["id"])
    # Timestamps → ISO strings
    for key in ("created_at", "last_login_at"):
        v = doc.get(key)
        if isinstance(v, datetime):
            doc[key] = v.strftime("%Y-%m-%dT%H:%M:%SZ")
    return doc


def _resolve_user_id(query: dict) -> int | None:
    if "_id" in query:
        try:
            return int(query["_id"])
        except (ValueError, TypeError):
            return None
    if "username" in query:
        rows = _q("SELECT id FROM users WHERE username = %s", (query["username"],))
        return rows[0]["id"] if rows else None
    if "email" in query:
        rows = _q("SELECT id FROM users WHERE email = %s", (query["email"],))
        return rows[0]["id"] if rows else None
    return None


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------

def get_chat_sessions_collection():
    return _ChatSessionsProxy()


class _ChatSessionsProxy:
    def insert_one(self, doc: dict):
        new_id = _exec_returning(
            """INSERT INTO chat_sessions (user_id, title, created_at, updated_at)
               VALUES (%s, %s, %s, %s) RETURNING id""",
            (
                int(doc["user_id"]),
                doc.get("title", "New chat"),
                doc.get("created_at", datetime.now(timezone.utc)),
                doc.get("updated_at", datetime.now(timezone.utc)),
            ),
        )
        doc["_id"] = str(new_id)

    def find(self, query: dict, **kwargs) -> list[dict]:
        rows = _q(
            "SELECT * FROM chat_sessions WHERE user_id = %s ORDER BY updated_at DESC",
            (int(query.get("user_id", 0)),),
        )
        return [_session_row_to_doc(r) for r in rows]

    def find_one(self, query: dict) -> dict | None:
        if "_id" in query and "user_id" in query:
            try:
                rows = _q(
                    "SELECT * FROM chat_sessions WHERE id = %s AND user_id = %s",
                    (str(query["_id"]), int(query["user_id"])),
                )
            except Exception:
                return None
        elif "_id" in query:
            try:
                rows = _q("SELECT * FROM chat_sessions WHERE id = %s", (str(query["_id"]),))
            except Exception:
                return None
        else:
            return None
        return _session_row_to_doc(rows[0]) if rows else None

    def update_one(self, query: dict, update: dict, upsert: bool = False):
        session_id = query.get("_id")
        user_id = query.get("user_id")
        set_data = update.get("$set", {})
        set_insert = update.get("$setOnInsert", {})

        if upsert and session_id:
            try:
                sid = str(session_id)
                cnt = _exec(
                    "UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s AND user_id = %s",
                    (sid, int(user_id or 0)),
                )
                if cnt == 0:
                    title = set_insert.get("title", set_data.get("title", "New chat"))
                    _exec(
                        "INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at) VALUES (%s, %s, %s, NOW(), NOW())",
                        (sid, int(user_id or 0), title),
                    )
            except Exception:
                pass
        elif session_id and "updated_at" in set_data:
            try:
                _exec(
                    "UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s",
                    (str(session_id),),
                )
            except Exception:
                pass

    def delete_one(self, query: dict):
        try:
            cnt = _exec(
                "DELETE FROM chat_sessions WHERE id = %s AND user_id = %s",
                (str(query["_id"]), int(query["user_id"])),
            )
        except Exception:
            cnt = 0
        return type("DelResult", (), {"deleted_count": cnt})()

    def count_documents(self, query: dict) -> int:
        rows = _q("SELECT COUNT(*) AS n FROM chat_sessions")
        return int(rows[0]["n"]) if rows else 0

    def aggregate(self, pipeline: list) -> list[dict]:
        rows = _q(
            "SELECT TO_CHAR(created_at, 'YYYY-MM-DD') AS day, COUNT(*) AS count"
            " FROM chat_sessions GROUP BY day"
        )
        return [{"_id": r["day"], "count": int(r["count"])} for r in rows]


def _session_row_to_doc(row: dict) -> dict:
    doc = dict(row)
    doc["_id"] = str(row["id"])
    for key in ("created_at", "updated_at"):
        v = doc.get(key)
        if isinstance(v, datetime):
            doc[key] = v.strftime("%Y-%m-%dT%H:%M:%SZ")
    return doc


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------

def get_chat_messages_collection():
    return _ChatMessagesProxy()


class _ChatMessagesProxy:
    def insert_one(self, doc: dict):
        _exec(
            """INSERT INTO chat_messages
               (session_id, user_id, role, content, sources, price_data, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                str(doc.get("session_id")) if doc.get("session_id") else None,
                _to_int_or_none(doc.get("user_id")),
                doc["role"],
                doc["content"],
                json.dumps(doc.get("sources")) if doc.get("sources") else None,
                json.dumps(doc.get("priceData")) if doc.get("priceData") else None,
                doc.get("created_at", datetime.now(timezone.utc)),
            ),
        )

    def find(self, query: dict, **kwargs) -> list[dict]:
        try:
            rows = _q(
                "SELECT * FROM chat_messages WHERE session_id = %s AND user_id = %s ORDER BY created_at ASC",
                (
                    str(query.get("session_id")) if query.get("session_id") else None,
                    _to_int_or_none(query.get("user_id", 0)),
                ),
            )
            return [_msg_row_to_doc(r) for r in rows]
        except Exception:
            return []

    def delete_many(self, query: dict):
        try:
            if "session_id" in query and "user_id" in query:
                _exec(
                    "DELETE FROM chat_messages WHERE session_id = %s AND user_id = %s",
                    (str(query["session_id"]), _to_int_or_none(query["user_id"])),
                )
            elif "session_id" in query:
                _exec("DELETE FROM chat_messages WHERE session_id = %s", (str(query["session_id"]),))
            elif "user_id" in query:
                _exec("DELETE FROM chat_messages WHERE user_id = %s", (_to_int_or_none(query["user_id"]),))
        except Exception:
            pass

    def update_one(self, query: dict, update: dict):
        session_id = str(query.get("session_id")) if query.get("session_id") else None
        role = query.get("role")
        content = query.get("content")
        set_data = update.get("$set", {})
        feedback = set_data.get("feedback")
        comment = set_data.get("feedbackComment", "")
        if session_id and role and content:
            try:
                _exec(
                    """UPDATE chat_messages SET feedback = %s, feedback_comment = %s
                       WHERE session_id = %s AND role = %s AND content = %s""",
                    (feedback, comment, session_id, role, content),
                )
            except Exception:
                pass

    def count_documents(self, query: dict) -> int:
        rows = _q("SELECT COUNT(*) AS n FROM chat_messages")
        return int(rows[0]["n"]) if rows else 0

    def aggregate(self, pipeline: list) -> list[dict]:
        rows = _q(
            "SELECT TO_CHAR(created_at, 'YYYY-MM-DD') AS day, COUNT(*) AS count"
            " FROM chat_messages WHERE role = 'assistant' GROUP BY day"
        )
        return [{"_id": r["day"], "count": int(r["count"])} for r in rows]


def _msg_row_to_doc(row: dict) -> dict:
    doc = dict(row)
    doc["_id"] = str(row["id"])
    v = doc.get("created_at")
    if isinstance(v, datetime):
        doc["created_at"] = v.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Deserialize JSONB fields
    for key in ("sources", "price_data"):
        val = doc.get(key)
        if isinstance(val, str):
            try:
                doc[key] = json.loads(val)
            except Exception:
                doc[key] = None
    # Map price_data → priceData for backward compat with chat router
    if "price_data" in doc:
        doc["priceData"] = doc.pop("price_data")
    return doc


# ---------------------------------------------------------------------------
# Drug prices (new structured format)
# ---------------------------------------------------------------------------

def get_drug_prices_collection():
    return _DrugPricesProxy()


class _DrugPricesProxy:
    def find(self, query: dict, **kwargs) -> list[dict]:
        regex = (query.get("drug_name") or {}).get("$regex", "")
        if regex:
            # PostgreSQL trigram / LIKE search
            search_term = regex.strip(".*").replace(".*", "%")
            rows = _q("""
                SELECT l.drug_name, l.active_ingredient, l.dosage_form, l.prescription_class, l.therapeutic_class,
                       i.selling_unit, i.packaging_size, i.retail_price, i.stock_quantity, i.expiry_date
                FROM drug_list l
                JOIN drug_inventory i ON l.drug_id = i.drug_id
                WHERE l.drug_name ILIKE %s LIMIT 20
            """, (f"%{search_term}%",))
        else:
            rows = _q("""
                SELECT l.drug_name, l.active_ingredient, l.dosage_form, l.prescription_class, l.therapeutic_class,
                       i.selling_unit, i.packaging_size, i.retail_price, i.stock_quantity, i.expiry_date
                FROM drug_list l
                JOIN drug_inventory i ON l.drug_id = i.drug_id
                LIMIT 20
            """)
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def get_feedback_collection():
    return _FeedbackProxy()


class _FeedbackProxy:
    def insert_one(self, doc: dict):
        _exec(
            """INSERT INTO feedback (timestamp, date, question, answer, rating, comment, session_id)
               VALUES (NOW(), CURRENT_DATE, %s, %s, %s, %s, %s)""",
            (
                (doc.get("question") or "")[:500],
                doc.get("answer"),
                doc.get("rating"),
                (doc.get("comment") or "")[:500],
                doc.get("session_id") or None,
            ),
        )

    def find(self, query: dict, **kwargs) -> list[dict]:
        rows = _q("SELECT * FROM feedback ORDER BY timestamp DESC LIMIT 200")
        return [_feedback_row_to_doc(r) for r in rows]

    def count_documents(self, query: dict) -> int:
        if "rating" in query:
            rows = _q("SELECT COUNT(*) AS n FROM feedback WHERE rating = %s", (query["rating"],))
        else:
            rows = _q("SELECT COUNT(*) AS n FROM feedback")
        return int(rows[0]["n"]) if rows else 0


def _feedback_row_to_doc(row: dict) -> dict:
    doc = dict(row)
    ts = doc.get("timestamp")
    if isinstance(ts, datetime):
        doc["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")
        doc["date"] = ts.strftime("%Y-%m-%d")
    return doc


# ---------------------------------------------------------------------------
# LLM Usage
# ---------------------------------------------------------------------------

def get_llm_usage_collection():
    return _LLMUsageProxy()


class _LLMUsageProxy:
    def update_one(self, query: dict, update: dict, upsert: bool = False):
        day = query.get("date")
        inc = update.get("$inc", {})
        pt = int(inc.get("prompt_tokens", 0))
        ct = int(inc.get("completion_tokens", 0))
        req = int(inc.get("requests", 0))
        _exec(
            """INSERT INTO llm_usage_daily (date, prompt_tokens, completion_tokens, requests)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (date) DO UPDATE SET
                   prompt_tokens = llm_usage_daily.prompt_tokens + EXCLUDED.prompt_tokens,
                   completion_tokens = llm_usage_daily.completion_tokens + EXCLUDED.completion_tokens,
                   requests = llm_usage_daily.requests + EXCLUDED.requests""",
            (day, pt, ct, req),
        )

    def find(self, query: dict, **kwargs) -> list[dict]:
        rows = _q(
            "SELECT TO_CHAR(date, 'YYYY-MM-DD') as date, prompt_tokens, completion_tokens, requests"
            " FROM llm_usage_daily ORDER BY date DESC LIMIT 30"
        )
        return list(rows)

    def aggregate(self, pipeline: list) -> list[dict]:
        rows = _q(
            "SELECT SUM(prompt_tokens) AS prompt_tokens,"
            "       SUM(completion_tokens) AS completion_tokens,"
            "       SUM(requests) AS requests"
            " FROM llm_usage_daily"
        )
        if rows:
            r = rows[0]
            return [{"_id": None, "prompt_tokens": r["prompt_tokens"] or 0,
                     "completion_tokens": r["completion_tokens"] or 0,
                     "requests": r["requests"] or 0}]
        return []


# ---------------------------------------------------------------------------
# Vector parents
# ---------------------------------------------------------------------------

def load_vector_parents(collection_name: str) -> dict[str, dict[str, Any]]:
    rows = _q(
        "SELECT parent_id, content, source"
        " FROM vector_parents WHERE collection_name = %s",
        (collection_name,),
    )
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        pid = str(r.get("parent_id") or "").strip()
        if pid:
            out[pid] = {
                "content": r.get("content", ""),
                "source": r.get("source", "Unknown"),
            }
    return out


def upsert_vector_parents(collection_name: str, parent_meta: dict[str, dict[str, Any]]) -> None:
    if not parent_meta:
        return
    now = datetime.now(timezone.utc)
    for parent_id, meta in parent_meta.items():
        document_year = _extract_year_from_issuance_date(meta.get("issuance_date"))
        _exec(
            """INSERT INTO vector_parents
               (collection_name, parent_id, content, source, document_year, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (collection_name, parent_id) DO UPDATE SET
                   content = EXCLUDED.content,
                   source = EXCLUDED.source,
                   document_year = EXCLUDED.document_year,
                   updated_at = EXCLUDED.updated_at""",
            (
                collection_name,
                parent_id,
                meta.get("content", ""),
                meta.get("source", "Unknown"),
                document_year,
                now,
            ),
        )


def list_vector_parents_by_source(
    collection_name: str,
    search: str = "",
    sort_by: str = "document_date",
    sort_order: str = "desc",
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = [collection_name]
    where = "collection_name = %s"
    if search:
        where += " AND source ILIKE %s"
        params.append(f"%{search}%")
    if year_from is not None:
        where += " AND document_year >= %s"
        params.append(year_from)
    if year_to is not None:
        where += " AND document_year <= %s"
        params.append(year_to)

    order = "DESC" if sort_order.lower() == "desc" else "ASC"
    if sort_by == "source":
        order_col = f"source {order}"
    elif sort_by == "parent_count":
        order_col = f"count {order}"
    else:
        order_col = f"document_year {order} NULLS LAST"

    rows = _q(
        f"""SELECT source,
                   COUNT(*) AS count,
                   MAX(document_year) AS document_year
            FROM vector_parents
            WHERE {where}
            GROUP BY source
            ORDER BY {order_col}""",
        params,
    )
    out = []
    for r in rows:
        dy = r.get("document_year")
        dy_src = "issuance_date" if dy is not None else "none"
        if dy is None:
            dy = _extract_year_from_source(r.get("source", ""))
            if dy:
                dy_src = "source_regex"
        out.append({
            "source": r["source"],
            "parent_count": int(r["count"]),
            "document_year": dy,
            "document_year_source": dy_src,
        })
    return out


def delete_vector_parents_by_source(collection_name: str, source: str) -> None:
    _exec(
        "DELETE FROM vector_parents WHERE collection_name = %s AND source = %s",
        (collection_name, source),
    )


# ---------------------------------------------------------------------------
# Sparse vocabulary (BM25)
# ---------------------------------------------------------------------------

def load_sparse_vocab_map(collection_name: str) -> dict[str, int]:
    rows = _q(
        "SELECT token, idx FROM vector_sparse_vocab WHERE collection_name = %s",
        (collection_name,),
    )
    return {r["token"]: int(r["idx"]) for r in rows}


def insert_sparse_vocab_entries(collection_name: str, token_to_idx: dict[str, int]) -> None:
    if not token_to_idx:
        return
    now = datetime.now(timezone.utc)
    with get_conn() as conn:
        with conn.cursor() as cur:
            for token, idx in token_to_idx.items():
                cur.execute(
                    """INSERT INTO vector_sparse_vocab (collection_name, token, idx, updated_at)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (collection_name, token) DO NOTHING""",
                    (collection_name, token, int(idx), now),
                )


# ---------------------------------------------------------------------------
# BM25 stats
# ---------------------------------------------------------------------------

def upsert_sparse_bm25_stats(
    collection_name: str,
    avgdl: float,
    idf_map: dict[str, float],
) -> None:
    _exec(
        """INSERT INTO vector_bm25_stats (collection_name, avgdl, idf_map, updated_at)
           VALUES (%s, %s, %s, NOW())
           ON CONFLICT (collection_name) DO UPDATE SET
               avgdl = EXCLUDED.avgdl,
               idf_map = EXCLUDED.idf_map,
               updated_at = NOW()""",
        (collection_name, float(avgdl), json.dumps(idf_map)),
    )


def load_sparse_bm25_stats(collection_name: str) -> tuple[float, dict[str, float]]:
    rows = _q(
        "SELECT avgdl, idf_map FROM vector_bm25_stats WHERE collection_name = %s",
        (collection_name,),
    )
    if not rows:
        return 1.0, {}
    r = rows[0]
    idf = r.get("idf_map") or {}
    if isinstance(idf, str):
        try:
            idf = json.loads(idf)
        except Exception:
            idf = {}
    return float(r.get("avgdl", 1.0)), dict(idf)


# ---------------------------------------------------------------------------
# Vector chunks (pgvector — replaces Qdrant collection operations)
# ---------------------------------------------------------------------------

def delete_collection_vector_meta(collection_name: str) -> None:
    _exec("DELETE FROM vector_parents WHERE collection_name = %s", (collection_name,))
    _exec("DELETE FROM vector_sparse_vocab WHERE collection_name = %s", (collection_name,))
    _exec("DELETE FROM vector_bm25_stats WHERE collection_name = %s", (collection_name,))
    _exec("DELETE FROM vector_chunks WHERE collection_name = %s", (collection_name,))


def get_collection_count(collection_name: str) -> int:
    rows = _q(
        "SELECT COUNT(*) AS n FROM vector_chunks WHERE collection_name = %s",
        (collection_name,),
    )
    return int(rows[0]["n"]) if rows else 0


def insert_vector_chunks(
    collection_name: str,
    records: list[dict[str, Any]],
) -> None:
    """Batch-insert child chunks with embeddings and sparse vectors.

    Each record must have:
        parent_id, chunk_index, text, embedding (list[float]),
        sparse_indices (list[int]), sparse_values (list[float]),
        payload (dict)
    """
    if not records:
        return

    # Determine vector dimension dynamically and ensure index exists
    dim = len(records[0].get("embedding") or [])
    if dim > 0:
        _ensure_vector_index(collection_name, dim)

    with get_conn() as conn:
        with conn.cursor() as cur:
            for rec in records:
                emb = rec.get("embedding")
                emb_val = f"[{','.join(str(x) for x in emb)}]" if emb else None
                cur.execute(
                    """INSERT INTO vector_chunks
                       (collection_name, parent_id, chunk_index, text,
                        embedding, sparse_indices, sparse_values, payload)
                       VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s)""",
                    (
                        collection_name,
                        rec["parent_id"],
                        int(rec.get("chunk_index", 0)),
                        rec["text"],
                        emb_val,
                        rec.get("sparse_indices") or [],
                        rec.get("sparse_values") or [],
                        json.dumps(rec.get("payload", {})),
                    ),
                )


def _ensure_vector_index(collection_name: str, dim: int) -> None:
    """Create IVFFlat index on embedding column if not already present."""
    # Check if any rows exist with embeddings of this dimension
    rows = _q(
        "SELECT 1 FROM vector_chunks WHERE collection_name = %s AND embedding IS NOT NULL LIMIT 1",
        (collection_name,),
    )
    # Index creation is a DDL statement — run it outside the pool transaction
    try:
        with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
            conn.execute(
                f"""CREATE INDEX IF NOT EXISTS idx_chunks_embedding_{collection_name.replace('-','_')}
                    ON vector_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    WHERE collection_name = '{collection_name}'"""
            )
    except Exception:
        pass  # Index may already exist or not enough rows yet


def similarity_search(
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 20,
    payload_filter: dict | None = None,
) -> list[dict[str, Any]]:
    """Dense vector similarity search using pgvector cosine distance."""
    emb_str = f"[{','.join(str(x) for x in query_embedding)}]"

    extra_where = ""
    params: list[Any] = [collection_name, emb_str, top_k]

    if payload_filter and payload_filter.get("is_pharma_related"):
        extra_where = " AND (payload->>'is_pharma_related')::boolean = TRUE"

    rows = _q(
        f"""SELECT id, parent_id, chunk_index, text, payload,
                   1 - (embedding <=> %s::vector) AS score
            FROM vector_chunks
            WHERE collection_name = %s AND embedding IS NOT NULL{extra_where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s""",
        (emb_str, collection_name, emb_str, top_k),
    )
    return rows


def list_collections() -> list[str]:
    """Return distinct collection names in vector_chunks."""
    rows = _q("SELECT DISTINCT collection_name FROM vector_chunks ORDER BY collection_name")
    return [r["collection_name"] for r in rows]


def delete_chunks_by_source(collection_name: str, source: str) -> int:
    """Delete all chunks whose payload->>'source' matches."""
    cnt = _exec(
        "DELETE FROM vector_chunks WHERE collection_name = %s AND payload->>'source' = %s",
        (collection_name, source),
    )
    return cnt


# ---------------------------------------------------------------------------
# Year helpers
# ---------------------------------------------------------------------------

def _extract_year_from_source(source: str) -> int | None:
    text = str(source or "")
    if not text:
        return None
    years = []
    for match in re.finditer(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text):
        try:
            years.append(int(match.group(1)))
        except Exception:
            continue
    return max(years) if years else None


def _extract_year_from_issuance_date(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split("/")
    if len(parts) >= 3:
        try:
            year_val = int(parts[-1])
            if 1900 <= year_val <= 2099:
                return year_val
        except Exception:
            pass
    return _extract_year_from_source(text)
