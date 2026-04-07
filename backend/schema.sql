-- ============================================================
-- RAG WNT — PostgreSQL + pgvector Schema
-- ============================================================
-- Run once against your database:
--   psql $DATABASE_URL -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- for fuzzy drug name search

-- ─── Users & Auth ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(100) NOT NULL UNIQUE,
    email           VARCHAR(200) UNIQUE,
    password_hash   TEXT NOT NULL,
    is_admin        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at   TIMESTAMPTZ
);

-- ─── Chat ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT 'New chat',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role            VARCHAR(20) NOT NULL,   -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    sources         JSONB,
    price_data      JSONB,
    feedback        VARCHAR(10),           -- 'up' | 'down'
    feedback_comment TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_user ON chat_messages(user_id);

-- ─── Drugs & Inventory ────────────────────────────────────────────────────────
-- Stores core clinical info about the drugs
CREATE TABLE IF NOT EXISTS drug_list (
    drug_id               TEXT PRIMARY KEY,
    drug_name             TEXT NOT NULL,
    active_ingredient     TEXT,
    dosage                TEXT,
    dosage_form           TEXT,
    prescription_class    TEXT,
    therapeutic_class     TEXT,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_drug_list_name ON drug_list USING gin(drug_name gin_trgm_ops);

-- Stores dynamic inventory and ERP variables
CREATE TABLE IF NOT EXISTS drug_inventory (
    inventory_id          SERIAL PRIMARY KEY,
    drug_id               TEXT NOT NULL REFERENCES drug_list(drug_id) ON DELETE CASCADE,
    selling_unit          TEXT,
    packaging_size        TEXT,
    retail_price          NUMERIC,
    stock_quantity        INTEGER NOT NULL DEFAULT 0,
    expiry_date           DATE,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(drug_id)
);

-- ─── Feedback ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    date        DATE NOT NULL DEFAULT CURRENT_DATE,
    question    TEXT,
    answer      TEXT,
    rating      VARCHAR(10) NOT NULL,    -- 'up' | 'down'
    comment     TEXT,
    session_id  UUID
);
CREATE INDEX IF NOT EXISTS idx_feedback_date ON feedback(date);

-- ─── LLM Usage ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS llm_usage_daily (
    date                DATE PRIMARY KEY,
    prompt_tokens       BIGINT NOT NULL DEFAULT 0,
    completion_tokens   BIGINT NOT NULL DEFAULT 0,
    requests            BIGINT NOT NULL DEFAULT 0
);

-- ─── Vector Parents (parent-child chunk metadata) ─────────────────────────────
CREATE TABLE IF NOT EXISTS vector_parents (
    id              SERIAL PRIMARY KEY,
    collection_name TEXT NOT NULL,
    parent_id       TEXT NOT NULL,
    content         TEXT,
    source          TEXT,
    document_year   INTEGER,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (collection_name, parent_id)
);
CREATE INDEX IF NOT EXISTS idx_vector_parents_coll ON vector_parents(collection_name);
CREATE INDEX IF NOT EXISTS idx_vector_parents_source ON vector_parents(collection_name, source);

-- ─── Sparse Vocabulary (BM25) ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vector_sparse_vocab (
    id              SERIAL PRIMARY KEY,
    collection_name TEXT NOT NULL,
    token           TEXT NOT NULL,
    idx             INTEGER NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (collection_name, token)
);
CREATE INDEX IF NOT EXISTS idx_sparse_vocab_coll ON vector_sparse_vocab(collection_name, idx);

-- ─── BM25 Stats ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vector_bm25_stats (
    collection_name TEXT PRIMARY KEY,
    avgdl           DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    idf_map         JSONB NOT NULL DEFAULT '{}',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Vector Chunks (pgvector) ────────────────────────────────────────────────
-- Dense embedding + sparse BM25 weights stored as JSONB
-- One row per child chunk; parent metadata looked up via parent_id.
CREATE TABLE IF NOT EXISTS vector_chunks (
    id              BIGSERIAL PRIMARY KEY,
    collection_name TEXT NOT NULL,
    parent_id       TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    text            TEXT NOT NULL,
    embedding       vector,              -- dimension set at ingest time (NULL = not yet embedded)
    sparse_indices  INTEGER[],           -- BM25 sparse vector indices
    sparse_values   DOUBLE PRECISION[],  -- BM25 sparse vector values
    payload         JSONB NOT NULL DEFAULT '{}',  -- all chunk metadata
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chunks_coll ON vector_chunks(collection_name);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON vector_chunks(collection_name, parent_id);

-- The vector index is created AFTER first documents are ingested since we need to know the dimension.
-- Run manually after ingest:
--   CREATE INDEX idx_chunks_embedding ON vector_chunks
--   USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
