# PharmaAI — Vietnamese Drug & Legal RAG Assistant

A full-stack RAG chatbot for Vietnamese pharmacy workflows:

- Drug information Q&A (`drug` collection)
- Legal/pharmacy regulation Q&A (`legal` collection)
- Real-time drug inventory lookup (price, stock, expiry, batch) from PostgreSQL ERP
- Authenticated chat sessions with conversation history
- Admin dashboard: user management, document ingestion, analytics, runtime LLM/model switching
- Automatic legal cross-reference pre-fetch (e.g. "theo khoản 1 Điều 13")

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, LangChain, psycopg v3 |
| Database | PostgreSQL 16 + pgvector + pg_trgm |
| LLM | OpenAI (`gpt-4o-mini`) or Anthropic (`claude-sonnet-4-6`) — runtime switchable |
| Embeddings | Voyage AI `voyage-law-2` (fixed) |
| Retrieval | True hybrid: parallel dense (HNSW) + BM25, fused with RRF, reranked with CrossEncoder |
| Ingestion | PDF via Docling, DOCX via python-docx |
| Frontend | React 18 + Vite |

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  React Frontend (Vite)                     │
│  Chat sessions · Auth · Admin dashboard · File upload      │
└──────────────────────────┬─────────────────────────────────┘
                           │ HTTP / JWT
┌──────────────────────────▼─────────────────────────────────┐
│                FastAPI Backend (Python)                     │
│  Auth · Chat · Admin · Ingest routers                      │
├────────────────────────────────────────────────────────────┤
│  Supervisor: context-aware intent classification           │
│  · Routes to: legal · drug · ERP (or combinations)        │
│  · Uses conversation history to resolve follow-up intent   │
├────────────────────────────────────────────────────────────┤
│  Retrieval: True Hybrid Search                             │
│  · Dense  — pgvector cosine similarity (HNSW index)        │
│  · Sparse — BM25 (independent PostgreSQL query, RRF fuse)  │
│  · Rerank — CrossEncoder (mMiniLMv2-L12-H384)             │
│  · CrossRef — pre-fetch articles cited in query text       │
├────────────────────────────────────────────────────────────┤
│  Ingestion Pipeline                                        │
│  · Load  — PDF (Docling markdown), DOCX (python-docx)     │
│  · Parse — Legal articles (Điều/Article/Section) + clauses │
│  · Chunk — Token-based parent-child hierarchy              │
│  · Store — PostgreSQL (dense + sparse vectors + metadata)  │
├────────────────────────────────────────────────────────────┤
│  LLM: OpenAI GPT-4o-mini | Anthropic Claude Sonnet 4.6     │
│  · Query rewrite · Expansion · Supervisor classification   │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│           PostgreSQL 16 + pgvector (Docker)                │
│  users · chat_sessions · chat_messages                     │
│  vector_chunks · vector_parents                            │
│  vector_sparse_vocab · vector_bm25_stats                   │
│  drug_list · drug_inventory                                │
│  feedback · llm_usage_daily · app_settings                 │
└────────────────────────────────────────────────────────────┘
```

## Project Layout

```text
rag_wnt/
├── backend/
│   ├── app.py                  # FastAPI app, routers, middleware
│   ├── config.py               # Pydantic settings (env vars)
│   ├── pg_client.py            # PostgreSQL connection pool + all DB operations
│   ├── schema.sql              # Full DB schema (auto-applied on first Docker run)
│   ├── ingest.py               # Ingestion pipeline (chunking, embedding, storage)
│   ├── retriever.py            # Hybrid search, reranking, query expansion
│   ├── agents.py               # RAG agent runners (federated + ERP)
│   ├── supervisor.py           # LLM-based intent classification (legal / drug / erp)
│   ├── prompts.py              # Prompt templates + LLM/embedding client builders
│   ├── drug_price_tool.py      # Text-to-SQL agent for ERP/inventory queries
│   ├── legal_crossref.py       # Vietnamese legal cross-reference detection + resolution
│   ├── legal_tokenizer.py      # Vietnamese legal document tokenizer (BM25)
│   ├── llm_usage.py            # Token usage tracking
│   ├── deps.py                 # FastAPI dependencies, Pydantic models, auth helpers
│   ├── routers/
│   │   ├── auth.py             # Register · Login · Change password
│   │   ├── chat.py             # /ask · sessions · messages · drug-price
│   │   ├── ingest_router.py    # File upload · async ingest jobs
│   │   └── admin.py            # Collections · docs · users · analytics · feedback · settings
│   ├── scripts/
│   │   ├── scrape_longchau.py         # Playwright scraper → drug_list / drug_inventory
│   │   └── seed_inventory_batches.py  # Seed realistic multi-batch inventory data
│   └── eval/
│       ├── ragas_runner.py     # RAGAS evaluation CLI
│       ├── ragas_adapter.py    # Pipeline adapter for evaluation
│       ├── dataset_loader.py   # Eval dataset loading + synthetic expansion
│       ├── reporting.py        # Result serialisation + output files
│       └── datasets/
│           ├── curated.jsonl       # Curated evaluation dataset
│           └── drug_erp_50.jsonl   # Drug ERP evaluation dataset
├── frontend/
│   ├── app.js                  # Main React component (auth, sessions, admin)
│   ├── chat_page.js            # Chat UI
│   ├── admin_page.js           # Admin dashboard UI
│   ├── landing.js              # Login / register page
│   ├── styles.css
│   ├── main.jsx
│   ├── vite.config.js
│   └── package.json
├── docker-compose.yml          # PostgreSQL + pgvector service
├── .env.example
└── README.md
```

## Database Schema

All tables are defined in `backend/schema.sql` and applied automatically on first Docker start.

| Table | Purpose |
|---|---|
| `users` | Auth: username, email, password_hash, is_admin |
| `chat_sessions` | Chat sessions per user |
| `chat_messages` | Messages with role, content, sources (JSONB), feedback, seq (ordering) |
| `vector_chunks` | Child chunks: dense embedding + sparse BM25 indices/values + payload (JSONB) |
| `vector_parents` | Parent chunk metadata: content, source, document_year |
| `vector_sparse_vocab` | BM25 vocabulary: token → index per collection |
| `vector_bm25_stats` | BM25 stats: avgdl + idf_map (JSONB) per collection |
| `drug_list` | Drug catalog: name, active ingredient, dosage form, therapeutic class |
| `drug_inventory` | Drug inventory: price, stock, batch number, batch date, expiry date |
| `feedback` | User ratings on answers |
| `llm_usage_daily` | Token usage per date |
| `app_settings` | Runtime key-value config (LLM provider/model) |

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for PostgreSQL + pgvector)
- Voyage AI API key (required — used for all embeddings)
- OpenAI API key **and/or** Anthropic API key (at least one required for the LLM)

## Environment Variables

Create `.env` at the project root (copy from `.env.example`):

```env
# LLM provider — "openai" (default) or "anthropic"
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...   # required if LLM_PROVIDER=anthropic

# Embeddings — Voyage AI only
VOYAGE_API_KEY=pa-...

JWT_SECRET=replace-with-strong-random-secret-at-least-32-chars

# PostgreSQL (matches docker-compose defaults)
DATABASE_URL=postgresql://admin:password@localhost:5433/pharmanet

# Optional: auto-admin at first registration (JSON array)
ADMIN_EMAILS=["admin@example.com"]

# Optional tuning
ASK_RATE_LIMIT=20/minute
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
ENABLE_CROSSREF_RESOLUTION=true
CROSSREF_MAX_REFS=3

# Retrieval speed knobs (advanced)
# HYBRID_TOP_K=15
# RERANK_INPUT_K=15
# DENSE_MULTI_QUERY_K=1
# QUERY_EXPANSION_COUNT=2
```

> **Important:** All API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`) must be set in `.env` or as environment variables. They are never stored in or read from the database.

## Install

```bash
# Backend
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

## Run (Development)

**1. Start PostgreSQL + pgvector:**

```bash
docker compose up -d
```

The schema is applied automatically from `backend/schema.sql` on first run.

**2. Start backend:**

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**3. Start frontend dev server (optional — for hot reload):**

```bash
cd frontend && npm run dev
```

Vite proxies API calls to `http://localhost:8000`.

## Run (Production)

Build the frontend and let FastAPI serve it as static files:

```bash
cd frontend && npm run build
cd ../backend && uvicorn app:app --host 0.0.0.0 --port 8000
```

| URL | Description |
|---|---|
| `http://localhost:8000/app/` | Chat application |
| `http://localhost:8000/admin/` | Admin dashboard |
| `http://localhost:8000/docs` | Interactive API docs |

## API Reference

### Auth

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login, returns JWT |
| PUT | `/auth/password` | Change own password |

### Chat

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Health check |
| POST | `/ask` | User | Main RAG endpoint (rate-limited) |
| POST | `/chat/sessions` | User | Create session |
| GET | `/chat/sessions` | User | List sessions |
| DELETE | `/chat/sessions/{id}` | User | Delete session |
| GET | `/chat/sessions/{id}/messages` | User | Get messages |
| POST | `/drug-price` | User | Drug price / inventory lookup |

### Ingest

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/ingest-file` | Admin | Upload PDF/DOCX (`?async=true` for background job) |
| GET | `/ingest-jobs/{job_id}` | Admin | Poll async job status |
| POST | `/ingest-jobs/{job_id}/cancel` | Admin | Cancel async job |

### Admin

| Method | Path | Description |
|---|---|---|
| POST | `/feedback` | Submit answer feedback (public) |
| GET | `/admin/feedback` | View all feedback |
| GET | `/admin/analytics` | 30-day stats + token usage |
| GET | `/admin/collections` | List collections |
| GET | `/admin/docs` | List sources in a collection |
| DELETE | `/admin/docs` | Delete document by source |
| DELETE | `/admin/collections/{name}` | Delete entire collection |
| GET | `/admin/users` | List users |
| PUT | `/admin/users/{id}/role` | Set admin flag |
| PUT | `/admin/users/{id}/password` | Reset user password |
| DELETE | `/admin/users/{id}` | Delete user |
| GET | `/admin/api-settings` | View active LLM provider/model |
| PUT | `/admin/api-settings` | Switch LLM provider or model |

## Retrieval Pipeline

Each `/ask` request goes through:

1. **Intent classification** (`supervisor.py`) — LLM routes to `legal`, `drug`, ERP, or any combination; uses conversation history to correctly handle follow-up questions (e.g. "loại nào không cần kê đơn" after a paracetamol lookup)
2. **Query reformulation** — rewrites follow-up questions into standalone queries using conversation history, so retrieval does not depend on pronoun resolution
3. **Query expansion** — rewrites + generates semantic variations in one LLM call
4. **True hybrid search** — for each collection, two independent searches run in parallel:
   - Dense: Voyage AI `voyage-law-2` embeddings → pgvector cosine similarity (HNSW index, `ef_search` hint)
   - Sparse: BM25 scoring via PostgreSQL lateral-unnest join (precomputed vocab + Robertson IDF, in-process cache)
   - Fused with Reciprocal Rank Fusion (k=60); dense results below cosine threshold (default 0.30) are discarded
5. **Reranker pre-filter** — top candidates by RRF score (default 20) are fed to CrossEncoder; avoids O(n) inference over all results
6. **Reranking** — CrossEncoder selects top 5 (top 8 for legal queries)
7. **Parent fetch** — retrieves full parent article text for reranked child chunks
8. **LLM generation** — answer generated from context + conversation history

## Legal Cross-Reference Pre-Fetch

`backend/legal_crossref.py` scans the incoming query for explicit article citations (e.g. "theo khoản 1 Điều 13" or "căn cứ Điều 55 khoản 2") and fetches those articles from the legal collection **before** the first generation pass. This ensures that cited article text is included in the first (and only) LLM call, eliminating shallow "see Article X" responses without requiring a second round-trip.

Configure via `.env`:

```env
ENABLE_CROSSREF_RESOLUTION=true   # set false to disable
CROSSREF_MAX_REFS=3               # max article references to resolve per query
```

## Ingestion Pipeline

Each uploaded PDF or DOCX goes through:

1. **Load** — PDF via Docling (→ Markdown), DOCX via python-docx
2. **Parse**
   - Vietnamese legal documents: split on `Điều`/`Article`/`Section` markers → one parent per article
   - Non-legal: semantic/header-based splitting
   - Articles exceeding 3 000 tokens are further split with `RecursiveCharacterTextSplitter`
3. **Child chunking** — each parent split on numbered clause markers; falls back to sentence splitting
4. **Embedding** — Voyage AI `voyage-law-2`
5. **Sparse vectors** — BM25 indices/values computed from corpus vocabulary; in-process BM25 cache invalidated on each ingest
6. **Store** — dense + sparse vectors → `vector_chunks`; parent metadata → `vector_parents`

## Drug Data

`backend/scripts/scrape_longchau.py` scrapes `nhathuoclongchau.com.vn` using Playwright and populates the `drug_list` and `drug_inventory` tables.

```bash
cd backend/scripts && python scrape_longchau.py
```

Supports resumable runs via `scrape_progress.json`.

To seed realistic multi-batch inventory data (same `drug_id`, different `batch_number` / `batch_date` / `expiry_date`):

```bash
cd backend/scripts && python seed_inventory_batches.py
```

## RAGAS Evaluation

```bash
source .venv/bin/activate && cd backend
python -m eval.ragas_runner \
  --dataset eval/datasets/curated.jsonl \
  --metrics faithfulness,answer_relevancy,context_precision,context_recall
```

Optional flags:

```bash
python -m eval.ragas_runner \
  --dataset eval/datasets/curated.jsonl \
  --include-synthetic \
  --synthetic-max-per-collection 40
```

Outputs written to `eval_results/`:

| File | Contents |
|---|---|
| `<run>.summary.json` | Aggregate metrics, run metadata, latency benchmark (avg / p50 / p95) |
| `<run>.raw.json` / `.raw.csv` | Per-row RAGAS scores |
| `<run>.pipeline_outputs.json` | Question, retrieved contexts, generated answer |

Optional eval settings in `.env`:

```env
EVAL_JUDGE_MODEL=gpt-4o-mini
EVAL_MAX_WORKERS=2
EVAL_TIMEOUT_SECONDS=90
EVAL_MAX_SAMPLES=0          # 0 = run all samples
```

## Admin Panel

The admin dashboard (`/admin/`) lets you switch the LLM provider and model without restarting the server. The active configuration is stored in the `app_settings` table and takes effect immediately on the next request.

| Setting | Options |
|---|---|
| LLM provider | `openai` or `anthropic` |
| LLM model | Any model supported by the chosen provider (e.g. `gpt-4o`, `claude-opus-4-7`) |

> **Note:** API keys are read exclusively from environment variables (`.env`) and are never stored in or configurable through the admin panel. To rotate a key, update `.env` and restart the backend.

## Operational Notes

- `/ask` rate limit: 20 requests/minute per user (configurable via `ASK_RATE_LIMIT`). Uses JWT subject when available, falls back to IP.
- LLM token usage is tracked daily in `llm_usage_daily` and visible in the admin analytics dashboard.
- PostgreSQL data is persisted in a named Docker volume (`postgres_data`).
- The `uploads/` directory is created automatically on first run.
- BM25 vocabulary and IDF stats are cached in-process per collection; the cache is invalidated automatically after each document ingest.
