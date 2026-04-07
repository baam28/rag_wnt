# PharmaAI RAG — Vietnamese Drug & Legal Assistant

A full-stack RAG chatbot for Vietnamese pharmacy workflows:

- Drug information Q&A (`drug` collection)
- Legal/pharmacy regulation Q&A (`legal` collection)
- Real-time drug price lookup integration
- Authenticated chat sessions with admin dashboard

## Tech Stack

- **Backend**: FastAPI, LangChain, psycopg (PostgreSQL v3 driver)
- **Database**: PostgreSQL 16 + pgvector (vector similarity) + pg_trgm (fuzzy search)
- **LLM / Embeddings**: OpenAI (`gpt-4o-mini`, `text-embedding-3-small` by default)
- **Retrieval**: Hybrid dense (pgvector) + sparse (BM25), reranked with CrossEncoder
- **Ingestion**: PDF (Docling), DOCX (python-docx), Hugging Face datasets
- **Frontend**: React 18 + Vite (SPA served from `frontend/dist`)

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
│  Retrieval: Hybrid Search                                  │
│  · Dense  — pgvector cosine similarity (IVFFlat index)     │
│  · Sparse — BM25 (precomputed vocab + Robertson IDF)       │
│  · Rerank — CrossEncoder (mMiniLMv2-L12-H384)             │
├────────────────────────────────────────────────────────────┤
│  Ingestion Pipeline                                        │
│  · Load  — PDF (Docling markdown), DOCX (python-docx)     │
│  · Parse — Legal articles (Điều/Article/Section) + clauses │
│  · Chunk — Token-based parent-child hierarchy              │
│  · Store — PostgreSQL (dense + sparse vectors + metadata)  │
├────────────────────────────────────────────────────────────┤
│  LLM: OpenAI GPT-4o-mini                                   │
│  · Query expansion · Reformulation · Summaries             │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│           PostgreSQL 16 + pgvector (Docker)                │
│  users · chat_sessions · chat_messages                     │
│  vector_chunks · vector_parents                            │
│  vector_sparse_vocab · vector_bm25_stats                   │
│  drug_list · drug_inventory                                │
│  feedback · llm_usage_daily                                │
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
│   ├── agents.py               # RAG agent runners (federated + price)
│   ├── supervisor.py           # Intent classification (legal / drug / general)
│   ├── prompts.py              # LLM prompt templates
│   ├── drug_price_tool.py      # Live drug price lookup tool
│   ├── llm_usage.py            # Token usage tracking
│   ├── utils.py                # Shared helpers
│   ├── legal_tokenizer.py      # Vietnamese legal document tokenizer
│   ├── routers/
│   │   ├── auth.py             # Register · Login · Change password
│   │   ├── chat.py             # /ask · sessions · messages
│   │   ├── ingest_router.py    # File upload · HF legal dataset · async jobs
│   │   └── admin.py            # Collections · docs · users · analytics · feedback
│   ├── scripts/
│   │   └── scrape_longchau.py  # Playwright scraper → drug_list / drug_inventory
│   └── eval/
│       ├── ragas_runner.py     # RAGAS evaluation runner
│       └── datasets/
│           └── curated.jsonl   # Evaluation dataset
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
| `chat_messages` | Messages with role, content, sources (JSONB), feedback |
| `vector_chunks` | Child chunks: dense embedding + sparse BM25 indices/values + payload (JSONB) |
| `vector_parents` | Parent chunk metadata: content, summary, target_question, source |
| `vector_sparse_vocab` | BM25 vocabulary: token → index per collection |
| `vector_bm25_stats` | BM25 stats: avgdl + idf_map (JSONB) per collection |
| `drug_list` | Drug catalog: name, active ingredient, dosage form, therapeutic class |
| `drug_inventory` | Drug inventory: price, stock, expiry (scraped from Long Châu) |
| `feedback` | User ratings on answers |
| `llm_usage_daily` | Token usage per date |

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for PostgreSQL + pgvector)
- OpenAI API key

## Environment Variables

Create `.env` at the project root (or in `backend/`):

```env
OPENAI_API_KEY=sk-...
JWT_SECRET=replace-with-strong-random-secret-at-least-32-chars

# Optional overrides
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
ASK_RATE_LIMIT=20/minute

# PostgreSQL (matches docker-compose defaults)
DATABASE_URL=postgresql://admin:password@localhost:5433/pharmanet

# Collections
LEGAL_COLLECTION_NAME=legal
DRUG_COLLECTION_NAME=drug

# Optional: Hugging Face legal dataset
HF_LEGAL_DATASET_REPO=th1nhng0/vietnamese-legal-documents
HF_LEGAL_METADATA_CONFIG=metadata
HF_LEGAL_CONTENT_CONFIG=content
HF_LEGAL_SPLIT=data
HF_LEGAL_BATCH_SIZE=25
HF_LEGAL_SKIP_SUMMARY_DEFAULT=true

# Optional: auto-admin at registration (JSON array)
ADMIN_EMAILS=["admin@example.com"]
```

> **Note**: The backend will refuse to start if `JWT_SECRET` is left as `CHANGE_ME`. `/ask` and ingestion require a valid `OPENAI_API_KEY`.

## Install

```bash
# Backend
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Frontend
cd frontend
npm install
cd ..
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

**3. Start frontend dev server (optional, for hot reload):**

```bash
cd frontend
npm run dev
```

Vite proxies API calls to `http://localhost:8000`.

## Run (Production — Single Server)

Build the frontend and let FastAPI serve it as static files:

```bash
cd frontend && npm run build
cd ../backend && uvicorn app:app --host 0.0.0.0 --port 8000
```

Access:
- App: `http://localhost:8000/app/`
- Admin: `http://localhost:8000/admin/`
- API docs: `http://localhost:8000/docs`

## API Endpoints

### Auth
| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login, returns JWT |
| PUT | `/auth/password` | Change own password |

### Chat
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ask` | Main RAG endpoint (rate-limited) |
| POST | `/chat/sessions` | Create session |
| GET | `/chat/sessions` | List sessions |
| DELETE | `/chat/sessions/{id}` | Delete session |
| GET | `/chat/sessions/{id}/messages` | Get messages |
| POST | `/drug-price` | Live price lookup |

### Ingest
| Method | Path | Description |
|---|---|---|
| POST | `/ingest-file` | Upload PDF/DOCX (`?async=true` for background job) |
| POST | `/ingest-hf-legal` | Ingest from Hugging Face legal dataset (async by default) |
| GET | `/ingest-jobs/{job_id}` | Poll async job status |
| POST | `/ingest-jobs/{job_id}/cancel` | Cancel async job |

### Admin (requires admin role)
| Method | Path | Description |
|---|---|---|
| POST | `/feedback` | Submit feedback (public) |
| GET | `/admin/feedback` | View all feedback |
| GET | `/admin/analytics` | 30-day dashboard stats + token usage |
| GET | `/admin/collections` | List collections |
| GET | `/admin/collections/{name}/docs` | List sources in collection |
| DELETE | `/admin/docs` | Delete document by source |
| GET | `/admin/users` | List users |
| PUT | `/admin/users/{id}/role` | Set admin flag |
| PUT | `/admin/users/{id}/password` | Reset user password |
| DELETE | `/admin/users/{id}` | Delete user |

## Retrieval Pipeline

Each `/ask` request goes through:

1. **Intent classification** (`supervisor.py`) — routes to `legal`, `drug`, or both collections
2. **Query reformulation** — rewrites follow-up questions into standalone queries using conversation history
3. **Query expansion** — generates 3 semantic variations via LLM
4. **Hybrid search** — for each collection:
   - Dense: cosine similarity via pgvector (IVFFlat index)
   - Sparse: BM25 re-scoring using stored vocab + Robertson IDF
   - Combined score: `0.7 × dense + 0.3 × normalized_sparse`
5. **Reranking** — CrossEncoder (`mMiniLMv2-L12-H384`) selects top 5
6. **Parent fetch** — retrieves full parent article text for reranked child chunks
7. **LLM generation** — answer generated from context + conversation history

## Ingestion Pipeline

Each uploaded PDF or DOCX goes through:

1. **Load** — PDF via Docling (→ Markdown), DOCX via python-docx
2. **Parse**
   - Vietnamese legal documents: split on `Điều`/`Article`/`Section` markers (≥5 required) → one parent per article
   - Non-legal: semantic/header-based splitting
   - Articles exceeding 1200 tokens are further split with `RecursiveCharacterTextSplitter`
3. **Child chunking** — each parent split on numbered clause markers (`1.`, `2.`, …); falls back to sentences
4. **Metadata enrichment** — LLM generates `summary` and `target_question` per parent (skippable via `skip_summary=true`)
5. **Embedding** — OpenAI `text-embedding-3-small` (or HuggingFace alternative)
6. **Sparse vectors** — BM25 indices/values computed from corpus vocabulary
7. **Store** — dense + sparse vectors → `vector_chunks`; parent metadata → `vector_parents`; BM25 vocab/stats stored per collection

### HF Legal Dataset Ingest

```bash
POST /ingest-hf-legal?collection_name=legal&mode=proxy_strict&skip_summary=true
```

Key params:
- `mode=proxy_strict` — whitelist filter by legal type + latest-version dedupe by `issuance_date`
- `only_pharma_related=true` — filter to pharma-tagged documents only
- `write_quarantine=true` — store rejected rows in `<collection_name>_quarantine`
- `skip_summary=true` — skip LLM summary generation (recommended for bulk ingest)

HF metadata fields stored: `hf_id`, `title`, `legal_type`, `legal_sectors`, `issuing_authority`, `issuance_date`, `url`, `source_dataset`, `is_pharma_related`, `pharma_tags`.

### CLI ingest (legacy scripts)

```bash
cd backend
python hf_legal_ingest.py --collection legal --skip-summary true --limit 500
python hf_legal_ingest.py --collection legal --filter proxy_strict --skip-summary true --limit 500
```

## Drug Data Scraper

`backend/scripts/scrape_longchau.py` scrapes `nhathuoclongchau.com.vn` using Playwright and populates the `drug_list` and `drug_inventory` tables directly in PostgreSQL.

```bash
cd backend/scripts
python scrape_longchau.py
```

Supports resumable runs via `scrape_progress.json`.

## RAGAS Evaluation

```bash
source .venv/bin/activate
cd backend
python -m eval.ragas_runner \
  --dataset eval/datasets/curated.jsonl \
  --metrics faithfulness,answer_relevancy,context_precision,context_recall
```

With synthetic expansion:

```bash
python -m eval.ragas_runner \
  --dataset eval/datasets/curated.jsonl \
  --include-synthetic \
  --synthetic-max-per-collection 40
```

Outputs written to `eval_results/`:
- `<run>.summary.json` — aggregate metrics + run metadata
- `<run>.raw.json` / `<run>.raw.csv` — per-row scores
- `<run>.pipeline_outputs.json` — question, retrieved contexts, generated answer

Optional eval settings in `.env`:

```env
EVAL_JUDGE_MODEL=gpt-4o-mini
EVAL_MAX_WORKERS=2
EVAL_TIMEOUT_SECONDS=90
EVAL_MAX_SAMPLES=0
EVAL_OUTPUT_DIR=eval_results
```

## Operational Notes

- `/ask` rate limit uses JWT `sub` when available, falls back to IP
- Runtime directories created automatically: `uploads/`
- LLM token usage is tracked daily in `llm_usage_daily` and visible in the admin analytics dashboard
- PostgreSQL data is persisted in a named Docker volume (`postgres_data`)
