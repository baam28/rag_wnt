# PharmaAI RAG (Drug + Legal Assistant)

A full-stack RAG assistant for Vietnamese pharmacy workflows:
- Drug information Q&A (`drug` collection)
- Legal/pharmacy regulation Q&A (`legal` collection)
- Real-time drug price lookup integration
- Authenticated chat sessions with admin dashboard

## Tech Stack

- Backend: FastAPI, LangChain, Qdrant, MongoDB
- LLM/Embeddings: OpenAI (`gpt-4o-mini`, `text-embedding-3-small` by default)
- Retrieval: hybrid dense + sparse, reranking with CrossEncoder
- Ingestion: PDF (Docling), DOC/DOCX (python-docx)
- Frontend: React + Vite (SPA served from `frontend/dist`)

## Current Architecture

- `backend/app.py`
  - Registers routers: auth, chat, admin, ingest
  - Enforces startup secret check (`JWT_SECRET` must not be `CHANGE_ME`)
  - Adds request rate limiting (`/ask`) and CORS
  - Serves built frontend at `/app` and `/admin`
- `backend/routers/auth.py`
  - Local register/login/password-change with JWT
- `backend/routers/chat.py`
  - `/ask` federates intent-based retrieval + optional live price context
  - Mongo-backed chat sessions/messages
- `backend/routers/ingest_router.py`
  - Sync ingest and async ingest jobs (`/ingest-jobs/{job_id}`)
- `backend/routers/admin.py`
  - Collection/doc management, user admin, feedback, analytics, DB clear

## Project Layout

```text
rag_wnt/
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── supervisor.py / agents.py / prompts.py
│   ├── routers/
│   │   ├── auth.py
│   │   ├── chat.py
│   │   ├── ingest_router.py
│   │   └── admin.py
│   └── requirements.txt
├── frontend/
│   ├── app.js
│   ├── chat_page.js
│   ├── admin_page.js
│   ├── landing.js
│   ├── styles.css
│   ├── main.jsx
│   ├── vite.config.js
│   └── package.json
├── docs/                # Source docs for ingestion
├── uploads/             # Uploaded files (runtime)
├── qdrant_db/           # Vector DB data (runtime)
├── feedback.json        # Feedback store (runtime)
├── llm_usage.json       # LLM usage stats (runtime)
└── README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB running locally (default: `mongodb://localhost:27017`)
- OpenAI API key

## Environment Variables

Create `.env` at project root:

```env
OPENAI_API_KEY=sk-...
JWT_SECRET=replace-with-strong-random-secret-at-least-32-chars

# Optional overrides
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
ASK_RATE_LIMIT=20/minute
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=rag_chatbot
LEGAL_COLLECTION_NAME=legal
DRUG_COLLECTION_NAME=drug

# Optional: users auto-marked admin at registration
# JSON array format is safest with pydantic-settings
ADMIN_EMAILS=["admin@example.com"]
```

Important:
- Backend startup fails if `JWT_SECRET` is left as `CHANGE_ME`.
- If `OPENAI_API_KEY` is missing, `/ask` and summary generation will fail.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

## Run (Development)

1) Start MongoDB.

2) Start backend:

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

3) Start frontend dev server (optional, for hot reload):

```bash
cd frontend
npm run dev
```

Vite proxies API calls to `http://localhost:8000`.

## Run (Single-server via FastAPI static mount)

Build frontend and let FastAPI serve it:

```bash
cd frontend
npm run build
cd ../backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then access:
- App: `http://localhost:8000/app/`
- Admin route: `http://localhost:8000/admin/` (same SPA)
- API docs: `http://localhost:8000/docs`

## Main Features

- JWT auth (register/login/change password)
- Chat sessions per user (MongoDB)
- Intent-based routing across legal/drug collections
- Combined answering when question needs both domains
- Price-agent integration for Vietnamese pharmacy price lookups
- Async ingestion jobs with polling + cancel
- Admin tools:
  - Collection/document management in Qdrant
  - Clear full vector DB
  - User role/password management
  - Feedback review
  - 30-day analytics (users/sessions/messages/feedback + token usage)

## API Endpoints

### Auth
- `POST /auth/register`
- `POST /auth/login`
- `PUT /auth/password`

### Chat
- `GET /health`
- `POST /ask`
- `POST /chat/sessions`
- `GET /chat/sessions`
- `DELETE /chat/sessions/{session_id}`
- `GET /chat/sessions/{session_id}/messages`
- `POST /drug-price`

### Ingest
- `POST /ingest-file` (supports `?async=true`)
- `GET /ingest-jobs/{job_id}`
- `POST /ingest-jobs/{job_id}/cancel`

### Feedback + Admin
- `POST /feedback` (public)
- `GET /admin/feedback`
- `GET /admin/analytics`
- `GET /admin/collections`
- `GET /admin/docs?collection_name=...`
- `DELETE /admin/docs`
- `DELETE /admin/collections/{collection_name}`
- `POST /db/clear`
- `GET /admin/users`
- `PUT /admin/users/{user_id}/role`
- `PUT /admin/users/{user_id}/password`
- `DELETE /admin/users/{user_id}`

## Ingestion Notes

- Supported files: `.pdf`, `.doc`, `.docx`
- PDF parsing uses Docling markdown export.
- Chunks are stored in Qdrant with parent/child metadata and sparse vocab side files.
- To skip LLM summary generation during ingest, set `skip_summary=true` in form data.

## Operational Notes

- Runtime files created automatically:
  - `uploads/`
  - `qdrant_db/`
  - `feedback.json`
  - `llm_usage.json`
- Keep `qdrant_db/` and MongoDB data persisted in production.
- `/ask` rate limit uses JWT `sub` when available; falls back to IP.
