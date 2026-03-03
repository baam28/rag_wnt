# RAG Chatbot – Web Nhà Thuốc

A RAG (Retrieval-Augmented Generation) chatbot with semantic chunking, parent-child hierarchy, hybrid search, re-ranking, and a full-featured web UI for both users and admins.

## Project Layout

```
rag_wnt/
├── backend/
│   ├── app.py            # FastAPI endpoints, feedback storage, static file serving
│   ├── config.py         # Settings (.env at project root)
│   ├── ingest.py         # PDF/DOCX → parent/child chunks → Qdrant + sparse vectors
│   ├── retriever.py      # Query expansion, hybrid search, re-ranking
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── index.html        # Entry point (React via CDN + Babel)
│   ├── styles.css        # Full application styles
│   └── app.js            # React app (chat, admin, feedback)
├── .env                  # OPENAI_API_KEY (project root, not committed)
├── requirements.txt      # Points to backend/requirements.txt
├── qdrant_db/            # Vector DB (created at runtime)
├── uploads/              # Uploaded documents (served at /uploads)
├── data/                 # Runtime data
└── feedback.json         # User feedback storage (created at runtime)
```

## Stack

- **Framework:** LangChain
- **Vector Store:** Qdrant (embedded, local)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** OpenAI (e.g. `gpt-4o-mini`)
- **Frontend:** React 18 (CDN, no build step) + vanilla CSS

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file at **project root**:

```
OPENAI_API_KEY=sk-your-key
```

## Run

```bash
cd backend && uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- **API:** `http://localhost:8000/`
- **Web UI:** `http://localhost:8000/app/`

## Features

### Chat Interface
- Multi-session chat with local storage persistence
- Animated typing effect for assistant responses
- Chat history sidebar with session management (create, switch, delete)
- Scrollable chat area with fixed input bar

### Source Display & Document Viewer
- Deduplicated source references grouped by document name and collection
- Expandable source cards showing extracted text chunks
- Clickable document names to view originals:
  - **PDF:** Opens in a new tab at the referenced page
  - **DOCX/other:** Opens a modal with extracted text content and download option
- Source summary display from LLM metadata

### User Feedback System
- **Thumbs up / thumbs down** buttons on every assistant response
- Optional text feedback form when rating negatively
- Feedback is persisted to the backend (`feedback.json`) with question, answer, rating, comment, and timestamp

### Admin Panel
- **Document Ingest:** Upload PDF/DOCX files into named collections with optional LLM summary generation
- **Database Management:**
  - View all collections with expand/collapse to see individual documents
  - Delete individual documents, entire collections, or the whole database (with confirmation)
  - View uploaded documents directly from the admin panel
- **Feedback Analytics:**
  - Summary stats: total ratings, thumbs up count, thumbs down count, positive rate percentage
  - "Thumbs down" tab: view all negatively rated Q&A pairs with user comments
  - "All" tab: browse all feedback entries with full question and answer text

### UI/UX
- Professional two-panel layout (sidebar + main content)
- Responsive admin panel with 2-column layout (ingest sidebar + database/feedback main area)
- High-contrast navigation buttons
- Confirmation dialogs for destructive actions

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Ask a question with optional chat history |
| `POST` | `/ingest-file` | Upload and ingest a PDF/DOCX file |
| `GET` | `/collections` | List all collections |
| `GET` | `/collections/{name}/documents` | List documents in a collection |
| `DELETE` | `/collections/{name}` | Delete a collection |
| `DELETE` | `/collections/{name}/documents/{source}` | Delete a document |
| `POST` | `/db/clear` | Clear entire Qdrant database |
| `POST` | `/feedback` | Submit user feedback (rating + comment) |
| `GET` | `/admin/feedback` | Get feedback analytics and entries |
| `GET` | `/uploads/{filename}` | Serve uploaded documents |

## Notes

- All paths in `backend/config.py` are relative to the **project root**.
- Hybrid search uses Qdrant native sparse vectors; re-ingest to enable sparse indexing on new collections.
- Frontend uses React via CDN with Babel transpilation — no build step required.
- Cache-busting query parameters on CSS/JS links in `index.html` ensure updates are picked up.
