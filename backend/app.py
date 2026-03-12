"""RAG Chatbot API — entry point.

Thin app configuration only: middleware, startup hook, router registration,
and static file mounts.  All endpoint logic lives in routers/.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from jose import jwt
from fastapi import Request

from config import get_settings
from routers import auth, chat, admin, ingest_router


# ---------------------------------------------------------------------------
# Rate-limiter key: per JWT sub (user) when present, fallback to IP
# ---------------------------------------------------------------------------

def _rate_limit_key(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1]
        try:
            settings = get_settings()
            payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm], options={"verify_exp": False})
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
        except Exception:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Chatbot API",
    description="Semantic RAG with reasoning & retrieval",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded. {exc.detail}"}),
)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _validate_settings() -> None:
    """Validate critical secrets once at boot time."""
    logger = logging.getLogger(__name__)
    settings = get_settings()

    if settings.jwt_secret == "CHANGE_ME":
        raise RuntimeError(
            "JWT_SECRET is still the default value 'CHANGE_ME'. "
            "Set a strong random secret in your .env file: JWT_SECRET=<at-least-32-random-chars>"
        )

    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set. The /ask endpoint and ingestion will fail until it is configured.")


# ---------------------------------------------------------------------------
# Apply rate limit to /ask (needs access to the limiter singleton here)
# ---------------------------------------------------------------------------

_original_ask = chat.router.routes  # reference kept for patching below

# Patch the /ask endpoint with the rate limiter decorator at app level
# so the limiter singleton is shared (limiter is defined in this module).
for route in chat.router.routes:
    if hasattr(route, "path") and route.path == "/ask":
        route.endpoint = limiter.limit(lambda: get_settings().ask_rate_limit)(route.endpoint)
        break


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(ingest_router.router)


@app.get("/")
def root():
    return {"message": "RAG Chatbot API. Use POST /ask with body: {\"question\": \"...\"}. UI: GET /app/"}


# ---------------------------------------------------------------------------
# Static file mounts (must come after routes)
# ---------------------------------------------------------------------------

_upload_dir = Path(__file__).resolve().parent.parent / "uploads"
_upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(_upload_dir)), name="uploads")

# Serve frontend at /app — from the Vite build output (frontend/dist/)
# Run `cd frontend && npm install && npm run build` once to generate it.
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
    # Serve same SPA at /admin (FastAPI re-routes, SPA handles client-side routing)
    app.mount("/admin", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend-admin")
else:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "frontend/dist/ not found — UI will not be served. "
        "Run: cd frontend && npm install && npm run build"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
