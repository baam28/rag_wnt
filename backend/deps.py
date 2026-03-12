"""Shared FastAPI dependencies, Pydantic models, and auth helpers.

Imported by all routers to avoid repeating boilerplate in each file.
"""

import time
from typing import Any, Optional

import bcrypt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel

from config import get_settings


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

security = HTTPBearer()


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pydantic models shared across routers
# ---------------------------------------------------------------------------

class CurrentUser(BaseModel):
    id: str
    email: str
    is_admin: bool


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    has_context: bool
    collection_name: Optional[str] = None
    price_data: Optional[dict[str, Any]] = None


class IngestResponse(BaseModel):
    file: str
    collection_name: str
    num_parents: int
    num_children: int
    total_chunks_in_db: int


class ChatSession(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    sources: Optional[list[dict[str, Any]]] = None
    priceData: Optional[dict[str, Any]] = None
    feedback: Optional[int] = None
    feedbackComment: Optional[str] = None


class CollectionInfo(BaseModel):
    name: str


class DocumentInfo(BaseModel):
    source: str
    parent_count: int


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: str
    username: str
    is_admin: bool


class AuthResponse(BaseModel):
    token: str
    user: UserOut


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class DeleteDocumentRequest(BaseModel):
    collection_name: str
    source: str


class DrugPriceRequest(BaseModel):
    drug_name: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str  # "up" or "down"
    comment: Optional[str] = None
    session_id: Optional[str] = None


class RoleUpdateRequest(BaseModel):
    is_admin: bool


class AdminSetPasswordRequest(BaseModel):
    new_password: str


class IngestJobResponse(BaseModel):
    job_id: str


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error" | "cancelled"
    phase: Optional[str] = None
    current: Optional[int] = None
    total: Optional[int] = None
    message: Optional[str] = None
    result: Optional[IngestResponse] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _user_from_doc(doc: dict[str, Any]) -> CurrentUser:
    return CurrentUser(
        id=str(doc["_id"]),
        email=doc.get("email", ""),
        is_admin=bool(doc.get("is_admin") or doc.get("roles", {}).get("admin")),
    )


def _create_jwt_for_user(user: CurrentUser) -> str:
    settings = get_settings()
    payload = {
        "sub": user.id,
        "email": user.email,
        "admin": user.is_admin,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


# ---------------------------------------------------------------------------
# FastAPI dependency functions
# ---------------------------------------------------------------------------

def get_current_user(
    creds: HTTPAuthorizationCredentials = Security(security),
) -> CurrentUser:
    settings = get_settings()
    token = creds.credentials
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("sub")
    email = payload.get("email")
    is_admin = bool(payload.get("admin"))

    if not user_id or not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    return CurrentUser(id=user_id, email=email, is_admin=is_admin)


def get_current_admin(user: CurrentUser = Security(get_current_user)) -> CurrentUser:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
