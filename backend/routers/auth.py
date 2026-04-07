"""Auth router: register, login, change password."""

import time

from fastapi import APIRouter, Depends, HTTPException

from config import get_settings
from deps import (
    CurrentUser,
    RegisterRequest,
    LoginRequest,
    AuthResponse,
    UserOut,
    ChangePasswordRequest,
    hash_password,
    verify_password,
    get_current_user,
    _create_jwt_for_user,
)
from pg_client import get_users_collection

router = APIRouter(tags=["auth"])


def _make_current_user(doc: dict) -> CurrentUser:
    return CurrentUser(
        id=str(doc["_id"]),
        username=doc.get("username", ""),
        email=doc.get("email", ""),
        is_admin=bool(doc.get("is_admin", False)),
    )


@router.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    """Register a new local user with username/password."""
    username = (req.username or "").strip().lower()
    email = (req.email or "").strip().lower()
    password = req.password or ""
    if not username or not email or not password:
        raise HTTPException(status_code=400, detail="Username, email and password are required")
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    users = get_users_collection()
    if users.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    if users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    settings = get_settings()
    admin_usernames = {e.strip().lower() for e in (settings.admin_emails or [])}
    is_admin = username in admin_usernames or email in admin_usernames

    doc = {
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "is_admin": is_admin,
        "created_at": now,
        "last_login_at": now,
    }
    res = users.insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    user = _make_current_user(doc)
    token = _create_jwt_for_user(user)
    return AuthResponse(token=token, user=UserOut(id=user.id, username=user.username, is_admin=user.is_admin))


@router.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """Login with username/password and return a JWT."""
    login_id = (req.username or "").strip().lower()
    password = req.password or ""
    if not login_id or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    users = get_users_collection()
    doc = users.find_one({"username": login_id}) or users.find_one({"email": login_id})
    if not doc or "password_hash" not in doc:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not verify_password(password, doc.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Update last_login_at
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    users.update_one({"_id": doc["_id"]}, {"$set": {"last_login_at": now}})

    user = _make_current_user(doc)
    token = _create_jwt_for_user(user)
    return AuthResponse(token=token, user=UserOut(id=user.id, username=user.username, is_admin=user.is_admin))


@router.put("/auth/password")
def change_password(
    req: ChangePasswordRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Change the current user's password."""
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")

    users = get_users_collection()
    doc = users.find_one({"_id": current_user.id})
    if not doc or "password_hash" not in doc:
        raise HTTPException(status_code=401, detail="User not found")
    if not verify_password(req.old_password, doc.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Incorrect old password")

    users.update_one({"_id": current_user.id}, {"$set": {"password_hash": hash_password(req.new_password)}})
    return {"message": "Password updated successfully"}
