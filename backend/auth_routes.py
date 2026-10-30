"""One shared Flask authentication blueprint for both PokerPilot game apps."""

from __future__ import annotations

import hmac
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from functools import wraps
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from flask import Blueprint, current_app, g, jsonify, make_response, request, send_file, session

from backend import auth

bp = Blueprint("pokerpilot_auth", __name__)
SESSION_COOKIE = "pokerpilot_session"
CSRF_COOKIE = "pokerpilot_csrf"
SESSION_SECONDS = 60 * 60 * 24 * 7
AUTH_PAGE = Path(__file__).with_name("auth.html")
AUTH_CSS = Path(__file__).with_name("auth.css")

_attempts: dict[str, deque[float]] = defaultdict(deque)
_attempts_lock = threading.Lock()


def _secure_cookie() -> bool:
    return os.getenv("POKERPILOT_COOKIE_SECURE", "0").lower() in {"1", "true", "yes"}


def _set_cookie_headers(response, token: str, csrf: str) -> None:
    secure = _secure_cookie()
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                        secure=secure, samesite="Lax", path="/")
    response.set_cookie(CSRF_COOKIE, csrf, max_age=SESSION_SECONDS, httponly=False,
                        secure=secure, samesite="Lax", path="/")


def _api_error(message: str, status: int):
    return jsonify({"detail": message}), status


def get_user(token: str | None = None):
    return auth.session_user(token if token is not None else request.cookies.get(SESSION_COOKIE))


def csrf_valid() -> bool:
    cookie = request.cookies.get(CSRF_COOKIE, "")
    header = request.headers.get("X-CSRF-Token", "")
    if not cookie or not header or not hmac.compare_digest(cookie, header):
        return False
    origin = request.headers.get("Origin")
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != request.host.lower():
            return False
    return True


def require_auth(func: Callable):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            user = get_user()
        except Exception:
            current_app.logger.exception("Authentication database access failed.")
            return _api_error("Authentication storage is unavailable.", 503)
        if user is None:
            return _api_error("Please sign in to continue.", 401)
        g.auth_user = user
        return func(*args, **kwargs)

    return wrapped


def require_csrf(func: Callable):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if not csrf_valid():
            return _api_error("CSRF token is missing or invalid.", 403)
        return func(*args, **kwargs)

    return wrapped


def register_auth(app) -> None:
    app.register_blueprint(bp)
    app.extensions.setdefault("pokerpilot_auth_logout_hooks", [])


def add_logout_hook(app, callback: Callable) -> None:
    app.extensions.setdefault("pokerpilot_auth_logout_hooks", []).append(callback)


def _rate_limit_key(identifier: str) -> str:
    ip = request.remote_addr or "unknown"
    return f"{ip}|{identifier.casefold()}"


def _check_rate_limit(key: str) -> bool:
    now = time.monotonic()
    with _attempts_lock:
        keys = (key, "ip:" + key.split("|", 1)[0])
        for item in keys:
            values = _attempts[item]
            while values and now - values[0] > 300:
                values.popleft()
            if len(values) >= 8:
                return False
    return True


def _record_failure(key: str) -> None:
    now = time.monotonic()
    with _attempts_lock:
        _attempts[key].append(now)
        _attempts["ip:" + key.split("|", 1)[0]].append(now)


def _clear_failures(key: str) -> None:
    with _attempts_lock:
        _attempts.pop(key, None)
        _attempts.pop("ip:" + key.split("|", 1)[0], None)


@bp.get("/auth")
def auth_page():
    return send_file(AUTH_PAGE)


@bp.get("/auth.css")
def auth_css():
    return send_file(AUTH_CSS, mimetype="text/css")


@bp.get("/api/auth/csrf")
def csrf_seed():
    token = secrets.token_urlsafe(24)
    response = make_response(jsonify({"csrf_token": token}))
    response.set_cookie(CSRF_COOKIE, token, max_age=SESSION_SECONDS, httponly=False,
                        secure=_secure_cookie(), samesite="Lax", path="/")
    return response


@bp.get("/api/auth/health")
def auth_health():
    try:
        auth.verify_schema()
    except Exception:
        current_app.logger.exception("Supabase auth schema check failed.")
        return _api_error("Supabase authentication storage is unavailable or incompatible.", 503)
    return jsonify({"status": "ok", "schema": "pokerpilot"})


@bp.post("/api/auth/signup")
@require_csrf
def signup():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "")
    email = data.get("email", "")
    password = data.get("password", "")
    if not all(isinstance(value, str) for value in (username, email, password)):
        return _api_error("Enter a username, email, and password.", 422)
    username = username.strip()
    email = email.strip().lower()
    if not re.fullmatch(r"[A-Za-z0-9_]{3,24}", username):
        return _api_error("Username must be 3–24 letters, numbers, or underscores.", 422)
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return _api_error("Enter a valid email address.", 422)
    if not 12 <= len(password) <= 128:
        return _api_error("Password must be 12–128 characters.", 422)
    try:
        row = auth.create_user(username, email, password)
        token, _ = auth.new_session(row["id"])
        csrf = secrets.token_urlsafe(24)
        response = make_response(jsonify({"user": auth.public_user(row)}))
        _set_cookie_headers(response, token, csrf)
        return response
    except Exception as exc:
        if getattr(exc, "sqlstate", None) == "23505":
            return _api_error("That username or email is already registered.", 409)
        current_app.logger.exception("Account creation failed.")
        return _api_error("Authentication storage is unavailable.", 503)


@bp.post("/api/auth/login")
@require_csrf
def login():
    data = request.get_json(silent=True) or {}
    identifier = data.get("identifier", "")
    password = data.get("password", "")
    if not isinstance(identifier, str) or not isinstance(password, str) or not identifier or not password:
        return _api_error("Enter your username/email and password.", 422)
    identifier = identifier.strip()
    key = _rate_limit_key(identifier)
    if not _check_rate_limit(key):
        return _api_error("Too many login attempts. Try again in a few minutes.", 429)
    try:
        row = auth.find_user(identifier)
        if row is None or not auth.verify_password(password, row["password_hash"]):
            _record_failure(key)
            return _api_error("Username/email or password is incorrect.", 401)
        _clear_failures(key)
        token, _ = auth.new_session(row["id"])
        csrf = secrets.token_urlsafe(24)
        response = make_response(jsonify({"user": auth.public_user(row)}))
        _set_cookie_headers(response, token, csrf)
        return response
    except Exception:
        current_app.logger.exception("Login failed because authentication storage is unavailable.")
        return _api_error("Authentication storage is unavailable.", 503)


@bp.get("/api/auth/me")
def me():
    try:
        row = get_user()
    except Exception:
        current_app.logger.exception("Session lookup failed.")
        return _api_error("Authentication storage is unavailable.", 503)
    if row is None:
        return _api_error("Please sign in to continue.", 401)
    return jsonify({"user": auth.public_user(row)})


@bp.post("/api/auth/logout")
@require_auth
@require_csrf
def logout():
    try:
        auth.revoke_session(request.cookies.get(SESSION_COOKIE))
    except Exception:
        current_app.logger.exception("Session revocation failed.")
        return _api_error("Authentication storage is unavailable.", 503)
    for callback in current_app.extensions.get("pokerpilot_auth_logout_hooks", []):
        try:
            callback(g.auth_user)
        except Exception:
            current_app.logger.exception("An app could not close a logged-out connection.")
    session.clear()
    response = make_response(jsonify({"ok": True}))
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True,
                           secure=_secure_cookie(), samesite="Lax")
    response.delete_cookie(CSRF_COOKIE, path="/", secure=_secure_cookie(), samesite="Lax")
    return response
