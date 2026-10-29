"""PostgreSQL-backed account and session storage for the ``pokerpilot`` schema.

The database URL is read only on the server. Existing ``users`` and
``sessions`` tables are used directly; this module never creates parallel
tables or performs schema changes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

SESSION_SECONDS = 60 * 60 * 24 * 7
SCHEMA = "pokerpilot"


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured for PostgreSQL authentication.")
    return url


def connect():
    """Open a transactional PostgreSQL connection using psycopg 3."""
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise RuntimeError("Install backend requirements to use PostgreSQL authentication.") from exc

    # psycopg commits on a successful context exit, rolls back on exceptions,
    # and closes the connection. Require TLS even if a URL omitted sslmode.
    return psycopg.connect(_database_url(), sslmode="require", row_factory=dict_row)


def verify_schema() -> None:
    """Check the pre-existing tables/columns without creating or altering them."""
    with connect() as db:
        rows = db.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name IN (%s, %s)",
            (SCHEMA, "users", "sessions"),
        ).fetchall()
        indexes = db.execute(
            "SELECT tablename, indexdef FROM pg_indexes "
            "WHERE schemaname = %s AND tablename IN (%s, %s)",
            (SCHEMA, "users", "sessions"),
        ).fetchall()

    columns: dict[str, set[str]] = {"users": set(), "sessions": set()}
    for row in rows:
        columns[row["table_name"]].add(row["column_name"])
    required = {
        "users": {"id", "username", "email", "password_hash", "created_at"},
        "sessions": {"token_hash", "user_id", "expires_at"},
    }
    missing = {
        table: sorted(names - columns.get(table, set()))
        for table, names in required.items()
        if names - columns.get(table, set())
    }
    if missing:
        details = "; ".join(f"{table}: {', '.join(names)}" for table, names in missing.items())
        raise RuntimeError(f"Existing pokerpilot auth tables are missing required columns ({details}).")
    definitions = {row["tablename"]: [] for row in indexes}
    for row in indexes:
        definitions[row["tablename"]].append(row["indexdef"].lower())
    for table, column in (("users", "username"), ("users", "email")):
        if not any("unique index" in definition and f"lower({column})" in definition
                   for definition in definitions.get(table, [])):
            raise RuntimeError(
                f"Existing pokerpilot.users table needs a case-insensitive unique index on {column}. "
                "Apply backend/migrations/001_auth_uniqueness.sql after checking for duplicate values."
            )


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$16384$8$1$" + base64.urlsafe_b64encode(salt).decode() + "$" + base64.urlsafe_b64encode(key).decode()


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_b64, key_b64 = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_b64)
        expected = base64.urlsafe_b64decode(key_b64)
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def public_user(row: dict[str, Any]) -> dict[str, Any]:
    user_id = row["id"]
    if not isinstance(user_id, (int, str)):
        user_id = str(user_id)
    return {"id": user_id, "username": row["username"], "email": row["email"]}


def find_user(identifier: str) -> dict[str, Any] | None:
    with connect() as db:
        return db.execute(
            "SELECT id, username, email, password_hash FROM pokerpilot.users "
            "WHERE lower(username) = lower(%s) OR lower(email) = lower(%s) LIMIT 1",
            (identifier, identifier),
        ).fetchone()


def create_user(username: str, email: str, password: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    with connect() as db:
        return db.execute(
            "INSERT INTO pokerpilot.users (username, email, password_hash, created_at) "
            "VALUES (%s, %s, %s, %s) RETURNING id, username, email, password_hash",
            (username, email, hash_password(password), now),
        ).fetchone()


def new_session(user_id: Any) -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires = datetime.now(timezone.utc) + timedelta(seconds=SESSION_SECONDS)
    with connect() as db:
        db.execute("DELETE FROM pokerpilot.sessions WHERE expires_at <= %s", (datetime.now(timezone.utc),))
        db.execute(
            "INSERT INTO pokerpilot.sessions (token_hash, user_id, expires_at) VALUES (%s, %s, %s)",
            (digest, user_id, expires),
        )
    return token, expires


def session_user(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    with connect() as db:
        return db.execute(
            "SELECT users.id, users.username, users.email "
            "FROM pokerpilot.sessions AS sessions "
            "JOIN pokerpilot.users AS users ON users.id = sessions.user_id "
            "WHERE sessions.token_hash = %s AND sessions.expires_at > %s LIMIT 1",
            (digest, now),
        ).fetchone()


def revoke_session(token: str | None) -> None:
    if not token:
        return
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with connect() as db:
        db.execute("DELETE FROM pokerpilot.sessions WHERE token_hash = %s", (digest,))
