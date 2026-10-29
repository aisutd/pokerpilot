import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

from backend import auth


class FakeResult:
    def __init__(self, row=None):
        self.row = row

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.row or []


class FakeConnection:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=()):
        self.calls.append((query, params))
        return FakeResult(self.responses.pop(0) if self.responses else None)


class AuthStorageTests(unittest.TestCase):
    def test_password_hash_uses_salted_scrypt(self):
        one = auth.hash_password("correct horse battery staple")
        two = auth.hash_password("correct horse battery staple")
        self.assertNotEqual(one, two)
        self.assertTrue(auth.verify_password("correct horse battery staple", one))
        self.assertFalse(auth.verify_password("wrong password", one))

    def test_queries_use_existing_schema_and_parameter_placeholders(self):
        row = {"id": 7, "username": "siri", "email": "siri@example.test", "password_hash": "hash"}
        db = FakeConnection([row])
        with patch.object(auth, "connect", return_value=db):
            result = auth.find_user("siri@example.test' OR true --")
        self.assertEqual(result, row)
        query, params = db.calls[0]
        self.assertIn("pokerpilot.users", query)
        self.assertIn("%s", query)
        self.assertNotIn("siri@example.test", query)
        self.assertEqual(params, ("siri@example.test' OR true --", "siri@example.test' OR true --"))

    def test_session_storage_hashes_tokens_and_checks_expiry(self):
        db = FakeConnection()
        with patch.object(auth, "connect", return_value=db):
            token, expires = auth.new_session(17)
        self.assertIsInstance(expires, datetime)
        self.assertGreater(expires, datetime.now(timezone.utc))
        self.assertEqual(len(db.calls), 2)
        insert_query, params = db.calls[1]
        self.assertIn("pokerpilot.sessions", insert_query)
        self.assertEqual(params[1], 17)
        self.assertNotEqual(params[0], token)
        self.assertEqual(len(params[0]), 64)

        user = {"id": 17, "username": "siri", "email": "siri@example.test"}
        lookup = FakeConnection([user])
        with patch.object(auth, "connect", return_value=lookup):
            self.assertEqual(auth.session_user(token), user)
        query, lookup_params = lookup.calls[0]
        self.assertIn("expires_at > %s", query)
        self.assertNotEqual(lookup_params[0], token)

    def test_schema_check_requires_tables_and_case_insensitive_unique_indexes(self):
        columns = [
            {"table_name": "users", "column_name": column}
            for column in ("id", "username", "email", "password_hash", "created_at")
        ] + [
            {"table_name": "sessions", "column_name": column}
            for column in ("token_hash", "user_id", "expires_at")
        ]
        indexes = [
            {"tablename": "users", "indexdef": "CREATE UNIQUE INDEX users_username_lower_unique ON pokerpilot.users (lower(username))"},
            {"tablename": "users", "indexdef": "CREATE UNIQUE INDEX users_email_lower_unique ON pokerpilot.users (lower(email))"},
        ]

        @contextmanager
        def connected():
            class MetadataConnection:
                def execute(self, query, params):
                    return FakeResult(columns if "information_schema" in query else indexes)
            yield MetadataConnection()

        with patch.object(auth, "connect", connected):
            auth.verify_schema()

        indexes.pop()
        with patch.object(auth, "connect", connected):
            with self.assertRaisesRegex(RuntimeError, "case-insensitive unique index on email"):
                auth.verify_schema()


if __name__ == "__main__":
    unittest.main()
