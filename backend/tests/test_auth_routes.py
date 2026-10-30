import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from flask import Flask

from backend.auth_routes import register_auth
from backend.auth_routes import _attempts, _attempts_lock


class DuplicateError(Exception):
    sqlstate = "23505"


class AuthRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.testing = True
        self.app.secret_key = "test-only-secret"
        register_auth(self.app)
        self.client = self.app.test_client()
        with _attempts_lock:
            _attempts.clear()
        self.user = {"id": 42, "username": "Siri", "email": "siri@example.test", "password_hash": "stored-hash"}

    def csrf(self):
        response = self.client.get("/api/auth/csrf")
        self.assertEqual(response.status_code, 200)
        return response.get_json()["csrf_token"]

    def post(self, path, body):
        token = self.client.get_cookie("pokerpilot_csrf").value
        return self.client.post(path, json=body, headers={"X-CSRF-Token": token})

    def test_signup_sets_opaque_session_cookie_and_session_is_shared_by_me(self):
        self.csrf()
        with patch("backend.auth_routes.auth.create_user", return_value=self.user), \
             patch("backend.auth_routes.auth.new_session", return_value=("raw-session-token", datetime.now(timezone.utc) + timedelta(days=7))), \
             patch("backend.auth_routes.auth.session_user", return_value=self.user):
            response = self.post("/api/auth/signup", {
                "username": "Siri", "email": "siri@example.test", "password": "a-long-password-123"
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {"user": {"id": 42, "username": "Siri", "email": "siri@example.test"}})
            cookie = next(value for value in response.headers.getlist("Set-Cookie") if value.startswith("pokerpilot_session="))
            self.assertIn("HttpOnly", cookie)
            self.assertIn("SameSite=Lax", cookie)
            self.assertEqual(self.client.get("/api/auth/me").status_code, 200)
            second_browser = self.app.test_client()
            second_browser.set_cookie("pokerpilot_session", "raw-session-token")
            self.assertEqual(second_browser.get("/api/auth/me").status_code, 200)

    def test_signup_rejects_missing_csrf_and_duplicate_account(self):
        payload = {"username": "Siri", "email": "siri@example.test", "password": "a-long-password-123"}
        response = self.client.post("/api/auth/signup", json=payload, headers={"X-CSRF-Token": "bad"})
        self.assertEqual(response.status_code, 403)
        self.csrf()
        with patch("backend.auth_routes.auth.create_user", side_effect=DuplicateError()), \
             patch("backend.auth_routes.auth.new_session"):
            duplicate = self.post("/api/auth/signup", payload)
        self.assertEqual(duplicate.status_code, 409)

    def test_login_failure_throttle_and_logout_revoke(self):
        self.csrf()
        with patch("backend.auth_routes.auth.find_user", return_value=self.user), \
             patch("backend.auth_routes.auth.verify_password", return_value=False):
            for _ in range(8):
                self.assertEqual(self.post("/api/auth/login", {"identifier": "Siri", "password": "wrong"}).status_code, 401)
            self.assertEqual(self.post("/api/auth/login", {"identifier": "Siri", "password": "wrong"}).status_code, 429)

        # A successful login returns the same public user and logout revokes the token.
        with _attempts_lock:
            _attempts.clear()
        with patch("backend.auth_routes.auth.find_user", return_value=self.user), \
             patch("backend.auth_routes.auth.verify_password", return_value=True), \
             patch("backend.auth_routes.auth.new_session", return_value=("session", datetime.now(timezone.utc) + timedelta(days=7))), \
             patch("backend.auth_routes.auth.session_user", return_value=self.user), \
             patch("backend.auth_routes.auth.revoke_session") as revoke:
            login = self.post("/api/auth/login", {"identifier": "Siri", "password": "right"})
            self.assertEqual(login.status_code, 200)
            logout = self.post("/api/auth/logout", {})
            self.assertEqual(logout.status_code, 200)
            revoke.assert_called_once_with("session")

    def test_auth_page_and_storage_health_report_missing_database_as_unavailable(self):
        self.assertEqual(self.client.get("/auth").status_code, 200)
        self.assertIn("Create an account", self.client.get("/auth").text)
        with patch("backend.auth_routes.auth.verify_schema", side_effect=RuntimeError("missing URL")):
            response = self.client.get("/api/auth/health")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
