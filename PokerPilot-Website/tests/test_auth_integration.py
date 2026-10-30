import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PokerPilot-Website"))
import app as website  # noqa: E402


class SinglePlayerAuthIntegrationTests(unittest.TestCase):
    def setUp(self):
        website.app.config.update(TESTING=True)
        self.client = website.app.test_client()

    def test_lobby_redirects_to_login_and_game_api_requires_auth(self):
        self.assertEqual(self.client.get("/").status_code, 302)
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json["detail"], "Please sign in to continue.")

    def test_state_change_requires_csrf_even_after_login(self):
        user = {"id": 1, "username": "test", "email": "test@example.test"}
        self.client.set_cookie("pokerpilot_session", "test-session")
        with patch("backend.auth_routes.auth.session_user", return_value=user):
            response = self.client.post("/api/new", json={"difficulty": "easy", "model": "rules"})
        self.assertEqual(response.status_code, 403)

    def test_svg_deck_is_served_by_game_app(self):
        response = self.client.get("/cards/deck.svg")
        self.assertEqual(response.status_code, 200)
        self.assertIn("image/svg+xml", response.content_type)
        self.assertIn(b"card-Kc", response.data)


if __name__ == "__main__":
    unittest.main()
