import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.room import Room  # noqa: E402
from app import app, player_belongs_to_user  # noqa: E402


class MultiplayerAuthIntegrationTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_room_creation_requires_authentication(self):
        response = self.client.post("/api/rooms", json={"name": "Guest"})
        self.assertEqual(response.status_code, 401)

    def test_websocket_membership_is_bound_to_account(self):
        room = Room("TEST", timers=False)
        token = room.join("Ann", account_id="account-a")
        player = room.players[token]
        self.assertTrue(player_belongs_to_user(player, {"id": "account-a"}))
        self.assertFalse(player_belongs_to_user(player, {"id": "account-b"}))

    def test_online_app_serves_shared_auth_page_and_deck(self):
        self.assertIn("Create an account", self.client.get("/auth").text)
        self.assertIn(b"card-As", self.client.get("/cards/deck.svg").data)


if __name__ == "__main__":
    unittest.main()
