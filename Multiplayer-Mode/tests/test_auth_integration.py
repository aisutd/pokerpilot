import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as multiplayer_app  # noqa: E402
from engine.room import Room  # noqa: E402
from app import app, leaderboard_store, persist_room_results, player_belongs_to_user  # noqa: E402


class MultiplayerAuthIntegrationTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_room_creation_requires_authentication(self):
        response = self.client.post("/api/rooms", json={"name": "Guest"})
        self.assertEqual(response.status_code, 401)

    def test_leaderboard_results_clear_only_after_successful_save(self):
        room = Room("RESULTS", timers=False)
        room.pending_round_results = [
            {"account_id": "account-a", "round_number": 2, "net_cents": 400},
            {"account_id": "account-b", "round_number": 2, "net_cents": -400},
            {"account_id": "account-a", "round_number": 2, "net_cents": 100},
        ]

        with patch.object(leaderboard_store, "save_rounds") as save_rounds:
            self.assertTrue(persist_room_results(room, required=True))
            self.assertEqual(save_rounds.call_count, 2)
            self.assertTrue(all(call.args[1] == room.leaderboard_game_id for call in save_rounds.call_args_list))
            account_a_results = next(
                call.args[2] for call in save_rounds.call_args_list if call.args[0] == "account-a"
            )
            self.assertEqual(account_a_results, [{"round_number": 2, "net_cents": 500}])
            self.assertEqual(room.pending_round_results, [])

        room.pending_round_results = [
            {"account_id": "account-a", "round_number": 3, "net_cents": 200},
        ]
        with patch.object(leaderboard_store, "save_rounds", side_effect=RuntimeError("database down")):
            self.assertFalse(persist_room_results(room))
        self.assertEqual(len(room.pending_round_results), 1)

    def test_leaderboard_requires_authentication(self):
        response = self.client.get("/api/leaderboard")
        self.assertEqual(response.status_code, 401)

    def test_authenticated_leaderboard_endpoint_returns_standings(self):
        user = {"id": "account-a", "username": "Ann", "email": "ann@example.test"}
        standings = {
            "standings": [{
                "player_id": "account-a",
                "name": "Ann",
                "net_winnings_cents": 250,
                "rounds_participated": 3,
                "rank": 1,
                "is_you": True,
            }],
            "ranking": "cumulative_net_winnings",
            "currency": "USD",
        }

        with patch("backend.auth_routes.auth.session_user", return_value=user), \
             patch.object(leaderboard_store, "ensure_player") as ensure_player, \
             patch.object(leaderboard_store, "standings", return_value=standings), \
             patch.object(multiplayer_app, "persist_all_room_results"):
            response = self.client.get("/api/leaderboard")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, standings)
        ensure_player.assert_called_once_with("account-a", "Ann")

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
