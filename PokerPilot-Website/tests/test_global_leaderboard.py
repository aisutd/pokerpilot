import tempfile
import unittest
import sqlite3
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import app, matches, match_chat_rooms
from leaderboard_store import LeaderboardStore


class GlobalLeaderboardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = LeaderboardStore(Path(self.temp.name) / 'scores.sqlite3')
        self.previous_store = app.config['LEADERBOARD_STORE']
        app.config['LEADERBOARD_STORE'] = self.database
        app.config['TESTING'] = True
        self.old_matches = set(matches)

    def tearDown(self):
        app.config['LEADERBOARD_STORE'] = self.previous_store
        for key in set(matches) - self.old_matches:
            matches.pop(key, None)
            match_chat_rooms.pop(key, None)
        self.temp.cleanup()

    def test_rankings_aggregate_losses_and_wins_across_games(self):
        first, second = self.database.create_player(), self.database.create_player()
        self.database.rename_player(first['id'], 'Alice')
        self.database.rename_player(second['id'], 'Bob')
        self.database.save_rounds(first['id'], 'game-1', [{'round_number': 1, 'net_cents': 1500}])
        self.database.save_rounds(first['id'], 'game-2', [{'round_number': 1, 'net_cents': -600}])
        self.database.save_rounds(second['id'], 'game-3', [{'round_number': 1, 'net_cents': 1000}])
        board = self.database.standings(first['id'])['standings']
        self.assertEqual([p['name'] for p in board], ['Bob', 'Alice'])
        self.assertEqual([p['net_winnings_cents'] for p in board], [1000, 900])
        self.assertEqual(board[1]['rounds_participated'], 2)
        self.assertTrue(board[1]['is_you'])

    def test_parallel_retries_never_duplicate_credit(self):
        player = self.database.create_player()
        def save(_):
            self.database.save_rounds(player['id'], 'game', [{'round_number': 1, 'net_cents': 123}])
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(save, range(12)))
        row = self.database.standings()['standings'][0]
        self.assertEqual(row['net_winnings_cents'], 123)
        self.assertEqual(row['rounds_participated'], 1)

    def test_store_restart_preserves_results_and_signing_key(self):
        player = self.database.create_player()
        self.database.save_rounds(player['id'], 'game', [{'round_number': 1, 'net_cents': -200}])
        secret = self.database.session_secret()
        restarted = LeaderboardStore(self.database.path)
        self.assertEqual(secret, restarted.session_secret())
        self.assertEqual(restarted.standings(), self.database.standings())
        self.assertEqual(restarted.profile(player['id']), player)

    def test_names_ties_and_identity(self):
        first, second = app.test_client(), app.test_client()
        one = first.post('/api/profile', json={'name': 'Alice'}).json
        two = second.post('/api/profile', json={'name': 'Bob'}).json
        self.assertNotEqual(one['id'], two['id'])
        self.assertEqual(first.get('/api/profile').json['id'], one['id'])
        self.assertEqual(second.post('/api/profile', json={'name': 'alice'}).status_code, 400)
        self.assertEqual(first.post('/api/profile', json={'name': 'x'}).status_code, 400)
        ranks = first.get('/api/leaderboard').json['standings']
        self.assertEqual([p['rank'] for p in ranks], [1, 1])
        self.assertEqual(sum(p['is_you'] for p in ranks), 1)

    def test_live_engine_round_saved_once_and_shared_between_users(self):
        first, second = app.test_client(), app.test_client()
        profile = first.post('/api/profile', json={'name': 'Alice'}).json
        second.post('/api/profile', json={'name': 'Bob'})
        view = first.post('/api/new', json={'difficulty': 'easy'}).json['view']
        # The client cannot supply a score; only authoritative engine payoffs count.
        for _ in range(100):
            if view['hand_over']:
                break
            result = first.post('/api/action', json={'action': 'check_call', 'net_cents': 99999999})
            self.assertEqual(result.status_code, 200)
            view = result.json['view']
        self.assertTrue(view['hand_over'])
        for _ in range(3):
            first.get('/api/state')
        standings = second.get('/api/leaderboard').json['standings']
        self.assertEqual(len(standings), 2)  # no AI players
        row = next(row for row in standings if row['player_id'] == profile['id'])
        self.assertEqual(row['net_winnings_cents'], view['result']['human_net'])
        self.assertEqual(row['rounds_participated'], 1)
        expected = row['net_winnings_cents']
        first.post('/api/quit', json={})
        first.post('/api/new', json={'difficulty': 'easy'})
        row = next(row for row in first.get('/api/leaderboard').json['standings'] if row['is_you'])
        self.assertEqual(row['net_winnings_cents'], expected)
        self.assertEqual(row['rounds_participated'], 1)

    def test_unfinished_round_is_not_saved(self):
        client = app.test_client()
        client.get('/api/profile')
        client.post('/api/new', json={'difficulty': 'easy'})
        client.post('/api/quit', json={})
        row = client.get('/api/leaderboard').json['standings'][0]
        self.assertEqual(row['rounds_participated'], 0)
        self.assertEqual(row['net_winnings_cents'], 0)

    def test_storage_failure_can_retry_without_losing_or_double_counting_round(self):
        client = app.test_client()
        client.get('/api/profile')
        with client.session_transaction() as session:
            player_id = session['player_id']
        view = client.post('/api/new', json={'difficulty': 'easy'}).json['view']
        with client.session_transaction() as session:
            match = matches[session['match_id']][0]
        # Simulate a DB outage during a round completion and the next resync.
        with patch.object(self.database, 'save_rounds', side_effect=sqlite3.OperationalError('test outage')):
            for _ in range(100):
                if match.hand_over:
                    break
                response = client.post('/api/action', json={'action': 'check_call'})
                self.assertEqual(response.status_code, 503)
            self.assertTrue(match.hand_over)
        expected = match.view()['result']['human_net']
        self.assertEqual(client.get('/api/state').status_code, 200)
        client.get('/api/state')
        row = next(r for r in self.database.standings()['standings'] if r['player_id'] == player_id)
        self.assertEqual(row['net_winnings_cents'], expected)
        self.assertEqual(row['rounds_participated'], 1)


if __name__ == '__main__':
    unittest.main()
