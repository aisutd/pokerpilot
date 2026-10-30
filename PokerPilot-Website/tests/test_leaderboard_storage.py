import unittest
from unittest.mock import Mock, patch

import app as website
from engine.game import Table


class LeaderboardStorageTests(unittest.TestCase):
    def setUp(self):
        self.previous = website.app.config['LEADERBOARD_STORE']
        self.store = Mock()
        website.app.config['LEADERBOARD_STORE'] = self.store

    def tearDown(self):
        website.app.config['LEADERBOARD_STORE'] = self.previous

    def test_completed_hand_is_saved_and_retries_keep_pending_results(self):
        table = Table('easy', seed=12)
        table.player_id = 'account-1'
        table.leaderboard_game_id = 'game-1'
        self.assertEqual(table.pending_round_results, [])
        while not table.hand_over:
            table.human_action('check_call')
        expected = [{'round_number': 1, 'net_cents': int(table.result['human_net'])}]
        self.assertEqual(table.pending_round_results, expected)
        self.store.save_rounds.side_effect = RuntimeError('offline')
        with self.assertRaises(website.LeaderboardUnavailable):
            website.persist_results(table)
        self.assertEqual(table.pending_round_results, expected)
        self.store.save_rounds.side_effect = None
        website.persist_results(table)
        self.assertEqual(table.pending_round_results, [])
        count = self.store.save_rounds.call_count
        website.persist_results(table)
        self.assertEqual(self.store.save_rounds.call_count, count)

    def test_leaderboard_reads_saved_standings_for_authenticated_account(self):
        self.store.standings.return_value = {'standings': [{'rank': 1}]}
        with patch('backend.auth_routes.auth.session_user', return_value={'id': 7}):
            response = website.app.test_client().get('/api/leaderboard')
        self.assertEqual(response.status_code, 200)
        self.store.standings.assert_called_once_with('7')
        self.assertEqual(response.json['standings'][0]['rank'], 1)
