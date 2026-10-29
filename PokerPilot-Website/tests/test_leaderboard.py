import unittest

from app import app, matches, match_chat_rooms
from engine.game import GameError, Table


class LeaderboardIntegrationTest(unittest.TestCase):
    def settle(self, table):
        for _ in range(100):
            if table.hand_over:
                return
            table.human_action('check_call')
        self.fail('Hand did not settle')

    def test_balances_publish_only_after_settlement(self):
        table = Table('easy', seed=10)
        initial = table.leaderboard.snapshot()
        self.assertEqual(len(initial['standings']), 6)
        self.assertTrue(all(p['current_chips'] == 200 for p in initial['standings']))
        self.assertTrue(all(p['net_profit'] == 0 for p in initial['standings']))
        self.settle(table)
        board = table.view()['leaderboard']
        profits = []
        for row in board['standings']:
            self.assertEqual(row['current_chips'], table.seats[int(row['player_id'])].stack / 100)
            self.assertEqual(row['hands_played'], 1)
            self.assertEqual(row['net_profit'], round(row['current_chips'] - 200, 2))
            profits.append(row['net_profit'])
        self.assertEqual(profits, sorted(profits, reverse=True))
        self.assertAlmostEqual(sum(profits), 0)
        if not table.game_over:
            table.next_hand()
            self.assertEqual(table.view()['leaderboard'], board)

    def test_eliminations_and_natural_game_finish(self):
        table = Table('easy', seed=11)
        for _ in range(150):
            while not table.hand_over:
                legal = table.legal_actions()
                if legal['raise']:
                    table.human_action('raise', legal['raise']['max'])
                else:
                    table.human_action('check_call')
            board = table.leaderboard.snapshot()
            self.assertEqual(len(board['standings']), 6)
            for row in board['standings']:
                if row['current_chips'] == 0:
                    self.assertEqual(row['status'], 'finished')
            if table.game_over:
                self.assertEqual(board['status'], 'finished')
                self.assertTrue(all(row['status'] == 'finished' for row in board['standings']))
                return
            table.next_hand()
        self.fail('Game did not finish')

    def test_quitting_freezes_last_settled_results(self):
        table = Table('easy', seed=10)
        self.settle(table)
        before = table.leaderboard.standings()
        if not table.game_over:
            table.next_hand()
        table.finish_session()
        table.finish_session()  # idempotent
        after = table.leaderboard.standings()
        self.assertEqual([r['current_chips'] for r in before], [r['current_chips'] for r in after])
        self.assertEqual(table.leaderboard.status, 'finished')
        with self.assertRaises(GameError):
            table.human_action('fold')
        with self.assertRaises(GameError):
            table.next_hand()

    def test_api_quit_resume_and_new_game_reset(self):
        app.config['TESTING'] = True
        client = app.test_client()
        response = client.post('/api/new', json={'difficulty': 'easy'})
        self.assertEqual(response.status_code, 200)
        board = response.json['view']['leaderboard']
        with client.session_transaction() as session:
            old_id = session['match_id']
        response = client.post('/api/quit', json={})
        self.assertEqual(response.json['leaderboard']['status'], 'finished')
        self.assertTrue(client.get('/api/state').json['view']['session_finished'])
        self.assertEqual(client.post('/api/action', json={'action': 'fold'}).status_code, 400)
        new_board = client.post('/api/new', json={'difficulty': 'easy'}).json['view']['leaderboard']
        self.assertNotEqual(board['game_id'], new_board['game_id'])
        self.assertTrue(all(r['net_profit'] == 0 for r in new_board['standings']))
        self.assertEqual(matches[old_id][0].leaderboard.status, 'finished')
        with client.session_transaction() as session:
            new_id = session['match_id']
        for key in (old_id, new_id):
            matches.pop(key, None)
            match_chat_rooms.pop(key, None)


if __name__ == '__main__':
    unittest.main()
