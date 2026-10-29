"""Tests for picking the opponents' AI model (engine/opponent_models.py).

Run from PokerPilot-Website:  python -m unittest tests.test_opponent_models
"""

import math
import unittest
from pathlib import Path

from pokerkit import Card

from engine.game import HUMAN_SEAT, GameError, Table
from engine.opponent_models import (
    availability,
    hand_features,
    list_models,
    position_name,
    table_state,
)

DATASET = Path(__file__).resolve().parents[2] / 'Dataset' / 'poker_decisions_test.csv'
XGBOOST_READY = availability('xgboost') is None


def play(table: Table, hands: int) -> int:
    """Human checks or folds; returns how many hands finished."""

    played = 0

    while played < hands and not table.game_over:
        if table.hand_over:
            table.next_hand()
            played += 1
        elif table.actor_seat == HUMAN_SEAT:
            table.human_action('check_call' if table.legal_actions()['check'] else 'fold')

    return played


class ModelRegistryTests(unittest.TestCase):
    def test_lists_every_model(self):
        keys = [m['key'] for m in list_models()]
        self.assertEqual(keys, ['rules', 'xgboost', 'dqn'])
        rules = list_models()[0]
        self.assertTrue(rules['available'])
        self.assertIsNone(rules['reason'])

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(GameError):
            Table('easy', seed=1, model='nope')

    def test_rules_is_the_default(self):
        table = Table('easy', seed=1)
        self.assertEqual(table.model, 'rules')
        self.assertEqual(table.view()['model'], 'rules')


class FeatureTests(unittest.TestCase):
    def test_positions_match_dataset_names(self):
        self.assertEqual([position_name([10, 20], s) for s in (10, 20)], ['BB', 'BTN/SB'])
        six = [1, 2, 3, 4, 5, 0]
        self.assertEqual(
            [position_name(six, s) for s in six],
            ['SB', 'BB', 'UTG', 'HJ', 'CO', 'BTN'],
        )
        self.assertEqual([position_name([4, 5, 0], s) for s in (4, 5, 0)], ['SB', 'BB', 'BTN'])

    @unittest.skipUnless(DATASET.is_file(), 'dataset not found')
    def test_card_features_match_the_dataset(self):
        import pandas as pd

        rows = pd.read_csv(DATASET).sample(300, random_state=0)
        columns = [
            'card_high_rank', 'card_low_rank', 'is_suited', 'chen_score',
            'preflop_percentile', 'made_hand', 'made_hand_rank', 'made_hand_strength',
            'pair_type', 'flush_draw', 'board_paired', 'board_suit_texture',
            'board_connectedness', 'board_high_rank',
        ]

        for _, row in rows.iterrows():
            hole = list(Card.parse(row.hole_cards.replace(' ', '')))
            board = list(Card.parse(row.board.replace(' ', ''))) if isinstance(row.board, str) else []
            features = hand_features(hole, board)

            for column in columns:
                expected = row[column]
                actual = features.get(column, math.nan)

                if isinstance(expected, str):
                    self.assertEqual(actual, expected, (column, row.hole_cards, row.board))
                elif not pd.isna(expected):
                    self.assertAlmostEqual(actual, expected, places=3,
                                           msg=(column, row.hole_cards, row.board))

    def test_table_state_uses_only_public_info(self):
        # A new table runs the bots until it is the human's turn.
        table = Table('medium', seed=9)
        self.assertEqual(table.actor_seat, HUMAN_SEAT)
        seat = next(s for s in table.order if s != HUMAN_SEAT)
        human_cards = {str(c) for c in table.hole[HUMAN_SEAT]}
        state = table_state(table, seat, equity_samples=20)

        self.assertIn(state['position'], {'SB', 'BB', 'UTG', 'HJ', 'CO', 'BTN', 'BTN/SB'})
        self.assertFalse(human_cards & set(state['hole_cards'].split()))
        self.assertGreater(state['pot_bb'], 0)


@unittest.skipUnless(XGBOOST_READY, f'XGBoost model unavailable: {availability("xgboost")}')
class XGBoostOpponentTests(unittest.TestCase):
    def test_games_play_out_on_every_difficulty(self):
        for difficulty in ('easy', 'medium', 'hard'):
            table = Table(difficulty, seed=21, model='xgboost')
            self.assertGreater(play(table, 6), 0)
            chips = sum(seat.stack for seat in table.seats)
            self.assertEqual(chips, 6 * 20_000)

    def test_bots_bet_and_fold_sometimes(self):
        table = Table('medium', seed=3, model='xgboost')
        play(table, 25)
        bots = [seat.stats for seat in table.seats[1:]]
        self.assertGreater(sum(s.aggressive for s in bots), 0)
        self.assertLess(sum(s.vpip for s in bots), sum(s.hands for s in bots))


class ApiTests(unittest.TestCase):
    def setUp(self):
        from app import app
        from backend.auth_routes import auth as auth_storage
        from unittest.mock import patch
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.client.set_cookie('pokerpilot_session', 'test-session')
        self.client.set_cookie('pokerpilot_csrf', 'test-csrf')
        self.auth_patch = patch.object(auth_storage, 'session_user', return_value={
            'id': 1, 'username': 'test', 'email': 'test@example.test',
        })
        self.auth_patch.start()

    def tearDown(self):
        self.auth_patch.stop()

    def test_models_endpoint(self):
        data = self.client.get('/api/models').get_json()
        self.assertEqual(data['models'][0]['key'], 'rules')

    def test_new_game_with_model(self):
        response = self.client.post('/api/new', json={'difficulty': 'easy', 'model': 'rules'},
                                    headers={'X-CSRF-Token': 'test-csrf'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['view']['model'], 'rules')

    def test_new_game_rejects_bad_model(self):
        response = self.client.post('/api/new', json={'difficulty': 'easy', 'model': 'gpt'},
                                    headers={'X-CSRF-Token': 'test-csrf'})
        self.assertEqual(response.status_code, 400)

    def test_unavailable_model_gives_clear_error(self):
        unavailable = [m for m in list_models() if not m['available']]

        if not unavailable:
            self.skipTest('every model is available here')

        response = self.client.post('/api/new', json={'difficulty': 'easy', 'model': unavailable[0]['key']},
                                    headers={'X-CSRF-Token': 'test-csrf'})
        self.assertEqual(response.status_code, 400)
        self.assertIn('not available', response.get_json()['error'])


if __name__ == '__main__':
    unittest.main()
