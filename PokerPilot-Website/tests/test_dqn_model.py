import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from app import app, match_chat_rooms, matches
from dqn_model_surya.env import (
    ACTION_NAMES,
    ACTION_COUNT,
    STATE_SIZE,
    PokerDQNEnv,
    STYLE_PERSONALITIES,
    action_to_engine_action,
    legal_action_mask,
    load_style_distribution,
)
from dqn_model_surya.evaluate import evaluate
from dqn_model_surya.model import QNetwork, masked_q_values
from dqn_model_surya.replay_buffer import ReplayBuffer
from dqn_model_surya.train import choose_action, epsilon_at_step
from engine.game import HUMAN_SEAT, Table


class DQNEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.style_distribution = {'Balanced Reg': 1, 'Maniac': 1}
        self.env = PokerDQNEnv(seed=123, style_distribution=self.style_distribution)

    def test_observation_and_action_mask_have_fixed_shapes(self) -> None:
        state, action_mask, _ = self.env.reset()

        self.assertEqual(state.shape, (STATE_SIZE,))
        self.assertEqual(action_mask.shape, (ACTION_COUNT,))
        self.assertEqual(state.dtype, np.float32)
        self.assertTrue(np.isfinite(state).all())
        self.assertLessEqual(np.abs(state).max(), 1.0)
        self.assertTrue(action_mask[1])

    def test_dataset_styles_are_installed_before_the_first_decision(self) -> None:
        self.env.reset()

        self.assertTrue(all(
            seat.bot.personality in STYLE_PERSONALITIES.values()
            for seat in self.env.table.seats[1:]
        ))

    def test_step_returns_hand_result_in_big_blinds(self) -> None:
        self.env.reset()
        done = False
        transitions = 0

        while not done and transitions < 100:
            state, mask, reward, done, info = self.env.step(1)
            transitions += 1
            if not done:
                self.assertTrue(mask[1])
                self.assertTrue(np.isfinite(state).all())
            else:
                self.assertEqual(reward, info['net_bb'])

        self.assertTrue(done, 'A single poker hand should eventually settle.')

    def test_illegal_action_is_rejected_before_engine_call(self) -> None:
        fake_table = SimpleNamespace(
            legal_actions=lambda: {'fold': False, 'raise': None},
        )
        mask = legal_action_mask(fake_table)

        self.assertFalse(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2:].any())
        with self.assertRaisesRegex(ValueError, 'Illegal DQN action'):
            action_to_engine_action(fake_table, 0)

    def test_legal_action_selection_masks_high_valued_illegal_action(self) -> None:
        q_values = torch.tensor([[1.0, 2.0, 1000.0]])
        mask = torch.tensor([[True, True, False]])

        chosen = masked_q_values(q_values, mask).argmax(dim=1).item()

        self.assertEqual(chosen, 1)

    def test_epsilon_greedy_never_samples_an_illegal_action(self) -> None:
        network = QNetwork(STATE_SIZE, ACTION_COUNT)
        mask = np.asarray([False, True, False, False, False, False], dtype=np.bool_)

        for seed in range(10):
            action = choose_action(
                network,
                np.zeros(STATE_SIZE, dtype=np.float32),
                mask,
                epsilon=1.0,
                rng=random.Random(seed),
            )
            self.assertEqual(action, 1)

    def test_training_and_test_style_profiles_are_separated(self) -> None:
        training_styles = load_style_distribution('train')
        test_styles = load_style_distribution('test')

        self.assertTrue(training_styles)
        self.assertTrue(test_styles)
        self.assertEqual(
            set(training_styles),
            {'Rock', 'Tight-Passive', 'Tight-Aggressive', 'Balanced Reg',
             'Loose-Passive', 'Loose-Aggressive', 'Maniac'},
        )

    def test_replay_buffer_copies_masks_and_states(self) -> None:
        buffer = ReplayBuffer(capacity=2, seed=4)
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        mask = np.ones(ACTION_COUNT, dtype=np.bool_)
        buffer.add(state, 1, 0.0, state, False, mask)
        state[0] = 1.0
        mask[0] = False

        transition = buffer.sample(1)[0]

        self.assertEqual(transition.state[0], 0.0)
        self.assertTrue(transition.next_action_mask[0])

    def test_epsilon_decay_is_bounded_and_monotonic(self) -> None:
        values = [epsilon_at_step(step, 1.0, 0.05, 100) for step in (0, 50, 100, 200)]

        np.testing.assert_allclose(values, [1.0, 0.525, 0.05, 0.05])

    def test_evaluation_refuses_fewer_than_ten_thousand_hands(self) -> None:
        with self.assertRaisesRegex(ValueError, 'at least 10,000 hands'):
            evaluate(Path('not-needed.pt'), 9_999, 2507)


class WebsiteAdvisorTest(unittest.TestCase):
    def test_advisor_endpoint_returns_a_legal_recommendation(self) -> None:
        table = Table('medium', seed=81)
        while table.hand_over:
            table.next_hand()
        self.assertEqual(table.actor_seat, HUMAN_SEAT)

        match_id = f'dqn-test-{id(self)}'
        client = app.test_client()
        with client.session_transaction() as flask_session:
            flask_session['match_id'] = match_id
        matches[match_id] = (table, time.time())

        try:
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / 'advisor.pt'
                torch.save({
                    'model_state_dict': QNetwork(STATE_SIZE, ACTION_COUNT).state_dict(),
                    'state_size': STATE_SIZE,
                    'action_count': ACTION_COUNT,
                    'action_names': list(ACTION_NAMES),
                }, checkpoint)
                with patch('app.ADVISOR_CHECKPOINT', checkpoint):
                    response = client.post('/api/advice', json={})

            self.assertEqual(response.status_code, 200)
            recommendation = response.get_json()
            self.assertIn(recommendation['action'], {'fold', 'check_call', 'raise'})
            self.assertIsInstance(recommendation['q_value_bb'], float)
            if recommendation['action'] == 'raise':
                self.assertTrue(table.state.can_complete_bet_or_raise_to(
                    recommendation['amount'],
                ))
        finally:
            matches.pop(match_id, None)
            match_chat_rooms.pop(match_id, None)


if __name__ == '__main__':
    unittest.main()
