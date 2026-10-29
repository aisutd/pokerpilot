import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pokerdqn.dqn import DQNConfig, DQNLearner, EpsilonSchedule, ReplayBuffer, masked


class ReplayBufferTest(unittest.TestCase):
    def test_circular_and_terminal_handling(self):
        buffer = ReplayBuffer(capacity=3, state_dim=2, num_actions=3)
        mask = np.array([True, True, False])

        for i in range(5):
            buffer.add(np.full(2, i), i % 3, float(i), np.full(2, i + 1), i == 4, mask, mask)

        self.assertEqual(len(buffer), 3)
        self.assertEqual(sorted(buffer.rewards.tolist()), [2.0, 3.0, 4.0])
        last = int(np.flatnonzero(buffer.dones)[0])
        self.assertFalse(buffer.next_masks[last].any())
        self.assertTrue(np.all(buffer.next_states[last] == 0))


class LearnerTest(unittest.TestCase):
    def test_masked_max_ignores_illegal_actions(self):
        q = torch.tensor([[1.0, 50.0, 2.0]])
        mask = torch.tensor([[True, False, True]])
        self.assertEqual(int(masked(q, mask).argmax()), 2)

    def test_learns_a_two_step_problem(self):
        """State A: action 0 pays +1 now; action 1 moves to state B, where
        action 0 pays +3 and action 1 (illegal) would pay +100. Optimal Q:
        Q(A,0)=1, Q(A,1)=3, Q(B,0)=3. Bootstrapping must respect the mask."""

        A, B = np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)
        both, first = np.array([True, True]), np.array([True, False])
        buffer = ReplayBuffer(1000, 2, num_actions=2, seed=0)

        for _ in range(200):
            buffer.add(A, 0, 1.0, None, True, both, None)
            buffer.add(A, 1, 0.0, B, False, both, first)
            buffer.add(B, 0, 3.0, None, True, first, None)

        learner = DQNLearner(DQNConfig(state_dim=2, num_actions=2, hidden=(32,), lr=3e-3,
                                       reward_scale=1.0, tau=0.05, batch_size=64), seed=0)

        for _ in range(1500):
            learner.update(buffer.sample(64))

        q = learner.q_values(np.stack([A, B]))
        self.assertAlmostEqual(q[0, 0], 1.0, delta=0.15)
        self.assertAlmostEqual(q[0, 1], 3.0, delta=0.3)
        self.assertAlmostEqual(q[1, 0], 3.0, delta=0.15)
        self.assertEqual(learner.greedy(A, both), 1)

    def test_cql_lowers_unseen_actions(self):
        state = np.array([1.0], np.float32)
        mask = np.array([True, True, True])
        buffer = ReplayBuffer(100, 1, num_actions=3)

        for _ in range(100):
            buffer.add(state, 0, 1.0, None, True, mask, None)

        learner = DQNLearner(DQNConfig(state_dim=1, num_actions=3, hidden=(16,), lr=3e-3,
                                       reward_scale=1.0, cql_alpha=1.0), seed=0)

        for _ in range(800):
            info = learner.update(buffer.sample(32))

        q = learner.q_values(state)[0]
        self.assertGreater(q[0], q[1] + 0.5)
        self.assertGreater(q[0], q[2] + 0.5)
        self.assertIn('cql_gap', info)

    def test_save_and_load(self):
        learner = DQNLearner(DQNConfig(state_dim=4, hidden=(8,)), seed=1)
        x = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            learner.save(path, {'note': 'test'})
            loaded = DQNLearner.load(path)

        np.testing.assert_allclose(loaded.q_values(x), learner.q_values(x), rtol=1e-6)
        self.assertEqual(loaded.extra['note'], 'test')

    def test_epsilon_schedule(self):
        schedule = EpsilonSchedule(1.0, 0.1, 100)
        self.assertEqual(schedule(0), 1.0)
        self.assertAlmostEqual(schedule(50), 0.55)
        self.assertAlmostEqual(schedule(1000), 0.1)


if __name__ == '__main__':
    unittest.main()
