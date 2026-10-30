import unittest

import numpy as np

from pokerdqn.dataset import (CONTEXT_COLUMNS, OUTCOME_COLUMNS, REPLAY_COLUMNS,
                              build_transitions, find_dataset_dir, load_tables, replay_all)


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_tables(find_dataset_dir())
        cls.replays = replay_all(cls.tables)
        # Few Monte Carlo samples: these tests check structure, not equity.
        cls.sets = build_transitions(cls.tables, cls.replays, equity_samples=10)

    def test_every_hand_replays_and_results_match(self):
        self.assertEqual(len(self.replays), 1747)
        self.assertTrue(all(r.ok for r in self.replays.values()),
                        [r.error for r in self.replays.values() if not r.ok][:3])
        self.assertTrue(all(r.payoffs_match for r in self.replays.values()))

    def test_simulator_context_matches_dataset_columns(self):
        total = sum(len(r.steps) for r in self.replays.values())
        self.assertEqual(total, len(self.tables.decisions))

        for column in REPLAY_COLUMNS:
            wrong = sum(r.mismatches.get(column, 0) for r in self.replays.values())
            self.assertLessEqual(wrong, 5, column)  # opp_stack_bb: 3 ambiguous rows

    def test_logged_actions_are_legal(self):
        for result in self.replays.values():
            for step in result.steps:
                self.assertTrue(step.mask[step.action])

    def test_inputs_never_include_outcomes_or_targets(self):
        for column in CONTEXT_COLUMNS:
            self.assertFalse(column.startswith(('outcome_', 'target_')), column)
            self.assertNotIn(column, OUTCOME_COLUMNS)

        # Rewriting every outcome and target column must leave the states
        # unchanged (targets only define the logged action, outcomes the reward).
        tampered = load_tables(find_dataset_dir())
        d = tampered.decisions

        for column in d.columns:
            if column.startswith('outcome_'):
                d[column] = 999.0

        d['target_action'] = 'fold'
        d['target_chips_added_bb'] = 0.0
        d['target_bet_size_pot_frac'] = 0.0
        sets = build_transitions(tampered, self.replays, equity_samples=10)

        for name in ('train', 'val', 'test'):
            np.testing.assert_array_equal(sets[name].states, self.sets[name].states)
            np.testing.assert_array_equal(sets[name].next_states, self.sets[name].next_states)
            self.assertTrue(np.all(sets[name].rewards[sets[name].dones] == 999.0))

    def test_transition_structure(self):
        d = self.tables.decisions
        train = self.sets['train']
        self.assertEqual(sum(len(s) for s in self.sets.values()), len(d))
        # Rewards are paid only at a player's last decision of the hand.
        self.assertTrue(np.all(train.rewards[~train.dones] == 0))
        np.testing.assert_allclose(train.rewards[train.dones], train.returns[train.dones])
        self.assertFalse(train.next_masks[train.dones].any())
        self.assertTrue(train.next_masks[~train.dones].any(axis=1).all())

        # The next state is the same player's next decision in the same hand.
        ids = list(train.decision_ids)
        position = {decision: i for i, decision in enumerate(ids)}
        rows = d.set_index('decision_id').loc[ids]

        for i in np.flatnonzero(~train.dones)[:300]:
            game, player = rows.iloc[i]['game_id'], rows.iloc[i]['player_id']
            later = rows[(rows['game_id'] == game) & (rows['player_id'] == player)
                         & (rows['decision_idx'] > rows.iloc[i]['decision_idx'])]
            j = position[later.index[0]]
            np.testing.assert_array_equal(train.next_states[i], train.states[j])

    def test_splits_are_disjoint_and_chronological(self):
        d = self.tables.decisions
        games = {s: set(d.loc[d['split'] == s, 'game_id']) for s in ('train', 'val', 'test')}
        self.assertFalse(games['train'] & games['val'])
        self.assertFalse(games['train'] & games['test'])
        self.assertFalse(games['val'] & games['test'])
        self.assertLess(max(games['train']), min(games['val']))
        self.assertLess(max(games['val']), min(games['test']))


if __name__ == '__main__':
    unittest.main()
