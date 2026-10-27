import random
import unittest

import numpy as np

from pokerdqn.actions import (ALL_IN, CALL, CHECK, FOLD, RAISE_HALF, RAISE_POT, chips_for,
                              dataset_action, legal_mask)
from pokerdqn.agents import CallingStation, RandomAgent, RuleBasedAgent, StyleAgent
from pokerdqn.env import EnvConfig, PokerEnv, play_hands
from pokerdqn.features import STATE_DIM


def random_table(rng):
    return [RandomAgent(random.Random(rng.random())) for _ in range(5)]


class Actions(unittest.TestCase):
    def test_masks(self):
        # Facing a bet with chips behind: fold, call and raises.
        mask = legal_mask(to_call=200, my_bet=100, pot=450, raise_range=(500, 10000))
        self.assertEqual(list(np.flatnonzero(mask)), [FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN])
        # Nothing owed: no fold.
        mask = legal_mask(0, 0, 300, (100, 10000))
        self.assertFalse(mask[FOLD])
        self.assertTrue(mask[CHECK])
        # Short stack: every size is the all-in.
        mask = legal_mask(0, 0, 300, (100, 120))
        self.assertEqual(list(np.flatnonzero(mask)), [CHECK, ALL_IN])
        # Covered by the bet: only fold or call.
        self.assertEqual(list(np.flatnonzero(legal_mask(500, 0, 800, None))), [FOLD, CALL])

    def test_sizes(self):
        # Preflop open from 1.5 bb: half pot raises to 2.25 bb, pot to 3.5 bb.
        self.assertEqual(chips_for(RAISE_HALF, 100, 0, 150, (200, 10000)), 225)
        self.assertEqual(chips_for(RAISE_POT, 100, 0, 150, (200, 10000)), 350)
        self.assertEqual(chips_for(ALL_IN, 100, 0, 150, (200, 10000)), 10000)
        # Never below the minimum raise.
        self.assertEqual(chips_for(RAISE_HALF, 0, 0, 100, (100, 10000)), 100)

    def test_dataset_mapping(self):
        self.assertEqual(dataset_action('raise', 2.5, 1.0, 1.5, 100), RAISE_HALF)  # 2.5x open
        self.assertEqual(dataset_action('raise', 3.5, 1.0, 1.5, 100), RAISE_POT)
        self.assertEqual(dataset_action('bet', 5.0, 0.0, 10.0, 100), RAISE_HALF)
        self.assertEqual(dataset_action('bet', 8.0, 0.0, 10.0, 100), RAISE_POT)
        self.assertEqual(dataset_action('all-in', 30, 10, 20, 30), ALL_IN)
        self.assertEqual(dataset_action('all-in', 8, 10, 20, 8), CALL)  # all-in call


class Environment(unittest.TestCase):
    def test_returns_are_chip_results_and_zero_sum(self):
        env = PokerEnv(random_table, EnvConfig(seed=1, stack_bb=[40, 100, 250]))
        hero = RandomAgent(random.Random(2))

        for _ in range(300):
            obs = env.reset()
            total = 0.0
            done = False

            while not done:
                self.assertTrue(obs.mask[obs.mask].all() and obs.mask.any())
                self.assertEqual(obs.vector().shape, (STATE_DIM,))
                obs, reward, done, info = env.step(hero.act(obs))
                total += reward

                if not done:
                    self.assertEqual(reward, 0.0)

            hand = env.hand
            self.assertEqual(sum(hand.payoffs), 0)
            self.assertAlmostEqual(total, env.results[-1].net_bb)
            self.assertAlmostEqual(total, hand.payoffs[env.order.index(0)] / 100)

    def test_observation_hides_other_hole_cards(self):
        env = PokerEnv(random_table, EnvConfig(seed=3))
        obs = env.reset()
        hand = env.hand
        hero = env.order.index(0)
        others = {c for i in range(hand.n) if i != hero for c in hand.hole[i]}
        self.assertFalse(others & (set(obs.hole) | set(obs.board)))
        self.assertTrue(set(obs.board) <= set(hand.board))
        # Unrevealed board cards stay hidden too.
        self.assertEqual(len(obs.board), {0: 0, 1: 3, 2: 4, 3: 5}[hand.street])

    def test_same_seed_same_deals_for_different_heroes(self):
        def holes(hero):
            env = PokerEnv(random_table, EnvConfig(seed=11, stack_bb=[50, 100, 200]))
            return [r.hero_hole for r in play_hands(env, hero, 200)]

        # Deals come from their own random stream, so they don't depend on
        # how the hero (or, in turn, the opponents) play.
        self.assertEqual(holes(CallingStation()), holes(RandomAgent(random.Random(0))))
        self.assertEqual(holes(RuleBasedAgent()), holes(CallingStation()))

    def test_button_rotates_and_every_position_is_played(self):
        env = PokerEnv(random_table, EnvConfig(seed=5))
        results = play_hands(env, CallingStation(), 60)
        self.assertEqual({r.position for r in results}, {'SB', 'BB', 'UTG', 'HJ', 'CO', 'BTN'})

    def test_opponent_stats_accumulate_within_session(self):
        env = PokerEnv(lambda rng: [CallingStation() for _ in range(5)],
                       EnvConfig(seed=2, session_hands=50))
        play_hands(env, CallingStation(), 40)
        seen = env.tracker.summary(1)
        self.assertEqual(seen['hands_seen'], 40)
        self.assertEqual(seen['pfr'], 0.0)  # calling stations never raise
        self.assertGreater(seen['vpip'], 0.6)

    def test_style_agent_only_picks_legal_actions(self):
        profiles = {'X': {'0|1|0|0': [50, 0, 50, 0, 0, 0]}}
        env = PokerEnv(lambda rng: [StyleAgent('X', profiles, random.Random(1)) for _ in range(5)],
                       EnvConfig(seed=4))
        play_hands(env, RandomAgent(random.Random(1)), 200)  # raises on illegal actions


if __name__ == '__main__':
    unittest.main()
