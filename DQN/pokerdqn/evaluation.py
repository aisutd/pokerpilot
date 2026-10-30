"""Opponent lineups, hero baselines and head-to-head evaluation.

Every hero is evaluated in a fresh ``PokerEnv`` with the same seed, so all
heroes get the same cards, stacks and opponent lineups hand for hand
(common random numbers). That allows a *paired* comparison: the per-hand
difference between two heroes has far less variance than either result.
Opponents' own random choices still diverge once the heroes act differently.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from .actions import ACTION_NAMES, NUM_ACTIONS
from .agents import (STYLES, Agent, CallingStation, RandomAgent, RuleBasedAgent,
                     StyleAgent, load_profiles)
from .env import EnvConfig, HandResult, PokerEnv, play_hands

LINEUPS = ('random', 'calling_station', 'rule_based', 'dataset_styles')
BASELINE_HEROES = ('random', 'calling_station', 'rule_based')


def _seed(rng: random.Random) -> random.Random:
    return random.Random(rng.getrandbits(32))


def load_style_data(path: Path) -> tuple[dict, dict]:
    """(profiles, style weights) from ``style_profiles.json``."""

    data = load_profiles(path)
    return data['profiles'], data['weights']


def lineup_factory(kind: str, profiles: dict | None = None, weights: dict | None = None,
                   opponents: int = 5):
    """A function rng -> list of opponents, called once per session."""

    def styles(rng):
        names = [s for s in STYLES if s in profiles]
        w = [weights.get(s, 1.0) for s in names]
        return [StyleAgent(rng.choices(names, w)[0], profiles, _seed(rng))
                for _ in range(opponents)]

    def mix(rng):
        # Training mixture: each seat independently drawn.
        table = []

        for _ in range(opponents):
            roll = rng.random()

            if roll < 0.15:
                table.append(RandomAgent(_seed(rng)))
            elif roll < 0.25:
                table.append(CallingStation())
            elif roll < 0.5 or not profiles:
                table.append(RuleBasedAgent())
            else:
                names = [s for s in STYLES if s in profiles]
                w = [weights.get(s, 1.0) for s in names]
                table.append(StyleAgent(rng.choices(names, w)[0], profiles, _seed(rng)))

        return table

    factories = {
        'random': lambda rng: [RandomAgent(_seed(rng)) for _ in range(opponents)],
        'calling_station': lambda rng: [CallingStation() for _ in range(opponents)],
        'rule_based': lambda rng: [RuleBasedAgent() for _ in range(opponents)],
        'dataset_styles': styles,
        'training_mix': mix,
    }

    if kind == 'dataset_styles' and not profiles:
        raise ValueError('dataset_styles needs style profiles (run scripts/preprocess.py)')

    return factories[kind]


def baseline_hero(kind: str, seed: int = 0) -> Agent:
    return {
        'random': lambda: RandomAgent(random.Random(seed)),
        'calling_station': CallingStation,
        'rule_based': RuleBasedAgent,
    }[kind]()


def summarize(results: list[HandResult]) -> dict:
    nets = np.array([r.net_bb for r in results], dtype=float)
    n = len(nets)
    counts = np.zeros(NUM_ACTIONS)

    for r in results:
        for a in r.hero_actions:
            counts[a] += 1

    se = nets.std(ddof=1) / math.sqrt(n) if n > 1 else float('nan')
    return {
        'hands': n,
        'bb_per_100': float(nets.mean() * 100),
        'ci95_bb_per_100': float(1.96 * se * 100),
        'total_bb': float(nets.sum()),
        'std_bb_per_hand': float(nets.std(ddof=1)) if n > 1 else float('nan'),
        'vpip': float(np.mean([r.vpip for r in results])),
        'pfr': float(np.mean([r.pfr for r in results])),
        'showdown_rate': float(np.mean([r.showdown for r in results])),
        'decisions': int(counts.sum()),
        'action_distribution': {
            name: float(c / counts.sum()) if counts.sum() else 0.0
            for name, c in zip(ACTION_NAMES, counts)
        },
    }


def paired_difference(a: list[HandResult], b: list[HandResult]) -> dict:
    """Per-hand difference a - b on identical deals (bb/100, 95% CI)."""

    diff = np.array([x.net_bb - y.net_bb for x, y in zip(a, b)], dtype=float)
    se = diff.std(ddof=1) / math.sqrt(len(diff))
    return {'bb_per_100': float(diff.mean() * 100), 'ci95_bb_per_100': float(1.96 * se * 100)}


def run(hero: Agent, lineup: str, hands: int, seed: int, profiles=None, weights=None,
        num_players: int = 6, stacks=100.0, equity_samples: int = 200) -> list[HandResult]:
    env = PokerEnv(
        lineup_factory(lineup, profiles, weights, num_players - 1),
        EnvConfig(num_players=num_players, stack_bb=stacks, seed=seed,
                  equity_samples=equity_samples),
    )
    return play_hands(env, hero, hands)
