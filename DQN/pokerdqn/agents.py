"""Non-learning players: baselines for evaluation and opponents for training.

    RandomAgent         uniform over the legal discrete actions
    CallingStation      always checks or calls
    RuleBasedAgent      a simple tight-aggressive heuristic (hand-strength
                        thresholds, pot odds)
    StyleAgent          samples actions from the frequencies logged for one
                        of the dataset's seven play styles (train split only),
                        conditioned on street, bet faced, raises so far and a
                        hand-strength bucket

Every agent sees only an ``Observation``: its own cards, public betting and
opponent statistics. None of them can see other players' hole cards.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .actions import ALL_IN, CALL, CHECK, FOLD, NUM_ACTIONS, RAISE_HALF, RAISE_POT
from .cards import hand_features, parse_cards
from .features import encode, player_equity


@dataclass
class Observation:
    context: dict
    hole: list[int]
    board: list[int]
    mask: np.ndarray
    rng: random.Random
    equity_samples: int = 200
    cards: dict = field(init=False)
    _equity: float | None = field(default=None, init=False)
    _vector: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.cards = hand_features(self.hole, self.board)

    def equity(self) -> float:
        if self._equity is None:
            self._equity = player_equity(
                self.hole, self.board, self.context['players_in_hand'], self.rng,
                self.equity_samples,
            )

        return self._equity

    def vector(self) -> np.ndarray:
        if self._vector is None:
            self._vector = encode(self.context, self.cards, self.equity())

        return self._vector


class Agent:
    name = 'agent'

    def act(self, obs: Observation) -> int:
        raise NotImplementedError

    def first_legal(self, obs: Observation, *preferences: int) -> int:
        for action in preferences:
            if obs.mask[action]:
                return action

        return int(np.flatnonzero(obs.mask)[0])


class RandomAgent(Agent):
    name = 'random'

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def act(self, obs: Observation) -> int:
        return self.rng.choice(list(np.flatnonzero(obs.mask)))


class CallingStation(Agent):
    name = 'calling_station'

    def act(self, obs: Observation) -> int:
        return self.first_legal(obs, CHECK, CALL)


def strength_bucket(cards: dict, street: int) -> int:
    """0 (trash) .. 4 (very strong), from the cards alone (no simulation)."""

    if street == 0:
        p = cards['preflop_percentile']
        return 0 if p < 0.4 else 1 if p < 0.6 else 2 if p < 0.75 else 3 if p < 0.9 else 4

    pair = cards['pair_type']

    if cards['made_hand_rank'] >= 2 and pair != 'board pair only':
        return 4

    if pair in ('top pair', 'overpair'):
        return 3

    if pair == 'middle pair':
        return 2

    if pair in ('bottom pair', 'underpair') or cards['draw_outs'] >= 8:
        return 1

    return 0


class RuleBasedAgent(Agent):
    """Tight-aggressive rules of thumb; deterministic given the state."""

    name = 'rule_based'

    def act(self, obs: Observation) -> int:
        c = obs.context
        pot, to_call = c['pot_bb'], c['to_call_bb']
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
        raises = c['raises_this_street']

        if c['street_idx'] == 0:
            p = obs.cards['preflop_percentile']

            if raises == 0:
                if p >= 0.8:
                    return self.first_legal(obs, RAISE_POT, RAISE_HALF, ALL_IN, CALL, CHECK)
                if p >= 0.6 and to_call <= 1.0:
                    return self.first_legal(obs, CHECK, CALL)
                return self.first_legal(obs, CHECK, FOLD)

            if raises == 1:
                if p >= 0.93:
                    return self.first_legal(obs, RAISE_POT, ALL_IN, CALL)
                if p >= 0.8:
                    return self.first_legal(obs, CHECK, CALL)
                return self.first_legal(obs, CHECK, FOLD)

            if p >= 0.97:
                return self.first_legal(obs, ALL_IN, CALL)
            if p >= 0.93:
                return self.first_legal(obs, CHECK, CALL)
            return self.first_legal(obs, CHECK, FOLD)

        strength = strength_bucket(obs.cards, c['street_idx'])

        if strength == 4:
            return self.first_legal(obs, RAISE_POT, RAISE_HALF, ALL_IN, CALL, CHECK)

        if strength == 3:
            if to_call == 0:
                return self.first_legal(obs, RAISE_HALF, CHECK)
            return self.first_legal(obs, CALL if pot_odds < 0.4 else FOLD)

        limit = {2: 0.3, 1: 0.25}.get(strength, 0.0)

        if to_call == 0:
            return CHECK

        return self.first_legal(obs, CALL if pot_odds < limit else FOLD)


# --------------------------------------------------- dataset-calibrated

def style_key(context: dict, cards: dict) -> tuple:
    return (
        int(context['street_idx'] > 0),
        int(context['to_call_bb'] > 0),
        min(int(context['raises_this_street']), 2),
        strength_bucket(cards, int(context['street_idx'])),
    )


def _backoffs(key: tuple) -> list[str]:
    street, facing, raises, strength = key
    return [
        f'{street}|{facing}|{raises}|{strength}',
        f'{street}|{facing}|*|{strength}',
        f'{street}|{facing}|*|*',
    ]


def build_style_profiles(decisions, hands, actions: np.ndarray) -> dict:
    """Action counts per play style and situation, from the given rows.

    ``decisions`` must contain only training-split rows; ``actions`` holds
    their actions in the discrete space (same order).
    """

    style_of = dict(zip(hands['player_id'], hands['play_style']))
    profiles: dict[str, dict[str, list[int]]] = {}

    for (_, row), action in zip(decisions.iterrows(), actions):
        context = {
            'street_idx': int(row['street_idx']),
            'to_call_bb': float(row['to_call_bb']),
            'raises_this_street': int(row['raises_this_street']),
        }
        cards = hand_features(parse_cards(row['hole_cards']),
                              parse_cards(row['board'] if isinstance(row['board'], str) else ''))
        style = style_of[row['player_id']]
        table = profiles.setdefault(style, {})

        for name in _backoffs(style_key(context, cards)):
            counts = table.setdefault(name, [0] * NUM_ACTIONS)
            counts[int(action)] += 1

    return profiles


def save_profiles(profiles: dict, path: Path) -> None:
    path.write_text(json.dumps(profiles, indent=1, sort_keys=True))


def load_profiles(path: Path) -> dict:
    return json.loads(Path(path).read_text())


class StyleAgent(Agent):
    """Samples from a play style's logged action frequencies, smoothing
    each situation towards the coarser one above it."""

    def __init__(self, style: str, profiles: dict, rng: random.Random | None = None,
                 smoothing: float = 4.0) -> None:
        if style not in profiles:
            raise ValueError(f'unknown style {style!r}; known: {sorted(profiles)}')

        self.style = style
        self.name = 'style:' + style
        self.table = profiles[style]
        self.rng = rng or random.Random()
        self.smoothing = smoothing

    def probabilities(self, obs: Observation) -> np.ndarray:
        legal = obs.mask.astype(float)
        probs = legal / legal.sum()

        for name in reversed(_backoffs(style_key(obs.context, obs.cards))):
            counts = np.asarray(self.table.get(name, [0] * NUM_ACTIONS), dtype=float) * legal
            probs = (counts + self.smoothing * probs) / (counts.sum() + self.smoothing)

        probs = probs * legal
        return probs / probs.sum()

    def act(self, obs: Observation) -> int:
        probs = self.probabilities(obs)
        return int(self.rng.choices(range(NUM_ACTIONS), weights=probs)[0])


STYLES = (
    'Tight-Aggressive', 'Balanced Reg', 'Loose-Aggressive', 'Loose-Passive',
    'Rock', 'Maniac', 'Tight-Passive',
)
