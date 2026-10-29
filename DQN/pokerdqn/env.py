"""Multi-player No-Limit Hold'em simulator with one learning seat (the hero).

One episode is one hand. ``reset()`` deals a hand and lets the opponents act
until it is the hero's turn; ``step(action)`` applies the hero's action and
plays on until the hero must act again or the hand ends. The reward is the
hero's net result in big blinds, paid when the hand ends (0 before), so an
episode's return is exactly the chips won or lost, blinds included.

Hands in which the hero never gets to act (e.g. everyone folds to the big
blind) still count in ``results`` but produce no transition.

Seats are fixed and the button moves one seat clockwise every hand, so the
hero plays every position. Opponents stay for ``session_hands`` hands and
their statistics (VPIP, PFR, ...) accumulate over the session, exactly like
the dataset's "earlier games only" columns; then a new lineup is drawn.

Randomness is split into independent streams (deck, stacks, opponents) so
that two heroes evaluated with the same seed get the same cards and stacks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .actions import ACTION_NAMES, NUM_ACTIONS, chips_for, legal_mask
from .agents import Agent, Observation
from .engine import Hand
from .features import BB, POSITION_NAMES, engine_context, main_opponent
from .stats import StatsTracker

LineupFactory = Callable[[random.Random], Sequence[Agent]]


@dataclass
class HandResult:
    net_bb: float
    hero_actions: list[int]
    position: str
    showdown: bool
    vpip: bool  # put money in voluntarily preflop
    pfr: bool  # raised preflop
    hero_hole: tuple[int, int]


@dataclass
class EnvConfig:
    num_players: int = 6
    session_hands: int = 200
    stack_bb: float | Sequence[float] = 100.0  # fixed, or values to sample from
    min_stack_bb: float = 20.0
    equity_samples: int = 200
    seed: int = 0


class PokerEnv:
    def __init__(self, lineup: LineupFactory, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()

        if not 2 <= self.config.num_players <= 9:
            raise ValueError('num_players must be 2..9')

        self.lineup_factory = lineup
        seed = self.config.seed
        self.deck_rng = random.Random(seed * 7919 + 1)
        self.stack_rng = random.Random(seed * 7919 + 2)
        self.lineup_rng = random.Random(seed * 7919 + 3)
        self.equity_rng = random.Random(seed * 7919 + 4)
        self.hero_seat = 0
        self.button = self.config.num_players - 1
        self.hands_in_session = 0
        self.opponents: list[Agent] = []
        self.tracker = StatsTracker()
        self.hand: Hand | None = None
        self.order: list[int] = []
        self.results: list[HandResult] = []
        self.action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        self._hero_actions: list[int] = []
        self._new_session()

    # ------------------------------------------------------------ setup

    def _new_session(self) -> None:
        lineup = list(self.lineup_factory(self.lineup_rng))

        if len(lineup) != self.config.num_players - 1:
            raise ValueError(f'lineup needs {self.config.num_players - 1} opponents')

        self.opponents = lineup
        self.tracker = StatsTracker()
        self.hands_in_session = 0

    def _stack(self) -> int:
        stack = self.config.stack_bb

        if not isinstance(stack, (int, float)):
            stack = self.stack_rng.choice(stack)

        return int(round(max(stack, self.config.min_stack_bb) * BB))

    def _agent(self, seat: int) -> Agent | None:
        if seat == self.hero_seat:
            return None

        return self.opponents[seat - 1 if seat > self.hero_seat else seat]

    def _deal(self) -> None:
        if self.hands_in_session >= self.config.session_hands:
            self._new_session()

        n = self.config.num_players
        self.button = (self.button + 1) % n
        # Engine index 0 is the small blind (the button itself heads-up).
        first = self.button if n == 2 else (self.button + 1) % n
        self.order = [(first + i) % n for i in range(n)]
        deck = list(range(52))
        self.deck_rng.shuffle(deck)
        stacks = [self._stack() for _ in range(n)]
        self.hand = Hand(stacks, deck)
        self._hero_actions = []

    # ------------------------------------------------------ observations

    def observation(self, index: int) -> Observation:
        """What the player at engine ``index`` (the actor) can see."""

        hand = self.hand
        seat_ids = self.order
        rival = main_opponent(hand, index)
        opponent = self.tracker.summary(seat_ids[rival]) if rival is not None else None

        vpips = [
            self.tracker.summary(seat_ids[i])['vpip']
            for i in hand.in_hand() if i != index
        ]
        context = engine_context(hand, index, opponent, vpips)
        mask = legal_mask(hand.to_call, hand.bets[index], hand.pot, hand.raise_range())
        return Observation(
            context, list(hand.hole[index]), list(hand.board), mask, self.equity_rng,
            self.config.equity_samples,
        )

    def _apply(self, action: int) -> None:
        hand = self.hand
        bounds = hand.raise_range()
        mask = legal_mask(hand.to_call, hand.bets[hand.actor], hand.pot, bounds)

        if not mask[action]:
            raise ValueError(f'illegal action {ACTION_NAMES[action]} (legal: '
                             f'{[ACTION_NAMES[a] for a in np.flatnonzero(mask)]})')

        target = chips_for(action, hand.to_call, hand.bets[hand.actor], hand.pot, bounds)

        if action == 0:
            hand.fold()
        elif target is None:
            hand.check_or_call()
        else:
            hand.raise_to(target)

    def _hero_index(self) -> int:
        return self.order.index(self.hero_seat)

    def _run_opponents(self) -> None:
        hand = self.hand

        while not hand.finished and self.order[hand.actor] != self.hero_seat:
            agent = self._agent(self.order[hand.actor])
            self._apply(agent.act(self.observation(hand.actor)))

    def _finish_hand(self) -> float:
        hand = self.hand
        hero = self._hero_index()
        self.tracker.record_hand(hand, self.order)
        self.hands_in_session += 1
        net = hand.payoffs[hero] / BB
        preflop = [r for r in hand.history if r.street == 0 and r.player == hero]
        self.results.append(HandResult(
            net, list(self._hero_actions),
            POSITION_NAMES[hand.n][hero], hand.showdown and not hand.folded[hero],
            any(r.added > 0 or r.aggressive for r in preflop),
            any(r.aggressive for r in preflop),
            tuple(hand.hole[hero]),
        ))
        return net

    # --------------------------------------------------------- gym API

    def reset(self) -> Observation:
        """Deal hands until the hero has a decision; return its observation."""

        while True:
            self._deal()
            self._run_opponents()

            if not self.hand.finished:
                return self.observation(self._hero_index())

            self._finish_hand()

    def step(self, action: int) -> tuple[Observation | None, float, bool, dict]:
        if self.hand is None or self.hand.finished or self.order[self.hand.actor] != self.hero_seat:
            raise RuntimeError('call reset() first; it is not the hero to act')

        self._apply(action)
        self._hero_actions.append(action)
        self.action_counts[action] += 1
        self._run_opponents()

        if self.hand.finished:
            net = self._finish_hand()
            return None, net, True, {'net_bb': net}

        return self.observation(self._hero_index()), 0.0, False, {}


def play_hands(env: PokerEnv, hero: Agent, hands: int) -> list[HandResult]:
    """Let a fixed policy play ``hands`` complete hands (for evaluation)."""

    start = len(env.results)

    while len(env.results) - start < hands:
        obs = env.reset()
        done = False

        while not done:
            obs, _, done, _ = env.step(hero.act(obs))

    return env.results[start:start + hands]
