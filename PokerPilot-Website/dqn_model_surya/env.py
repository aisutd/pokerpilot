"""Step 1-2: Wrap the existing poker engine and define its state, actions, and reward."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    from engine.bots import Bot, Personality, SKILLS
    from engine.game import BIG_BLIND, HUMAN_SEAT, SEAT_COUNT, STARTING_STACK, Table
except ImportError:
    # Multiplayer loads a separate engine package with the same module name.
    # These constants and type placeholders cover DQN inference without
    # importing the single-player game implementation.
    BIG_BLIND = 200
    HUMAN_SEAT = 0
    SEAT_COUNT = 6
    STARTING_STACK = 20_000
    Table = object

    from engine.bots import Personality

if TYPE_CHECKING:
    from collections.abc import Mapping


ACTION_NAMES = (
    'fold',
    'check_call',
    'raise_min',
    'raise_2_5x',
    'raise_pot',
    'raise_all_in',
)
ACTION_COUNT = len(ACTION_NAMES)
RANKS = '23456789TJQKA'
SUITS = 'shdc'
CARD_COUNT = len(RANKS) * len(SUITS)
STREET_FEATURE_COUNT = 4
POSITION_FEATURE_COUNT = 2
SCALAR_FEATURE_NAMES = (
    'pot_bb',
    'to_call_bb',
    'my_bet_bb',
    'stack_bb',
    'effective_stack_bb',
    'spr',
    'opponent_count',
    'mean_opponent_vpip',
    'mean_opponent_aggression',
    'mean_opponent_fold_to_bet',
    'opponent_raise_fraction',
    'opponent_call_fraction',
    'players_behind',
    'raises_this_street',
    'limpers',
    'is_preflop_raiser',
    'pot_odds',
)
MAX_TABLE_BB = SEAT_COUNT * STARTING_STACK / BIG_BLIND
STATE_SIZE = (
    2 * CARD_COUNT
    + STREET_FEATURE_COUNT
    + POSITION_FEATURE_COUNT
    + len(SCALAR_FEATURE_NAMES)
)

PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_HANDS_DATASET = REPOSITORY_ROOT / 'Dataset' / 'poker_hands_by_player.csv'


# The training and test CSVs contain decisions, not state transitions.
# Their hand-level style labels are used only to sample opponent personalities.
STYLE_PERSONALITIES = {
    'Rock': Personality(looseness=0.55, aggression=0.6, bluffing=0.3),
    'Tight-Passive': Personality(looseness=0.7, aggression=0.65, bluffing=0.35, stickiness=0.02),
    'Tight-Aggressive': Personality(looseness=0.85, aggression=1.2, bluffing=0.75),
    'Balanced Reg': Personality(looseness=1.0, aggression=1.0, bluffing=0.9),
    'Loose-Passive': Personality(looseness=1.35, aggression=0.7, bluffing=0.55, stickiness=0.1),
    'Loose-Aggressive': Personality(looseness=1.25, aggression=1.45, bluffing=1.3),
    'Maniac': Personality(looseness=1.65, aggression=1.8, bluffing=1.7, stickiness=0.05),
}


def load_style_distribution(
    split: str,
    dataset_path: Path = DEFAULT_HANDS_DATASET,
) -> dict[str, int]:
    """Count style labels in one data split without loading player identities."""

    if split not in {'train', 'test'}:
        raise ValueError("The opponent style split must be 'train' or 'test'.")

    counts: Counter[str] = Counter()
    with dataset_path.open(newline='', encoding='utf-8') as dataset:
        for row in csv.DictReader(dataset):
            if row.get('split', '').strip().lower() != split:
                continue
            style = row.get('play_style', '').strip()
            if style:
                if style not in STYLE_PERSONALITIES:
                    raise ValueError(f'Unsupported dataset play style: {style!r}.')
                counts[style] += 1

    if not counts:
        raise ValueError(f'No opponent styles were found in the {split!r} data split.')
    return dict(counts)


def _raise_limits(table: Table, seat: int) -> dict | None:
    """Legal action info for ``seat`` (the human's or a bot's turn)."""

    if hasattr(table, 'legal_actions'):
        try:
            return table.legal_actions(seat)
        except TypeError:
            return table.legal_actions()
    if table.actor_seat != seat:
        return None

    state = table.state
    can_raise = state.can_complete_bet_or_raise_to()
    return {
        'fold': state.checking_or_calling_amount > 0,
        'raise': {
            'min': state.min_completion_betting_or_raising_to_amount,
            'max': state.max_completion_betting_or_raising_to_amount,
        } if can_raise else None,
    }


def legal_action_mask(table: Table, seat: int = HUMAN_SEAT) -> np.ndarray:
    """Return one True/False value per action; False actions are never selectable."""

    legal = _raise_limits(table, seat)
    if legal is None:
        return np.zeros(ACTION_COUNT, dtype=np.bool_)

    can_raise = legal['raise'] is not None
    return np.asarray(
        [legal['fold'], True, can_raise, can_raise, can_raise, can_raise],
        dtype=np.bool_,
    )


def _card_index(card: object) -> int:
    rank = getattr(card, 'rank').value
    suit = getattr(card, 'suit').value
    return RANKS.index(rank) * len(SUITS) + SUITS.index(suit)


def observation_from_table(table: Table, seat: int = HUMAN_SEAT) -> np.ndarray:
    """Encode only the agent's cards and public table/opponent statistics.

    ``seat`` is the seat the agent plays (the human seat for the advisor,
    a bot seat when the DQN is chosen as the opponents' model)."""

    if table.actor_seat != seat:
        raise RuntimeError('The agent observation is only available on its turn.')

    view = table.bot_view(seat)
    state = np.zeros(STATE_SIZE, dtype=np.float32)

    for card in view.hole:
        state[_card_index(card)] = 1.0
    board_offset = CARD_COUNT
    for card in view.board:
        state[board_offset + _card_index(card)] = 1.0

    offset = 2 * CARD_COUNT
    state[offset + view.street] = 1.0
    offset += STREET_FEATURE_COUNT

    relative_seat = (seat - table.button) % SEAT_COUNT
    state[offset] = math.cos(2 * math.pi * relative_seat / SEAT_COUNT)
    state[offset + 1] = math.sin(2 * math.pi * relative_seat / SEAT_COUNT)
    offset += POSITION_FEATURE_COUNT

    opponents = view.opponents
    opponent_count = len(opponents)
    opponent_states = table.view()['seats']
    seat_number = seat
    opponent_stacks = [
        seat['stack']
        for seat_index, seat in enumerate(opponent_states)
        if seat_index != seat_number and seat['status'] in {'active', 'all-in'}
    ]
    effective_stack_bb = (
        min(view.my_stack, max(opponent_stacks)) / BIG_BLIND
        if opponent_stacks else 0.0
    )
    raise_count = sum(opponent.preflop == 'raise' for opponent in opponents)
    call_count = sum(opponent.preflop == 'call' for opponent in opponents)
    aggression = (
        sum(math.log1p(max(0.0, opponent.aggression_factor)) for opponent in opponents)
        / opponent_count / math.log(11)
        if opponent_count else 0.0
    )
    aggression = min(1.0, aggression)
    pot_odds = (
        view.to_call / (view.pot + view.to_call)
        if view.pot + view.to_call else 0.0
    )

    scalars = (
        min(1.0, view.pot / BIG_BLIND / MAX_TABLE_BB),
        min(1.0, view.to_call / BIG_BLIND / MAX_TABLE_BB),
        min(1.0, view.my_bet / BIG_BLIND / MAX_TABLE_BB),
        min(1.0, view.my_stack / BIG_BLIND / MAX_TABLE_BB),
        min(1.0, effective_stack_bb / MAX_TABLE_BB),
        min(1.0, view.my_stack / max(view.pot, 1) / 10),
        opponent_count / 5,
        sum(opponent.vpip_rate for opponent in opponents) / opponent_count if opponents else 0.0,
        aggression,
        sum(opponent.fold_to_bet for opponent in opponents) / opponent_count if opponents else 0.0,
        raise_count / opponent_count if opponent_count else 0.0,
        call_count / opponent_count if opponent_count else 0.0,
        view.players_behind / 5,
        min(view.raises_this_street, 5) / 5,
        view.limpers / 5,
        float(view.was_preflop_raiser),
        pot_odds,
    )
    state[offset:] = np.asarray(scalars, dtype=np.float32)
    return state


def action_to_engine_action(
        table: Table, action_index: int, seat: int = HUMAN_SEAT,
) -> tuple[str, int | None]:
    """Convert a legal DQN action into the engine's existing action format."""

    if (
            isinstance(action_index, bool)
            or not isinstance(action_index, (int, np.integer))
            or not 0 <= action_index < ACTION_COUNT
    ):
        raise ValueError(f'Unknown DQN action index: {action_index!r}.')
    action_index = int(action_index)
    if not legal_action_mask(table, seat)[action_index]:
        raise ValueError(f'Illegal DQN action: {ACTION_NAMES[action_index]}.')

    if action_index == 0:
        return 'fold', None
    if action_index == 1:
        return 'check_call', None

    raise_limits = _raise_limits(table, seat)['raise']
    bot_view = table.bot_view(seat)
    low, high = raise_limits['min'], raise_limits['max']
    if action_index == 2:
        amount = low
    elif action_index == 3:
        current_bet = max(bot_view.my_bet + bot_view.to_call, BIG_BLIND)
        amount = round(2.5 * current_bet)
    elif action_index == 4:
        amount = round(
            bot_view.my_bet + bot_view.to_call + bot_view.pot + bot_view.to_call,
        )
    else:
        amount = high

    # The action represents a sizing bucket; clamp it to the engine's legal range.
    return 'raise', max(low, min(high, int(amount)))


class PokerDQNEnv:
    """A one-hand learning episode around the engine's existing Table class."""

    def __init__(
        self,
        seed: int = 0,
        difficulty: str = 'medium',
        style_split: str = 'train',
        style_distribution: Mapping[str, int] | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.difficulty = difficulty
        self.style_split = style_split
        self.style_distribution = dict(
            load_style_distribution(style_split)
            if style_distribution is None else style_distribution,
        )
        if not self.style_distribution:
            raise ValueError('The opponent style distribution cannot be empty.')
        for style, weight in self.style_distribution.items():
            if style not in STYLE_PERSONALITIES or weight <= 0:
                raise ValueError(f'Invalid opponent style entry: {style!r}={weight!r}.')
        self.table: Table | None = None

    def _new_table(self) -> None:
        self.table = _DatasetStyleTable(
            self.difficulty,
            seed=self.rng.randrange(2**32),
            style_rng=random.Random(self.rng.randrange(2**32)),
            style_distribution=self.style_distribution,
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Start the next hand, preserving stacks and public opponent tendencies."""

        if seed is not None:
            self.rng.seed(seed)
        if self.table is not None and not self.table.hand_over:
            raise RuntimeError('Finish the current hand before resetting the environment.')

        skipped_rewards: list[float] = []
        if self.table is None:
            self._new_table()
        while self.table.hand_over:
            if self.table.result is not None:
                skipped_rewards.append(
                    self.table.result['human_net'] / BIG_BLIND,
                )
            if self.table.game_over:
                self._new_table()
                continue
            self.table.next_hand()

        if self.table is None or self.table.hand_over:
            raise RuntimeError('The poker engine did not start a playable hand.')

        return (
            observation_from_table(self.table),
            legal_action_mask(self.table),
            {'skipped_hand_rewards_bb': skipped_rewards},
        )

    def step(
        self,
        action_index: int,
    ) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """Apply one legal agent action and settle the reward when the hand ends."""

        if self.table is None:
            raise RuntimeError('Call reset() before taking an environment step.')
        action, amount = action_to_engine_action(self.table, action_index)
        self.table.human_action(action, amount)
        done = self.table.hand_over
        reward = (
            self.table.result['human_net'] / BIG_BLIND
            if done and self.table.result is not None else 0.0
        )
        if done:
            next_state = np.zeros(STATE_SIZE, dtype=np.float32)
            next_mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        else:
            next_state = observation_from_table(self.table)
            next_mask = legal_action_mask(self.table)

        return next_state, next_mask, float(reward), done, {
            'net_bb': float(reward) if done else None,
            'hand_number': self.table.hand_number,
        }


class _DatasetStyleTable(Table):
    """Install dataset-derived fixed bot styles before the engine deals its first hand."""

    def __init__(
        self,
        difficulty: str,
        seed: int,
        style_rng: random.Random,
        style_distribution: Mapping[str, int],
    ) -> None:
        self.style_rng = style_rng
        self.style_distribution = style_distribution
        self.styles_installed = False
        super().__init__(difficulty, seed=seed)

    def start_hand(self) -> None:
        if not self.styles_installed:
            styles = sorted(self.style_distribution)
            weights = [self.style_distribution[style] for style in styles]
            for seat in self.seats[1:]:
                style = self.style_rng.choices(styles, weights=weights, k=1)[0]
                seat.bot = Bot(
                    SKILLS[self.difficulty],
                    STYLE_PERSONALITIES[style],
                    random.Random(self.style_rng.randrange(2**32)),
                )
            self.styles_installed = True
        super().start_hand()
