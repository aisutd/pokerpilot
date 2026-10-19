"""Load the PokerPilot dataset and turn it into offline RL transitions.

The dataset (``Dataset/`` in the team repo) logs 12,940 decisions from
1,747 hands. Each row is one decision by one player; the same player's
next decision in the same hand is the next state, and the hand's net result
(``outcome_hand_net_bb``) is the reward when the player's last decision is
reached. Those are *behaviour-policy* transitions: the actions are the
logged players' choices, never the DQN's, so they can only be used
off-policy (offline Q-learning / pre-training), see ``README.md``.

Every hand is also replayed through the engine (``replay_game``) to check
that the simulator's state definitions match the dataset and to get the
exact legal actions (minimum raise, whether raising was allowed).
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .actions import CALL, NUM_ACTIONS, dataset_action, dataset_legal_mask, legal_mask
from .cards import hand_features, parse_cards
from .engine import Hand, IllegalAction
from .features import BB, encode, engine_context, player_equity

DECISIONS = 'poker_decisions_all.csv'
HANDS = 'poker_hands_by_player.csv'
GAMES = 'poker_games.csv'
PLAYERS = 'poker_player_summary.csv'

# Context columns read straight from a decision row.
CONTEXT_COLUMNS = (
    'street_idx', 'position', 'num_players', 'players_in_hand', 'players_yet_to_act',
    'in_position', 'pot_bb', 'to_call_bb', 'stack_bb', 'effective_stack_bb',
    'raises_this_street', 'callers_this_street', 'is_street_aggressor',
    'was_prev_street_aggressor', 'own_prev_action_street', 'last_opp_action',
    'opp_stack_bb', 'opp_hands_seen', 'opp_vpip', 'opp_pfr', 'opp_aggression_factor',
    'opp_fold_to_bet', 'table_avg_vpip',
)

# Columns the replay must reproduce (the rest are opponent statistics).
REPLAY_COLUMNS = CONTEXT_COLUMNS[:17]

# Never used as model input (feature dictionary: "LEAKS - do not use").
OUTCOME_COLUMNS = (
    'outcome_hand_net_bb', 'outcome_ev_net_bb', 'outcome_won_hand', 'outcome_final_street',
)


def find_dataset_dir(start: Path | None = None) -> Path:
    """The ``Dataset`` folder next to (or above) this package."""

    here = (start or Path(__file__)).resolve()

    for parent in [here, *here.parents]:
        candidate = parent / 'Dataset'

        if (candidate / DECISIONS).exists():
            return candidate

    raise FileNotFoundError('Dataset/ folder with poker_decisions_all.csv not found')


@dataclass
class Tables:
    decisions: pd.DataFrame
    hands: pd.DataFrame
    games: pd.DataFrame
    players: pd.DataFrame


def load_tables(dataset_dir: Path | str | None = None) -> Tables:
    root = Path(dataset_dir) if dataset_dir else find_dataset_dir()
    decisions = pd.read_csv(root / DECISIONS, keep_default_na=False, na_values=[''])
    decisions = decisions.sort_values(['game_id', 'decision_idx'], kind='stable').reset_index(drop=True)
    return Tables(
        decisions=decisions,
        hands=pd.read_csv(root / HANDS, keep_default_na=False, na_values=['']),
        games=pd.read_csv(root / GAMES, keep_default_na=False, na_values=['']),
        players=pd.read_csv(root / PLAYERS, keep_default_na=False, na_values=['']),
    )


def _value(row, column):
    value = row[column]

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    return value


def row_context(row) -> dict:
    """Decision context from a dataset row (blank statistics -> None)."""

    context = {column: _value(row, column) for column in CONTEXT_COLUMNS}

    for column in ('own_prev_action_street', 'last_opp_action'):
        context[column] = context[column] or 'none'

    context['opp_hands_seen'] = context['opp_hands_seen'] or 0
    context['opp_stack_bb'] = context['opp_stack_bb'] or 0.0
    return context


def row_cards(row) -> tuple[list[int], list[int]]:
    return parse_cards(row['hole_cards']), parse_cards(_value(row, 'board'))


def decision_seed(decision_id: str) -> int:
    return zlib.crc32(decision_id.encode())


# ---------------------------------------------------------------- replay

@dataclass
class ReplayStep:
    context: dict
    mask: np.ndarray
    action: int  # the logged action in the discrete space


@dataclass
class ReplayResult:
    game_id: str
    ok: bool
    error: str | None = None
    steps: list[ReplayStep] = field(default_factory=list)
    mismatches: dict = field(default_factory=dict)  # column -> count
    payoffs_match: bool = False
    covered_shoves: int = 0  # logged all-ins over an all-in player (= calls)


def _close(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)

    return abs(float(a) - float(b)) <= 0.011


def replay_game(game_rows: pd.DataFrame, hand_rows: pd.DataFrame, game: pd.Series) -> ReplayResult:
    """Replay one hand through the engine, comparing the context at every
    logged decision with the dataset's columns."""

    result = ReplayResult(game_id=game['game_id'], ok=False)
    n = int(game['num_players'])
    seats = hand_rows.sort_values('seat')

    if list(seats['seat']) != list(range(1, n + 1)):
        result.error = 'seats are not 1..n'
        return result

    stacks = [int(round(s * BB)) for s in seats['starting_stack_bb']]
    holes = [parse_cards(h) for h in seats['hole_cards']]
    board = parse_cards(_value(game, 'board'))
    used = {c for h in holes for c in h} | set(board)
    filler = [c for c in range(52) if c not in used]  # board never seen
    runout = board + filler[:5 - len(board)]
    deck = [h[0] for h in holes] + [h[1] for h in holes] + runout
    ids = list(seats['player_id'])
    hand = Hand(stacks, deck)

    for _, row in game_rows.iterrows():
        player = ids.index(row['player_id'])

        if hand.actor != player:
            result.error = f"{row['decision_id']}: engine expected index {hand.actor}, row has {player}"
            return result

        context = engine_context(hand, player)

        for column in REPLAY_COLUMNS:
            if not _close(context[column], row_context(row)[column]):
                result.mismatches[column] = result.mismatches.get(column, 0) + 1

        bounds = hand.raise_range()
        mask = legal_mask(hand.to_call, hand.bets[player], hand.pot, bounds)
        action = dataset_action(
            row['target_action'], float(row['target_chips_added_bb']),
            float(row['to_call_bb']), float(row['pot_bb']), float(row['stack_bb']),
        )
        chips = int(round(float(row['target_chips_added_bb']) * BB))

        try:
            if row['target_action'] == 'fold':
                hand.fold()
            elif chips <= min(hand.to_call, hand.stacks[player]):
                hand.check_or_call()
            elif bounds is None and hand.to_call > 0:
                # The dataset sometimes logs an all-in over an opponent who
                # is already all-in for less. The excess can never be called
                # (pokerkit refuses the raise), so it is a call.
                hand.check_or_call()
                action = CALL
                result.covered_shoves += 1
            else:
                hand.raise_to(hand.bets[player] + chips)
        except IllegalAction as exc:
            result.error = f"{row['decision_id']}: {exc}"
            return result

        result.steps.append(ReplayStep(context, mask, action))

    if not hand.finished:
        result.error = 'hand unfinished after the logged decisions'
        return result

    nets = [int(round(v * BB)) for v in seats['net_bb']]
    result.payoffs_match = all(abs(a - b) <= 1 for a, b in zip(hand.payoffs, nets))
    result.ok = True
    return result


def replay_all(tables: Tables) -> dict[str, ReplayResult]:
    hands_by_game = dict(tuple(tables.hands.groupby('game_id')))
    games = tables.games.set_index('game_id', drop=False)
    results = {}

    for game_id, rows in tables.decisions.groupby('game_id', sort=True):
        results[game_id] = replay_game(rows, hands_by_game[game_id], games.loc[game_id])

    return results


# ------------------------------------------------------------ transitions

@dataclass
class TransitionSet:
    """Arrays for one split. ``returns`` is the realised hand result for the
    acting player (for evaluation only)."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    masks: np.ndarray
    next_masks: np.ndarray
    returns: np.ndarray
    streets: np.ndarray
    equity: np.ndarray  # our Monte Carlo equity (also inside the state)
    decision_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> 'TransitionSet':
        with np.load(path, allow_pickle=False) as data:
            return cls(**{key: data[key] for key in data.files})


def build_transitions(tables: Tables, replays: dict[str, ReplayResult] | None = None,
                      reward: str = 'net', equity_samples: int = 200) -> dict[str, TransitionSet]:
    """One transition per logged decision, grouped by the ``split`` column
    (chronological by game, so no hand appears in two splits)."""

    reward_column = {'net': 'outcome_hand_net_bb', 'ev': 'outcome_ev_net_bb'}[reward]
    d = tables.decisions
    states = []
    actions = []
    masks = []
    equities = []
    replay_steps = {}

    if replays:
        for game_id, result in replays.items():
            if result.ok:
                rows = d.index[d['game_id'] == game_id]

                for index, step in zip(rows, result.steps):
                    replay_steps[index] = step

    for index, row in d.iterrows():
        hole, board = row_cards(row)
        context = row_context(row)
        rng = random.Random(decision_seed(row['decision_id']))
        equity = player_equity(hole, board, context['players_in_hand'], rng, equity_samples)
        equities.append(equity)
        states.append(encode(context, hand_features(hole, board), equity))
        action = dataset_action(
            row['target_action'], float(row['target_chips_added_bb']),
            float(row['to_call_bb']), float(row['pot_bb']), float(row['stack_bb']),
        )

        if index in replay_steps:
            mask = replay_steps[index].mask.copy()
            action = replay_steps[index].action
        else:
            mask = dataset_legal_mask(float(row['to_call_bb']), float(row['pot_bb']), float(row['stack_bb']))

        # The logged action was evidently legal; a size bucket can disagree
        # with the approximate mask only through rounding.
        mask[action] = True
        actions.append(action)
        masks.append(mask)

    states = np.stack(states)
    actions = np.asarray(actions, dtype=np.int64)
    masks = np.stack(masks)
    n = len(d)
    next_index = np.full(n, -1)
    key = list(zip(d['game_id'], d['player_id']))

    last_seen = {}

    for i in range(n - 1, -1, -1):
        if key[i] in last_seen:
            next_index[i] = last_seen[key[i]]

        last_seen[key[i]] = i

    dones = next_index < 0
    outcome = d[reward_column].to_numpy(dtype=np.float32)
    rewards = np.where(dones, outcome, 0.0).astype(np.float32)
    safe_next = np.where(dones, np.arange(n), next_index)
    next_states = states[safe_next]
    next_masks = masks[safe_next].copy()
    next_masks[dones] = False
    split = d['split'].to_numpy()
    out = {}

    for name in ('train', 'val', 'test'):
        rows = np.flatnonzero(split == name)
        out[name] = TransitionSet(
            states=states[rows],
            actions=actions[rows],
            rewards=rewards[rows],
            next_states=next_states[rows],
            dones=dones[rows],
            masks=masks[rows],
            next_masks=next_masks[rows],
            returns=outcome[rows],
            streets=d['street_idx'].to_numpy(dtype=np.int64)[rows],
            equity=np.asarray(equities, dtype=np.float32)[rows],
            decision_ids=d['decision_id'].to_numpy(dtype=str)[rows],
        )

    return out


assert NUM_ACTIONS == 6
