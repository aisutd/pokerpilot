"""Pick which AI model drives the five opponents (PvE mode).

The lobby lets the player choose one of these models:

    rules    Classic hand-written bots in engine/bots.py (always available)
    xgboost  The team's XGBoost model in ../pokerpilot_model/, trained to play
             like the winning players in the dataset
    dqn      Surya's DQN in dqn_model_surya/ (needs a trained checkpoint)

Difficulty still matters for the ML models: it sets how greedily a bot follows
the model. Hard takes the model's best action almost every time, Medium samples
from the model's probabilities, Easy plays loosely and makes more mistakes.
Each character's personality (aggression, looseness) nudges the model's
probabilities so the five opponents still feel different.

To add another model: write a class with ``decide_at_table(table, seat)``
returning ('fold', None), ('check_call', None) or ('raise', amount_to), add an
entry to OPPONENT_MODELS, and handle it in create_opponent().
"""

from __future__ import annotations

import math
import random
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

from pokerkit import Card, Deck, StandardHighHand

from .bots import (
    RANK_ORDER,
    Bot,
    BotView,
    Character,
    Decision,
    Personality,
    chen_score,
    create_bot,
    equity_vs_random,
)

if TYPE_CHECKING:
    from .game import Table

WEBSITE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WEBSITE_DIR.parent
XGBOOST_DIR = REPO_ROOT / 'pokerpilot_model'
DQN_CHECKPOINT = WEBSITE_DIR / 'dqn_model_surya' / 'checkpoints' / 'pokerpilot_dqn.pt'


class ModelUnavailable(Exception):
    """The chosen model can't be loaded (missing package or trained file)."""


@dataclass(frozen=True)
class OpponentModel:
    key: str
    label: str
    description: str


OPPONENT_MODELS = {
    'rules': OpponentModel(
        'rules', 'Classic Bots',
        'Hand-written strategy with a personality for each player.',
    ),
    'xgboost': OpponentModel(
        'xgboost', 'XGBoost',
        'Learned from the dataset to play like its winning players.',
    ),
    'dqn': OpponentModel(
        'dqn', 'DQN',
        'Reinforcement-learning agent trained by self-play.',
    ),
}


# ------------------------------------------------------------ availability

def _missing_packages(*names: str) -> list[str]:
    missing = []

    for name in names:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)

    return missing


def availability(key: str) -> str | None:
    """None if the model can be used, else a short reason why not."""

    if key == 'rules':
        return None

    if key == 'xgboost':
        missing = _missing_packages('xgboost', 'pandas')

        if missing:
            return f'Install {" and ".join(missing)} (pip install -r requirements.txt).'

        if not (XGBOOST_DIR / 'artifacts' / 'action_model.json').is_file():
            return 'No trained model in pokerpilot_model/artifacts. Run train.py there.'

        return None

    if key == 'dqn':
        if _missing_packages('torch'):
            return 'Install torch (pip install -r requirements.txt).'

        if not DQN_CHECKPOINT.is_file():
            return 'Train the DQN first (see dqn_model_surya/README.md).'

        return None

    return 'Unknown model.'


def list_models() -> list[dict]:
    """What the lobby shows: every model, and whether it can be picked."""

    models = []

    for model in OPPONENT_MODELS.values():
        reason = availability(model.key)
        models.append({
            'key': model.key,
            'label': model.label,
            'description': model.description,
            'available': reason is None,
            'reason': reason,
        })

    return models


def create_opponent(
        model: str, difficulty: str, character: Character, rng: random.Random | None = None,
):
    if model not in OPPONENT_MODELS:
        raise ValueError(f'Unknown AI model: {model!r}')

    reason = availability(model)

    if reason is not None:
        raise ModelUnavailable(f'{OPPONENT_MODELS[model].label} is not available: {reason}')

    if model == 'rules':
        return create_bot(difficulty, character, rng)

    if model == 'xgboost':
        return XGBoostBot(difficulty, character.personality, rng)

    return DQNBot(difficulty, rng)


# How closely each difficulty follows the model.
#   temperature: <1 sharpens the model's probabilities, >1 flattens them
#   blunder:     chance of picking a random legal action instead
@dataclass(frozen=True)
class ModelSkill:
    temperature: float
    blunder: float


MODEL_SKILLS = {
    'easy': ModelSkill(temperature=1.8, blunder=0.15),
    'medium': ModelSkill(temperature=1.0, blunder=0.03),
    'hard': ModelSkill(temperature=0.35, blunder=0.0),
}


# ------------------------------------------------- table -> dataset features

DATASET_POSITIONS = {
    # Same names as poker_decisions_*.csv, in pokerkit's order after the
    # blinds: early position first, button last.
    3: ['BTN'],
    4: ['CO', 'BTN'],
    5: ['HJ', 'CO', 'BTN'],
    6: ['UTG', 'HJ', 'CO', 'BTN'],
}

HAND_LABELS = {
    'High card': 'High Card',
    'One pair': 'Pair',
    'Two pair': 'Two Pair',
    'Three of a kind': 'Three of a Kind',
    'Straight': 'Straight',
    'Flush': 'Flush',
    'Full house': 'Full House',
    'Four of a kind': 'Four of a Kind',
    'Straight flush': 'Straight Flush',
}
HAND_RANKS = {label: i for i, label in enumerate(HAND_LABELS.values())}
HAND_INDEX_MAX = 7461  # pokerkit has 7,462 distinct 5-card hands; higher = stronger


_CHEN_SCORES = sorted(chen_score(*combo) for combo in combinations(Deck.STANDARD, 2))


def preflop_percentile(score: float) -> float:
    """Share of the 1,326 starting hands this Chen score beats (1 = best),
    counting ties as half, the same way the dataset computes it."""

    below = bisect_left(_CHEN_SCORES, score)
    tied = bisect_right(_CHEN_SCORES, score) - below

    return (below + tied / 2) / len(_CHEN_SCORES)


def rank_value(card: Card) -> int:
    return RANK_ORDER.index(card.rank.value) + 2  # 2..14, A = 14


def position_name(order: list[int], seat: int) -> str:
    """Dataset position name. ``order`` is pokerkit's order: small blind
    first and button last (heads-up: big blind, then button/small blind)."""

    index = order.index(seat)

    if len(order) == 2:
        return 'BB' if index == 0 else 'BTN/SB'

    if index == 0:
        return 'SB'

    if index == 1:
        return 'BB'

    return DATASET_POSITIONS[len(order)][index - 2]


def straight_draw(cards: list[Card], hole: list[Card]) -> str:
    """'open-ended', 'gutshot' or 'none' (draws must use a hole card)."""

    ranks = {rank_value(c) for c in cards}
    hole_ranks = {rank_value(c) for c in hole}

    if 14 in ranks:
        ranks.add(1)

    if 14 in hole_ranks:
        hole_ranks.add(1)

    # Already a straight: no draw.
    for low in range(1, 11):
        if all(r in ranks for r in range(low, low + 5)):
            return 'none'

    best = 'none'

    for low in range(1, 11):
        window = range(low, low + 5)
        missing = [r for r in window if r not in ranks]

        if len(missing) != 1 or not hole_ranks & set(window):
            continue

        # Open-ended: four in a row that can be completed on either end.
        run = [r for r in window if r in ranks]

        if missing[0] in (low, low + 4) and run == list(range(run[0], run[0] + 4)) \
                and 1 < run[0] and run[-1] < 14:
            return 'open-ended'

        best = 'gutshot'

    return best


def hand_features(hole: list[Card], board: list[Card]) -> dict:
    """Card and board features with the dataset's column names."""

    high, low = sorted((rank_value(c) for c in hole), reverse=True)
    pocket_pair = high == low
    features = {
        'hole_cards': ' '.join(str(c.rank.value) + str(c.suit.value) for c in hole),
        'card_high_rank': high,
        'card_low_rank': low,
        'is_pocket_pair': int(pocket_pair),
        'is_suited': int(hole[0].suit == hole[1].suit),
        'rank_gap': high - low,
        'chen_score': chen_score(*hole),
        'preflop_percentile': preflop_percentile(chen_score(*hole)),
    }

    if not board:
        features.update({
            'made_hand': 'Pair' if pocket_pair else 'High Card',
            'made_hand_rank': int(pocket_pair),
            'pair_type': 'pocket pair' if pocket_pair else 'none',
            'flush_draw': 0,
            'straight_draw': 'none',
            'draw_outs': 0,
        })
        return features

    hand = StandardHighHand.from_game(hole, board)
    made = HAND_LABELS[hand.entry.label.value]
    made_rank = HAND_RANKS[made]
    board_ranks = sorted((rank_value(c) for c in board), reverse=True)

    if made_rank >= 2:
        pair_type = 'two pair or better'
    elif made_rank == 0:
        pair_type = 'none'
    elif pocket_pair and high not in board_ranks:
        pair_type = 'overpair' if high > board_ranks[0] else 'underpair'
    elif high in board_ranks or low in board_ranks:
        paired = high if high in board_ranks else low

        if paired == board_ranks[0]:
            pair_type = 'top pair'
        elif paired == board_ranks[-1]:
            pair_type = 'bottom pair'
        else:
            pair_type = 'middle pair'
    else:
        pair_type = 'board pair only'

    river = len(board) == 5
    all_cards = hole + board
    suit_counts = {}

    for card in all_cards:
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1

    flush_draw = int(
        not river and made_rank < HAND_RANKS['Flush']
        and any(suit_counts.get(c.suit) == 4 for c in hole)
    )
    draw = 'none' if river or made_rank >= HAND_RANKS['Straight'] else straight_draw(all_cards, hole)
    outs = 9 * flush_draw + {'open-ended': 8, 'gutshot': 4}.get(draw, 0)

    if flush_draw and draw != 'none':
        outs -= 2  # shared outs

    board_suits = {}

    for card in board:
        board_suits[card.suit] = board_suits.get(card.suit, 0) + 1

    most_suited = max(board_suits.values())
    unique = set(board_ranks) | ({1} if 14 in board_ranks else set())

    features.update({
        'made_hand': made,
        'made_hand_rank': made_rank,
        'made_hand_strength': hand.entry.index / HAND_INDEX_MAX,
        'pair_type': pair_type,
        'flush_draw': flush_draw,
        'straight_draw': draw,
        'draw_outs': outs,
        'board': ' '.join(str(c.rank.value) + str(c.suit.value) for c in board),
        'board_paired': int(len(set(board_ranks)) < len(board_ranks)),
        'board_suit_texture': 'monotone' if most_suited >= 3 else
        'two-tone' if most_suited == 2 else 'rainbow',
        'board_connectedness': max(
            sum(1 for r in unique if low_end <= r < low_end + 5) for low_end in range(1, 11)
        ),
        'board_high_rank': board_ranks[0],
    })

    return features


def table_state(table: Table, seat: int, equity_samples: int = 150) -> dict:
    """Describe the table from ``seat``'s point of view using the same
    column names as poker_decisions_*.csv. Only public information and the
    seat's own hole cards are used."""

    from .game import BIG_BLIND, STARTING_STACK

    state = table.state
    order = table.order
    index = order.index(seat)
    view: BotView = table.bot_view(seat)
    bb = BIG_BLIND
    street = view.street
    live = [s for i, s in enumerate(order) if state.statuses[i]]
    actions = table.street_actions[-1]

    # Players still to act after us this street (pokerkit order = action order
    # after the flop; preflop it is computed by players_behind).
    if street == 0:
        yet_to_act = view.players_behind
    else:
        yet_to_act = sum(
            1 for i in range(index + 1, len(order))
            if state.statuses[i] and state.stacks[i] > 0
        )

    # Dataset definition: 1 if we act last after the flop among players left.
    in_position = int(all(order.index(s) < index for s in live if s != seat))

    # Last aggressor this street, and on earlier streets.
    aggressor = next((s for s, kind in reversed(actions) if kind in ('bet', 'raise')), None)
    prev_aggressor = None

    for earlier in reversed(table.street_actions[:-1]):
        prev_aggressor = next((s for s, kind in reversed(earlier) if kind in ('bet', 'raise')), None)

        if prev_aggressor is not None:
            break

    callers = 0

    for s, kind in reversed(actions):
        if kind in ('bet', 'raise'):
            break

        callers += kind == 'call'

    own_prev = next((kind for s, kind in reversed(actions) if s == seat), 'none')
    last_any = actions[-1][1] if actions else 'none'

    # Main opponent: last aggressor, else the biggest investor still in.
    others = [s for s in live if s != seat]
    if aggressor is not None and aggressor != seat:
        main = aggressor
    elif others:
        main = max(others, key=lambda s: state.bets[order.index(s)])
    else:
        main = None

    def stats_of(s: int) -> dict:
        stats = table.seats[s].stats
        seen = max(stats.hands - 1, 0)  # earlier hands only, like the dataset

        if seen == 0:
            return {'hands_seen': 0}

        net_bb = (table.seats[s].stack - STARTING_STACK) / bb
        return {
            'hands_seen': seen,
            'vpip': stats.vpip / stats.hands,
            'pfr': stats.pfr / stats.hands,
            'aggression_factor': stats.aggressive / stats.calls if stats.calls else float(stats.aggressive),
            'fold_to_bet': stats.folds_to_bet / stats.faced_bets if stats.faced_bets else math.nan,
            'bb_per_100': net_bb / seen * 100,
        }

    opp = stats_of(main) if main is not None else {'hands_seen': 0}
    vpips = [stats_of(s).get('vpip') for s in others]
    vpips = [v for v in vpips if v is not None]

    biggest_other = max((state.stacks[order.index(s)] for s in others), default=0)
    effective = min(view.my_stack, biggest_other) / bb
    pot_bb = view.pot / bb
    to_call_bb = view.to_call / bb
    hole = view.hole

    result = {
        'street': ('Preflop', 'Flop', 'Turn', 'River')[street],
        'street_idx': street,
        'position': position_name(order, seat),
        'num_players': len(order),
        'players_in_hand': len(live),
        'players_yet_to_act': yet_to_act,
        'in_position': in_position,
        'pot_bb': pot_bb,
        'to_call_bb': to_call_bb,
        'pot_odds': view.to_call / (view.pot + view.to_call) if view.to_call else 0.0,
        'facing_bet': int(view.to_call > 0),
        'facing_bet_pot_frac': view.to_call / view.pot if view.pot else 0.0,
        'stack_bb': view.my_stack / bb,
        'effective_stack_bb': effective,
        'spr': effective / pot_bb if pot_bb else 0.0,
        'raises_this_street': view.raises_this_street,
        'callers_this_street': callers,
        'is_street_aggressor': int(aggressor == seat),
        'was_prev_street_aggressor': int(prev_aggressor == seat),
        'own_prev_action_street': own_prev,
        'last_opp_action': last_any,
        'equity_vs_random': equity_vs_random(hole, view.board, max(len(live) - 1, 1), equity_samples),
        'opp_hands_seen': opp['hands_seen'],
        'opp_vpip': opp.get('vpip', math.nan),
        'opp_pfr': opp.get('pfr', math.nan),
        'opp_aggression_factor': opp.get('aggression_factor', math.nan),
        'opp_fold_to_bet': opp.get('fold_to_bet', math.nan),
        'opp_bb_per_100': opp.get('bb_per_100', math.nan),
        'table_avg_vpip': sum(vpips) / len(vpips) if vpips else math.nan,
    }
    result.update(hand_features(hole, view.board))

    return result


# ------------------------------------------------------------------ XGBoost

@lru_cache(maxsize=1)
def load_xgboost():
    """Load the team's XGBoost models once and share them across games."""

    if str(XGBOOST_DIR) not in sys.path:
        sys.path.insert(0, str(XGBOOST_DIR))

    import pandas as pd
    from features import ACTIONS, apply_mask, build_features
    from predict import PokerPilotModel

    model = PokerPilotModel(XGBOOST_DIR / 'artifacts')

    def predict(state: dict) -> tuple[dict[str, float], float]:
        """Legal-action probabilities and the predicted bet size
        (chips added as a fraction of the pot)."""

        X = build_features(pd.DataFrame([state]))
        probs = apply_mask(model.action_model.predict_proba(X), [state['facing_bet']])[0]
        size = float(model.sizing_model.predict(X)[0])

        return dict(zip(ACTIONS, (float(p) for p in probs))), size

    return predict


class XGBoostBot:
    """Opponent driven by pokerpilot_model's XGBoost action + sizing models."""

    def __init__(self, difficulty: str, personality: Personality, rng: random.Random | None = None):
        self.skill = MODEL_SKILLS[difficulty]
        self.personality = personality
        self.rng = rng or random.Random()
        self.predict = load_xgboost()

    def adjust(self, probs: dict[str, float], facing_bet: bool) -> dict[str, float]:
        """Apply personality and difficulty to the model's probabilities."""

        p = dict(probs)
        p['bet'] *= self.personality.aggression
        p['raise'] *= self.personality.aggression

        if facing_bet:
            # Loose or sticky players fold less.
            p['fold'] /= self.personality.looseness + 2 * self.personality.stickiness

        p = {a: v ** (1 / self.skill.temperature) for a, v in p.items()}
        total = sum(p.values())

        if total <= 0:
            return {'call' if facing_bet else 'check': 1.0}

        return {a: v / total for a, v in p.items() if v > 0}

    def choose(self, probs: dict[str, float]) -> str:
        if self.rng.random() < self.skill.blunder:
            return self.rng.choice(list(probs))

        roll = self.rng.random()

        for action, p in probs.items():
            roll -= p

            if roll <= 0:
                return action

        return max(probs, key=probs.get)

    def decide_at_table(self, table: Table, seat: int) -> Decision:
        state = table_state(table, seat)
        probs, size = self.predict(state)
        action = self.choose(self.adjust(probs, bool(state['facing_bet'])))

        if action == 'fold':
            return 'fold', None

        if action in ('check', 'call'):
            return 'check_call', None

        view = table.bot_view(seat)

        if view.min_raise_to is None:
            return 'check_call', None

        # The sizing model predicts chips added as a fraction of the pot
        # (target_bet_size_pot_frac). Wobble it a little so it isn't readable.
        frac = min(3.0, max(0.25, size * self.rng.uniform(0.9, 1.1)))
        amount = view.my_bet + frac * view.pot
        amount = round(amount / (view.big_blind / 4)) * (view.big_blind / 4)  # whole 50-cent sizes

        # Reuse the classic bot's clamping (min raise, all-in when close).
        return Bot.raise_to(self, view, amount)


# ---------------------------------------------------------------------- DQN

DQN_EPSILON = {'easy': 0.3, 'medium': 0.1, 'hard': 0.0}


class DQNBot:
    """Opponent driven by Surya's trained DQN (dqn_model_surya/)."""

    def __init__(self, difficulty: str, rng: random.Random | None = None):
        self.epsilon = DQN_EPSILON[difficulty]
        self.rng = rng or random.Random()

    def decide_at_table(self, table: Table, seat: int) -> Decision:
        from dqn_model_surya.env import (
            ACTION_NAMES,
            action_to_engine_action,
            legal_action_mask,
            observation_from_table,
        )
        from dqn_model_surya.model import choose_greedy_action, load_policy

        network, metadata = load_policy(str(DQN_CHECKPOINT), DQN_CHECKPOINT.stat().st_mtime_ns)
        observation = observation_from_table(table, seat)
        mask = legal_action_mask(table, seat)

        if metadata.get('state_size') != len(observation) or metadata.get('action_names') != list(ACTION_NAMES):
            raise RuntimeError('The trained DQN checkpoint uses incompatible state/actions.')

        if self.rng.random() < self.epsilon:
            chosen = self.rng.choice([i for i, ok in enumerate(mask) if ok])
        else:
            chosen, _ = choose_greedy_action(network, observation, mask)

        return action_to_engine_action(table, chosen, seat)
