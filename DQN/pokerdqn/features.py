"""The state representation shared by dataset rows and simulator states.

A decision is first described as a *context* dict using the dataset's
column names (``pot_bb``, ``to_call_bb``, ``players_yet_to_act``...). For
dataset rows the context comes straight from the CSV; in the simulator
``engine_context`` computes it from the engine with the same definitions
(``tests/test_dataset.py`` replays the dataset's hands through the engine
and checks that both agree). ``encode`` turns a context plus the acting
player's cards into a fixed-length float vector.

Only information available to the acting player when deciding is used: its
own hole cards, the visible board, public betting and stacks, and opponent
statistics from earlier hands. Outcome columns, other players' hole cards,
the logged action and player identities are never part of the state.
"""

from __future__ import annotations

import math
import random

import numpy as np

from .cards import PAIR_TYPES, equity_vs_random, hand_features
from .engine import Hand

BB = 100  # chips per big blind in the simulator

# Position names by table size, by index from the small blind (dataset
# convention: seat = index + 1).
POSITION_NAMES = {
    2: ('BTN/SB', 'BB'),
    3: ('SB', 'BB', 'BTN'),
    4: ('SB', 'BB', 'CO', 'BTN'),
    5: ('SB', 'BB', 'HJ', 'CO', 'BTN'),
    6: ('SB', 'BB', 'UTG', 'HJ', 'CO', 'BTN'),
    7: ('SB', 'BB', 'UTG', 'LJ', 'HJ', 'CO', 'BTN'),
    8: ('SB', 'BB', 'UTG', 'UTG+1', 'LJ', 'HJ', 'CO', 'BTN'),
    9: ('SB', 'BB', 'UTG', 'UTG+1', 'UTG+2', 'LJ', 'HJ', 'CO', 'BTN'),
}

POSITION_GROUPS = ('SB', 'BB', 'BTN', 'CO', 'HJ', 'LJ', 'EARLY')
OWN_PREV = ('none', 'check', 'call', 'bet', 'raise')
LAST_ACTION = ('none', 'fold', 'check', 'call', 'bet', 'raise', 'all-in')
TEXTURES = ('rainbow', 'two-tone', 'monotone')
OPP_STATS = ('vpip', 'pfr', 'aggression_factor', 'fold_to_bet')


# ------------------------------------------------------- engine context

def postflop_order(n: int) -> list[int]:
    return [1, 0] if n == 2 else list(range(n))


def main_opponent(hand: Hand, player: int) -> int | None:
    """Last aggressor still in the hand, else the biggest investor."""

    for record in reversed(hand.history):
        if record.aggressive and record.player != player and not hand.folded[record.player]:
            return record.player

    rivals = [i for i in hand.in_hand() if i != player]

    if not rivals:
        return None

    return max(rivals, key=lambda i: (hand.contrib[i], -i))


def engine_context(hand: Hand, player: int, opp_summary=None, table_vpips=None) -> dict:
    """Decision context for ``player`` (who must be the actor).

    ``opp_summary`` is the main opponent's ``PlayerStats.summary()`` (or
    None) and ``table_vpips`` the historical VPIPs of the opponents still in
    the hand (None entries for unknown players).
    """

    n = hand.n
    street = hand.street
    in_hand = hand.in_hand()
    opponents = [i for i in in_hand if i != player]
    this_street = [r for r in hand.history if r.street == street]
    earlier = [r for r in hand.history if r.street < street]

    yet_to_act = sum(
        1 for i in opponents
        if hand.stacks[i] > 0 and (not hand.acted[i] or hand.bets[i] < hand.max_bet)
    )
    order = [i for i in postflop_order(n) if not hand.folded[i]]
    callers = 0
    raises = 0
    aggressor = None

    for record in this_street:
        if record.aggressive:
            raises += 1
            callers = 0
            aggressor = record.player
        elif record.added > 0:
            callers += 1

    prev_aggressor = next((r.player for r in reversed(earlier) if r.aggressive), None)
    own = [r for r in this_street if r.player == player]
    own_prev = 'none'

    if own:
        last = own[-1]
        own_prev = 'raise' if last.aggressive and last.kind == 'all-in' else last.kind

        if own_prev == 'all-in':
            own_prev = 'call'

    opponent = main_opponent(hand, player)
    stacks_in = [hand.stacks[i] for i in opponents]
    summary = opp_summary or {}
    known = [v for v in (table_vpips or []) if v is not None]

    return {
        'street_idx': street,
        'position': POSITION_NAMES[n][player],
        'num_players': n,
        'players_in_hand': len(in_hand),
        'players_yet_to_act': yet_to_act,
        'in_position': int(bool(order) and order[-1] == player),
        'pot_bb': hand.pot / BB,
        'to_call_bb': hand.to_call / BB,
        'stack_bb': hand.stacks[player] / BB,
        'effective_stack_bb': min(hand.stacks[player], max(stacks_in, default=0)) / BB,
        'raises_this_street': raises,
        'callers_this_street': callers,
        'is_street_aggressor': int(aggressor == player),
        'was_prev_street_aggressor': int(prev_aggressor == player),
        'own_prev_action_street': own_prev,
        'last_opp_action': this_street[-1].kind if this_street else 'none',
        'opp_stack_bb': hand.stacks[opponent] / BB if opponent is not None else 0.0,
        'opp_hands_seen': summary.get('hands_seen', 0),
        'opp_vpip': summary.get('vpip'),
        'opp_pfr': summary.get('pfr'),
        'opp_aggression_factor': summary.get('aggression_factor'),
        'opp_fold_to_bet': summary.get('fold_to_bet'),
        'table_avg_vpip': sum(known) / len(known) if known else None,
        'main_opponent': opponent,  # an index, not encoded
    }


# -------------------------------------------------------------- encoding

def _one_hot(value, choices) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _log_scale(value: float, top: float) -> float:
    return math.log1p(max(value, 0.0)) / math.log1p(top)


def _position_groups(name: str) -> list[float]:
    groups = set()

    for part in name.split('/'):
        groups.add('EARLY' if part.startswith('UTG') else part)

    return [1.0 if g in groups else 0.0 for g in POSITION_GROUPS]


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def encode(context: dict, cards: dict, equity: float) -> np.ndarray:
    """Context + ``cards.hand_features`` output + equity -> state vector."""

    pot = context['pot_bb']
    to_call = context['to_call_bb']
    facing = to_call > 0
    pot_odds = to_call / (pot + to_call) if facing else 0.0
    bet_frac = to_call / pot if pot > 0 else 0.0
    spr = context['effective_stack_bb'] / pot if pot > 0 else 0.0

    v = []
    v += _one_hot(context['street_idx'], range(4))
    v += _position_groups(context['position'])
    v += [
        context['num_players'] / 9,
        context['players_in_hand'] / 9,
        min(context['players_yet_to_act'], 8) / 8,
        float(context['in_position']),
        _log_scale(pot, 500),
        _log_scale(to_call, 300),
        _log_scale(context['stack_bb'], 300),
        _log_scale(context['effective_stack_bb'], 300),
        pot_odds,
        float(facing),
        min(bet_frac, 3.0) / 3,
        _log_scale(spr, 100),
        min(context['raises_this_street'], 4) / 4,
        min(context['callers_this_street'], 4) / 4,
        float(context['is_street_aggressor']),
        float(context['was_prev_street_aggressor']),
    ]
    v += _one_hot(context['own_prev_action_street'], OWN_PREV)
    v += _one_hot(context['last_opp_action'], LAST_ACTION)

    v += [
        cards['card_high_rank'] / 14,
        cards['card_low_rank'] / 14,
        float(cards['is_pocket_pair']),
        float(cards['is_suited']),
        cards['rank_gap'] / 12,
        (cards['chen_score'] + 1.5) / 21.5,
        cards['preflop_percentile'],
    ]
    v += _one_hot(cards['made_hand_rank'], range(9))
    v += _one_hot(cards['pair_type'], PAIR_TYPES)
    v += [
        float(cards['flush_draw']),
        float(cards['straight_draw'] == 'gutshot'),
        float(cards['straight_draw'] == 'open-ended'),
        min(cards['draw_outs'], 15) / 15,
        float(cards['board_paired']),
    ]
    v += _one_hot(cards['board_suit_texture'], TEXTURES)
    v += [
        cards['board_connectedness'] / 5,
        cards['board_high_rank'] / 14,
        equity,
    ]

    for key in OPP_STATS:
        value = context['opp_' + key]

        if _missing(value):
            v += [0.0, 1.0]
        elif key == 'aggression_factor':
            v += [min(float(value), 10.0) / 10, 0.0]
        else:
            v += [float(value), 0.0]

    table_vpip = context['table_avg_vpip']
    v += [0.0, 1.0] if _missing(table_vpip) else [float(table_vpip), 0.0]
    v += [
        _log_scale(float(context['opp_hands_seen'] or 0), 1000),
        _log_scale(context['opp_stack_bb'], 300),
    ]
    return np.asarray(v, dtype=np.float32)


STATE_DIM = len(encode(
    {
        'street_idx': 0, 'position': 'BB', 'num_players': 6, 'players_in_hand': 6,
        'players_yet_to_act': 0, 'in_position': 0, 'pot_bb': 1.5, 'to_call_bb': 0.0,
        'stack_bb': 100.0, 'effective_stack_bb': 100.0, 'raises_this_street': 0,
        'callers_this_street': 0, 'is_street_aggressor': 0,
        'was_prev_street_aggressor': 0, 'own_prev_action_street': 'none',
        'last_opp_action': 'none', 'opp_stack_bb': 100.0, 'opp_hands_seen': 0,
        'opp_vpip': None, 'opp_pfr': None, 'opp_aggression_factor': None,
        'opp_fold_to_bet': None, 'table_avg_vpip': None,
    },
    hand_features([48, 44], []),
    0.5,
))


def player_equity(hole: list[int], board: list[int], players_in_hand: int,
                  rng: random.Random, samples: int = 200) -> float:
    return equity_vs_random(hole, board, players_in_hand - 1, rng, samples)
