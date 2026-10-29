"""The discrete action space shared by the DQN, the baselines and the data.

    0 FOLD        only when facing a bet (folding when checking is free is
                  never better, and pokerkit warns about it)
    1 CHECK       only when nothing is owed
    2 CALL        only when facing a bet (calling a bigger bet than the stack
                  is an all-in call)
    3 RAISE_HALF  bet / raise by half the pot
    4 RAISE_POT   bet / raise by the full pot
    5 ALL_IN      bet / raise all chips

Raise sizes follow the usual pot-limit convention: after calling, add
``fraction * (pot + to_call)``, where ``pot`` already includes this street's
bets. The result is clamped to the legal minimum raise. A size that would
use the whole stack is the same action as ALL_IN, so it is masked out.
The dataset's bet sizes measured this way sit between 0.34 and 1.18 (see
``README.md``), hence the half-pot / pot buckets split at 0.75.
"""

from __future__ import annotations

import numpy as np

FOLD, CHECK, CALL, RAISE_HALF, RAISE_POT, ALL_IN = range(6)
NUM_ACTIONS = 6
ACTION_NAMES = ('fold', 'check', 'call', 'raise_half', 'raise_pot', 'all_in')
RAISE_FRACTIONS = {RAISE_HALF: 0.5, RAISE_POT: 1.0}
SIZE_SPLIT = 0.75  # dataset raises below this fraction map to RAISE_HALF


def raise_target(my_bet: int, to_call: int, pot: int, fraction: float) -> int:
    """Street total for a ``fraction``-of-pot bet or raise."""

    return int(round(my_bet + to_call + fraction * (pot + to_call)))


def legal_mask(to_call: int, my_bet: int, pot: int,
               raise_range: tuple[int, int] | None) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)

    if to_call > 0:
        mask[FOLD] = True
        mask[CALL] = True
    else:
        mask[CHECK] = True

    if raise_range is not None:
        low, high = raise_range
        mask[ALL_IN] = True
        previous = None

        for action in (RAISE_HALF, RAISE_POT):
            amount = max(low, raise_target(my_bet, to_call, pot, RAISE_FRACTIONS[action]))

            if amount < high and amount != previous:
                mask[action] = True
                previous = amount

    return mask


def chips_for(action: int, to_call: int, my_bet: int, pot: int,
              raise_range: tuple[int, int] | None) -> int | None:
    """Street total to raise to for a raise action, None for fold/check/call."""

    if action in (FOLD, CHECK, CALL):
        return None

    low, high = raise_range

    if action == ALL_IN:
        return high

    amount = raise_target(my_bet, to_call, pot, RAISE_FRACTIONS[action])
    return min(high, max(low, amount))


def dataset_action(kind: str, chips_added: float, to_call: float, pot: float,
                   stack: float) -> int:
    """Map a logged dataset action (bb amounts) onto the discrete space."""

    if kind == 'fold':
        return FOLD

    if kind == 'check':
        return CHECK

    if kind == 'call':
        return CALL

    if kind == 'all-in' or chips_added >= stack - 1e-6:
        # An all-in that doesn't exceed the amount owed is just a call.
        return CALL if stack <= to_call + 1e-6 else ALL_IN

    if kind in ('bet', 'raise'):
        fraction = (chips_added - to_call) / (pot + to_call)
        return RAISE_HALF if fraction < SIZE_SPLIT else RAISE_POT

    raise ValueError(f'unknown action {kind!r}')


def dataset_legal_mask(to_call: float, pot: float, stack: float,
                       min_raise_to: float | None = None, my_bet: float = 0.0,
                       raise_allowed: bool = True) -> np.ndarray:
    """Legal actions for a logged decision, in bb.

    ``min_raise_to`` / ``raise_allowed`` come from replaying the hand
    (see ``dataset.replay_game``); without them raising is assumed legal
    whenever the player has chips beyond the call.
    """

    scale = 100  # work in chips so the rounding matches the simulator
    to_call_c = int(round(to_call * scale))
    pot_c = int(round(pot * scale))
    my_bet_c = int(round(my_bet * scale))
    top = my_bet_c + int(round(stack * scale))
    raise_range = None

    if raise_allowed and top > my_bet_c + to_call_c:
        low = my_bet_c + to_call_c + 100 if min_raise_to is None else int(round(min_raise_to * scale))
        raise_range = (min(low, top), top)

    return legal_mask(to_call_c, my_bet_c, pot_c, raise_range)
