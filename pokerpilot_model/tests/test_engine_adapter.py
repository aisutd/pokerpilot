import random

from pokerkit import Deck

from pokerpilot_model.engine_adapter import deterministic_equity


def test_equity_sampling_is_repeatable_for_the_same_seed():
    hole = list(Deck.STANDARD[:2])
    board = list(Deck.STANDARD[2:5])
    first = deterministic_equity(hole, board, 2, 20, random.Random(41))
    second = deterministic_equity(hole, board, 2, 20, random.Random(41))
    assert first == second
