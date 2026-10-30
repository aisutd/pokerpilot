"""The fast engine must agree with pokerkit (the website's rules engine)."""

import random
import unittest
import warnings
from collections import deque

from pokerkit import Automation, Card, Deck, Mode, NoLimitTexasHoldem

from pokerdqn.cards import parse_card
from pokerdqn.engine import Hand, IllegalAction

AUTOMATIONS = tuple(
    a for a in Automation
    if a not in (Automation.HOLE_DEALING, Automation.HOLE_CARDS_SHOWING_OR_MUCKING)
)


def to_int(card: Card) -> int:
    return parse_card(repr(card))


def pk_index(i: int, n: int) -> int:
    """Our index -> pokerkit index (pokerkit puts the heads-up BB first)."""

    return 1 - i if n == 2 else i


def make_pair(stacks, seed):
    n = len(stacks)
    pk_stacks = [0] * n

    for i, stack in enumerate(stacks):
        pk_stacks[pk_index(i, n)] = stack

    state = NoLimitTexasHoldem.create_state(
        AUTOMATIONS, True, 0, (50, 100), 100, tuple(pk_stacks), n, mode=Mode.CASH_GAME,
    )
    deck = list(Deck.STANDARD)
    random.Random(seed).shuffle(deck)
    state.deck_cards = deque(deck)

    while state.can_deal_hole():
        state.deal_hole()

    rest = list(state.deck_cards)
    # pokerkit burns one card before each street.
    runout = [to_int(c) for c in (rest[1], rest[2], rest[3], rest[5], rest[7])]
    ours = [0] * (2 * n)

    for i in range(n):
        first, second = state.hole_cards[pk_index(i, n)]
        ours[i], ours[n + i] = to_int(first), to_int(second)

    return state, Hand(list(stacks), ours + runout)


class EngineMatchesPokerkit(unittest.TestCase):
    def check_same(self, state, hand):
        n = hand.n

        # Split pots: pokerkit may hand an odd chip to another winner.
        slack = 1 if hand.finished and len(hand.winners) > 1 else 0

        for i in range(n):
            self.assertLessEqual(abs(hand.stacks[i] - state.stacks[pk_index(i, n)]), slack)

        if hand.actor is None:
            self.assertTrue(hand.finished)
            return

        self.assertEqual(pk_index(hand.actor, n), state.actor_index)
        self.assertEqual(min(hand.to_call, hand.stacks[hand.actor]),
                         state.checking_or_calling_amount)
        bounds = hand.raise_range()

        if state.can_complete_bet_or_raise_to():
            self.assertIsNotNone(bounds)
            self.assertEqual(bounds, (
                state.min_completion_betting_or_raising_to_amount,
                state.max_completion_betting_or_raising_to_amount,
            ))
        else:
            self.assertIsNone(bounds)

    def play(self, stacks, seed):
        rng = random.Random(seed)
        state, hand = make_pair(stacks, seed)
        self.check_same(state, hand)

        while not hand.finished:
            choices = ['call']

            if hand.can_fold():
                choices.append('fold')

            bounds = hand.raise_range()

            if bounds:
                choices += ['min', 'any', 'max']

            choice = rng.choice(choices)

            if choice == 'fold':
                hand.fold()
                state.fold()
            elif choice == 'call':
                hand.check_or_call()
                state.check_or_call()
            else:
                low, high = bounds
                amount = {'min': low, 'max': high}.get(choice) or rng.randint(low, high)
                hand.raise_to(amount)
                state.complete_bet_or_raise_to(amount)

            while state.status and state.actor_index is None and state.can_show_or_muck_hole_cards():
                state.show_or_muck_hole_cards(True)

            self.check_same(state, hand)

        self.assertFalse(state.status)
        n = hand.n
        theirs = [state.payoffs[pk_index(i, n)] for i in range(n)]
        self.assertEqual(sum(hand.payoffs), 0)

        if len(hand.winners) > 1:
            for ours, pk in zip(hand.payoffs, theirs):
                self.assertLessEqual(abs(ours - pk), 1)
        else:
            self.assertEqual(hand.payoffs, theirs)

        if hand.showdown:
            self.assertEqual(hand.board, [to_int(c) for c in state.get_board_cards(0)])

    def test_random_hands(self):
        rng = random.Random(7)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            for seed in range(1500):
                n = rng.randint(2, 9)
                # Mix deep and short stacks so all-ins and side pots happen.
                stacks = [rng.choice([rng.randint(100, 1500), rng.randint(1000, 25000)])
                          for _ in range(n)]

                with self.subTest(seed=seed, stacks=stacks):
                    self.play(stacks, seed)


class EngineRules(unittest.TestCase):
    def test_cannot_fold_when_checking_is_free(self):
        hand = Hand([10000, 10000], list(range(9)))
        hand.check_or_call()  # small blind completes

        with self.assertRaises(IllegalAction):
            hand.fold()  # big blind may check

    def test_uncalled_bet_is_returned(self):
        hand = Hand([10000, 10000, 10000], list(range(11)))
        hand.raise_to(10000)  # button shoves
        hand.fold()
        hand.fold()
        self.assertEqual(hand.payoffs, [-50, -100, 150])

    def test_side_pot(self):
        # Deck: hole cards for 3 players, then the board.
        aces = [parse_card(c) for c in ('As', 'Ks', 'Qs', 'Ah', 'Kh', 'Qh')]
        board = [parse_card(c) for c in ('2c', '7d', '9h', 'Jc', '3s')]
        hand = Hand([1000, 5000, 5000], aces + board)
        hand.raise_to(5000)  # button (Qs Qh) all-in
        self.assertIsNone(hand.raise_range())  # too short to raise
        hand.check_or_call()  # small blind (As Ah) calls all-in for less
        hand.check_or_call()  # big blind (Ks Kh) calls
        self.assertTrue(hand.finished)
        # Aces win the 3000 main pot; queens lose 5000, kings win the side pot.
        self.assertEqual(hand.payoffs, [2000, 3000, -5000])


if __name__ == '__main__':
    unittest.main()
