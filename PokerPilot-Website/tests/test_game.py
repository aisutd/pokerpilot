import random
import unittest

from engine.bots import DIFFICULTIES
from engine.game import HUMAN_SEAT, SEAT_COUNT, STARTING_STACK, GameError, Table

TOTAL_CHIPS = SEAT_COUNT * STARTING_STACK


def random_human_move(table: Table, rng: random.Random) -> None:
    legal = table.legal_actions()
    options = ['check_call', 'check_call']

    if legal['fold']:
        options.append('fold')

    if legal['raise']:
        options.append('raise')

    action = rng.choice(options)
    amount = None

    if action == 'raise':
        low, high = legal['raise']['min'], legal['raise']['max']
        amount = rng.choice([low, high, rng.randint(low, high)])

    table.human_action(action, amount)


def check_invariants(test: unittest.TestCase, table: Table) -> None:
    view = table.view()
    chips = sum(seat['stack'] for seat in view['seats']) + view['pot']
    test.assertEqual(chips, TOTAL_CHIPS)

    for i, seat in enumerate(view['seats']):
        test.assertGreaterEqual(seat['stack'], 0)

        if i != HUMAN_SEAT and not view['hand_over']:
            test.assertIsNone(seat['cards'], 'AI cards leaked mid-hand')

    if not view['hand_over']:
        test.assertEqual(view['to_act'], HUMAN_SEAT)
        test.assertIsNotNone(view['legal'])


class TableTest(unittest.TestCase):
    def play(self, difficulty: str, seed: int, max_hands: int) -> Table:
        rng = random.Random(seed)
        table = Table(difficulty, seed=seed)

        for _ in range(max_hands):
            while not table.hand_over:
                check_invariants(self, table)
                random_human_move(table, rng)

            check_invariants(self, table)

            for frame in table.frames:
                chips = sum(s['stack'] for s in frame['seats']) + frame['pot']
                self.assertEqual(chips, TOTAL_CHIPS)

            if table.game_over:
                break

            table.next_hand()

        return table

    def test_all_difficulties(self) -> None:
        for difficulty in DIFFICULTIES:
            for seed in range(2):
                with self.subTest(difficulty=difficulty, seed=seed):
                    self.play(difficulty, seed, max_hands=15)

    def test_game_runs_until_someone_wins_or_human_busts(self) -> None:
        table = Table('easy', seed=11)

        while not table.game_over:
            while not table.hand_over:
                legal = table.legal_actions()

                if legal['raise']:
                    table.human_action('raise', legal['raise']['max'])
                else:
                    table.human_action('check_call')

            if not table.game_over:
                table.next_hand()

        total = sum(seat.stack for seat in table.seats)
        self.assertEqual(total, TOTAL_CHIPS)

        with self.assertRaises(GameError):
            table.next_hand()

    def test_busted_players_sit_out(self) -> None:
        table = Table('medium', seed=4)

        while table.hand_over is False:
            table.human_action('fold' if table.legal_actions()['fold'] else 'check_call')

        table.seats[3].stack, table.seats[1].stack = 0, table.seats[1].stack + table.seats[3].stack
        table.next_hand()
        self.assertNotIn(3, table.order)
        self.assertEqual(table.view()['seats'][3]['status'], 'out')

    def test_illegal_requests_are_rejected(self) -> None:
        table = Table('hard', seed=5)

        while table.hand_over:
            table.next_hand()

        legal = table.legal_actions()

        if legal['raise']:
            with self.assertRaises(GameError):
                table.human_action('raise', legal['raise']['max'] + 1)

        with self.assertRaises(GameError):
            table.human_action('dance')

        with self.assertRaises(GameError):
            table.next_hand()

        if legal['check']:
            with self.assertRaises(GameError):
                table.human_action('fold')

    def test_button_moves_clockwise(self) -> None:
        table = Table('easy', seed=3)
        first = table.button

        while not table.hand_over:
            table.human_action('fold' if table.legal_actions()['fold'] else 'check_call')

        table.next_hand()
        self.assertEqual(table.button, table.next_active(first))
        self.assertEqual(table.order[-1], table.button)


if __name__ == '__main__':
    unittest.main()
