import random
import unittest

from pokerkit import Card, StandardHighHand

from pokerdqn.cards import (PREFLOP_PERCENTILE, card_str, category, chen_score, equity,
                            evaluate, hand_class, hand_features, parse_cards)
from pokerdqn.dataset import find_dataset_dir, load_tables


def pk_hand(cards):
    return StandardHighHand.from_game(Card.parse(''.join(card_str(c) for c in cards)), ())


class EvaluatorMatchesPokerkit(unittest.TestCase):
    def test_random_pairs_of_hands_compare_like_pokerkit(self):
        rng = random.Random(3)

        for _ in range(3000):
            cards = rng.sample(range(52), 9)
            board = cards[4:]
            a, b = cards[:2] + board, cards[2:4] + board
            ours = (evaluate(a) > evaluate(b)) - (evaluate(a) < evaluate(b))
            pa, pb = pk_hand(a), pk_hand(b)
            theirs = (pa > pb) - (pa < pb)
            self.assertEqual(ours, theirs, [card_str(c) for c in cards])

    def test_categories(self):
        cases = {
            'AsKsQsJsTs': 8, '5c4c3c2cAc': 8, '9h9d9c9s2d': 7, 'KhKdKc2s2d': 6,
            'Ah9h7h3h2h': 5, '5d4c3h2sAd': 4, '7h7d7c2sKd': 3, '7h7d2c2sKd': 2,
            '7h7d3c2sKd': 1, 'Ah9d7c3s2h': 0,
        }

        for text, expected in cases.items():
            self.assertEqual(category(evaluate(parse_cards(text))), expected, text)

    def test_wheel_loses_to_six_high_straight(self):
        self.assertLess(evaluate(parse_cards('5d4c3h2sAd')), evaluate(parse_cards('6d5c4h3s2d')))

    def test_seven_cards_pick_best_five(self):
        # Board pairs twice: the third pair is only a kicker candidate.
        two_pair = parse_cards('AhAd KcKs 3h3d Qc')
        self.assertEqual(category(evaluate(two_pair)), 2)
        self.assertGreater(evaluate(two_pair), evaluate(parse_cards('AhAd KcKs 3h3d Jc')))


class FeaturesMatchDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decisions = load_tables(find_dataset_dir()).decisions

    def test_card_features_agree_with_dataset_columns(self):
        columns = ('card_high_rank', 'card_low_rank', 'is_pocket_pair', 'is_suited',
                   'rank_gap', 'chen_score', 'preflop_percentile', 'made_hand_rank',
                   'pair_type', 'flush_draw', 'straight_draw', 'draw_outs')
        disagreements = 0

        for _, row in self.decisions.iterrows():
            board = row['board'] if isinstance(row['board'], str) else ''
            ours = hand_features(parse_cards(row['hole_cards']), parse_cards(board))

            for column in columns:
                theirs = row[column]

                if isinstance(theirs, str):
                    disagreements += theirs != ours[column]
                else:
                    disagreements += abs(float(theirs) - float(ours[column])) > 0.002

        # A handful of draws on boards that already make a straight/flush are
        # counted differently (we don't count draws to a hand already made).
        self.assertLess(disagreements, 20)

    def test_chen_and_percentile_examples(self):
        self.assertEqual(chen_score(*parse_cards('AsAh')), 20)
        self.assertEqual(chen_score(*parse_cards('5c4d')), 3.5)
        self.assertEqual(chen_score(*parse_cards('9c2h')), -0.5)
        self.assertAlmostEqual(PREFLOP_PERCENTILE[hand_class(*parse_cards('Jd7s'))], 0.2805, places=4)
        self.assertEqual(len(PREFLOP_PERCENTILE), 169)


class Equity(unittest.TestCase):
    def test_known_preflop_matchups(self):
        rng = random.Random(0)
        aces = equity(parse_cards('AsAh'), [], 1, 4000, rng)
        self.assertAlmostEqual(aces, 0.85, delta=0.02)
        trash = equity(parse_cards('7c2d'), [], 1, 4000, rng)
        self.assertAlmostEqual(trash, 0.35, delta=0.03)

    def test_nuts_on_river(self):
        rng = random.Random(0)
        self.assertEqual(equity(parse_cards('AsKs'), parse_cards('QsJsTs2d3c'), 3, 200, rng), 1.0)


if __name__ == '__main__':
    unittest.main()
