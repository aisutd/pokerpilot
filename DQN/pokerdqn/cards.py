"""Cards, hand evaluation and hand-strength features.

Cards are ints 0..51: ``rank * 4 + suit`` with rank 0 = deuce .. 12 = ace
and suit 0..3 = c, d, h, s. The evaluator scores 5-7 cards as one int
(higher is better): ``category << 20 | five ranks packed 4 bits each``,
category 0 = high card .. 8 = straight flush.

Every feature here is computed from the acting player's hole cards and the
visible board only, so it is identical for dataset rows and simulator states.
"""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

RANKS = '23456789TJQKA'
SUITS = 'cdhs'

CATEGORY_NAMES = (
    'High Card', 'Pair', 'Two Pair', 'Three of a Kind', 'Straight',
    'Flush', 'Full House', 'Four of a Kind', 'Straight Flush',
)

PAIR_TYPES = (
    'none', 'pocket pair', 'overpair', 'underpair', 'top pair',
    'middle pair', 'bottom pair', 'board pair only', 'two pair or better',
)


def parse_card(text: str) -> int:
    text = text.strip()
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


def parse_cards(text: str | None) -> list[int]:
    """'As Kd' or 'AsKd' -> [51, 45]. Empty / None -> []."""

    if not text or not isinstance(text, str):
        return []

    text = text.replace(' ', '').replace(',', '')
    return [parse_card(text[i:i + 2]) for i in range(0, len(text), 2)]


def card_str(card: int) -> str:
    return RANKS[card >> 2] + SUITS[card & 3]


def rank_of(card: int) -> int:
    """2..14 (ace = 14), the convention used by the dataset."""

    return (card >> 2) + 2


# ------------------------------------------------------------ evaluator

def _build_straight_table() -> list[int]:
    """rank bitmask -> (highest straight's top rank index + 1), 0 if none."""

    table = [0] * 8192
    windows = [(0b11111 << low, low + 4) for low in range(8, -1, -1)]
    wheel = (1 << 12) | 0b1111  # A-2-3-4-5

    for mask in range(8192):
        for window, top in windows:
            if mask & window == window:
                table[mask] = top + 1
                break
        else:
            if mask & wheel == wheel:
                table[mask] = 3 + 1  # five-high

    return table


STRAIGHT_TOP = _build_straight_table()


def _pack(ranks) -> int:
    value = 0
    count = 0

    for rank in ranks:
        value = (value << 4) | rank
        count += 1

    return value << (4 * (5 - count))


def _top_ranks(mask: int, count: int) -> list[int]:
    out = []
    rank = 12

    while rank >= 0 and len(out) < count:
        if mask >> rank & 1:
            out.append(rank)
        rank -= 1

    return out


def evaluate(cards) -> int:
    """Score the best five-card hand among 5-7 cards (higher wins)."""

    counts = [0] * 13
    suit_masks = [0, 0, 0, 0]
    suit_counts = [0, 0, 0, 0]
    mask = 0

    for card in cards:
        rank = card >> 2
        suit = card & 3
        counts[rank] += 1
        suit_masks[suit] |= 1 << rank
        suit_counts[suit] += 1
        mask |= 1 << rank

    # With at most 7 cards a flush excludes quads and full houses.
    for suit in range(4):
        if suit_counts[suit] >= 5:
            flush_mask = suit_masks[suit]
            top = STRAIGHT_TOP[flush_mask]

            if top:
                return (8 << 20) | ((top - 1) << 16)

            return (5 << 20) | _pack(_top_ranks(flush_mask, 5))

    quads = []
    trips = []
    pairs = []

    for rank in range(12, -1, -1):
        count = counts[rank]

        if count == 4:
            quads.append(rank)
        elif count == 3:
            trips.append(rank)
        elif count == 2:
            pairs.append(rank)

    if quads:
        quad = quads[0]
        kicker = _top_ranks(mask & ~(1 << quad), 1)
        return (7 << 20) | _pack([quad] + kicker)

    if trips and (len(trips) > 1 or pairs):
        second = max(trips[1] if len(trips) > 1 else -1, pairs[0] if pairs else -1)
        return (6 << 20) | _pack([trips[0], second])

    top = STRAIGHT_TOP[mask]

    if top:
        return (4 << 20) | ((top - 1) << 16)

    if trips:
        kickers = _top_ranks(mask & ~(1 << trips[0]), 2)
        return (3 << 20) | _pack([trips[0]] + kickers)

    if len(pairs) >= 2:
        high, low = pairs[0], pairs[1]
        kicker = _top_ranks(mask & ~(1 << high) & ~(1 << low), 1)
        return (2 << 20) | _pack([high, low] + kicker)

    if pairs:
        kickers = _top_ranks(mask & ~(1 << pairs[0]), 3)
        return (1 << 20) | _pack([pairs[0]] + kickers)

    return _pack(_top_ranks(mask, 5))


def category(score: int) -> int:
    return score >> 20


# ------------------------------------------------------- preflop strength

def chen_score(first: int, second: int) -> float:
    """Bill Chen's preflop formula (no rounding), as in the dataset."""

    points = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}
    high = max(rank_of(first), rank_of(second))
    low = min(rank_of(first), rank_of(second))
    score = points.get(high, high / 2)

    if high == low:
        return max(5.0, score * 2)

    if first & 3 == second & 3:
        score += 2

    gap = high - low - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)

    if gap <= 1 and high < 12:
        score += 1

    return score


def _build_preflop_percentile() -> dict[tuple[int, int, bool], float]:
    """(high rank, low rank, suited) -> mid-rank percentile of the Chen score
    among the 1,326 starting hands (ties split evenly), as in the dataset."""

    scores = [chen_score(a, b) for a, b in combinations(range(52), 2)]
    total = len(scores)
    table = {}

    for a, b in combinations(range(52), 2):
        key = hand_class(a, b)

        if key not in table:
            score = chen_score(a, b)
            below = sum(1 for s in scores if s < score)
            at_most = sum(1 for s in scores if s <= score)
            table[key] = (below + at_most) / 2 / total

    return table


def hand_class(first: int, second: int) -> tuple[int, int, bool]:
    """One of the 169 starting-hand classes: (high, low, suited)."""

    high = max(rank_of(first), rank_of(second))
    low = min(rank_of(first), rank_of(second))
    return high, low, high != low and first & 3 == second & 3


PREFLOP_PERCENTILE = _build_preflop_percentile()


# ------------------------------------------------------- postflop features

def _straight_draw(hole: list[int], board: list[int]) -> tuple[str, int]:
    """'open-ended' / 'gutshot' / 'none' and straight outs, for draws that
    use at least one hole card. Made straights are not draws."""

    cards = hole + board
    mask = 0

    for card in cards:
        mask |= 1 << (card >> 2)

    if STRAIGHT_TOP[mask]:
        return 'none', 0

    board_mask = 0

    for card in board:
        board_mask |= 1 << (card >> 2)

    completing = []

    for rank in range(13):
        if mask >> rank & 1:
            continue

        new = mask | (1 << rank)

        # The straight must need a hole card to count as the player's draw.
        if STRAIGHT_TOP[new] and not STRAIGHT_TOP[board_mask | (1 << rank)]:
            completing.append(rank)

    if len(completing) >= 2:
        return 'open-ended', 8

    if len(completing) == 1:
        return 'gutshot', 4

    return 'none', 0


def _pair_type(hole: list[int], board: list[int], made: int) -> str:
    if not board:
        return 'pocket pair' if hole[0] >> 2 == hole[1] >> 2 else 'none'

    if made >= 2:
        return 'two pair or better'

    if made == 0:
        return 'none'

    board_ranks = sorted({card >> 2 for card in board}, reverse=True)
    h1, h2 = hole[0] >> 2, hole[1] >> 2

    if h1 == h2:
        if h1 > board_ranks[0]:
            return 'overpair'
        return 'underpair'

    for rank in (h1, h2):
        if rank in board_ranks:
            if rank == board_ranks[0]:
                return 'top pair'
            if rank == board_ranks[-1]:
                return 'bottom pair'
            return 'middle pair'

    return 'board pair only'


def hand_features(hole: list[int], board: list[int]) -> dict:
    """Everything the state encoder needs about the cards."""

    h1, h2 = hole
    high = max(rank_of(h1), rank_of(h2))
    low = min(rank_of(h1), rank_of(h2))
    suited = h1 & 3 == h2 & 3
    made = category(evaluate(hole + board)) if len(board) >= 3 else (1 if high == low else 0)

    feats = {
        'card_high_rank': high,
        'card_low_rank': low,
        'is_pocket_pair': int(high == low),
        'is_suited': int(suited),
        'rank_gap': high - low,
        'chen_score': chen_score(h1, h2),
        'preflop_percentile': PREFLOP_PERCENTILE[hand_class(h1, h2)],
        'made_hand_rank': made,
        'pair_type': _pair_type(hole, board, made),
        'n_board': len(board),
        'flush_draw': 0,
        'straight_draw': 'none',
        'draw_outs': 0,
        'board_paired': 0,
        'board_suit_texture': 'none',
        'board_connectedness': 0,
        'board_high_rank': 0,
    }

    if not board:
        return feats

    suit_counts = [0, 0, 0, 0]

    for card in board:
        suit_counts[card & 3] += 1

    outs = 0

    if len(board) < 5:
        if made < 5:
            for suit in range(4):
                mine = sum(1 for card in hole if card & 3 == suit)

                if mine and suit_counts[suit] + mine == 4:
                    feats['flush_draw'] = 1
                    outs += 9

        draw, straight_outs = _straight_draw(hole, board)
        feats['straight_draw'] = draw

        if made < 4:
            # Two of the straight outs also complete the flush.
            outs += straight_outs - 2 if outs and straight_outs else straight_outs

    feats['draw_outs'] = outs
    ranks = [card >> 2 for card in board]
    feats['board_paired'] = int(len(set(ranks)) < len(ranks))
    max_suit = max(suit_counts)
    feats['board_suit_texture'] = (
        'monotone' if max_suit >= 3 else 'two-tone' if max_suit == 2 else 'rainbow'
    )
    unique = set(ranks)

    if 12 in unique:
        unique_low = unique | {-1}  # ace plays low too
    else:
        unique_low = unique

    feats['board_connectedness'] = max(
        sum(1 for r in unique_low if low_r <= r < low_r + 5) for low_r in range(-1, 9)
    )
    feats['board_high_rank'] = max(ranks) + 2
    return feats


# ----------------------------------------------------------------- equity

def equity(hole: list[int], board: list[int], opponents: int, samples: int,
           rng: random.Random) -> float:
    """Monte Carlo share of the pot won against ``opponents`` random hands."""

    if opponents <= 0:
        return 1.0

    known = set(hole) | set(board)
    deck = [card for card in range(52) if card not in known]
    need_board = 5 - len(board)
    draw = need_board + 2 * opponents
    total = 0.0

    for _ in range(samples):
        picked = rng.sample(deck, draw)
        full_board = board + picked[:need_board]
        mine = evaluate(hole + full_board)
        best = -1
        ties = 0
        pos = need_board

        for _ in range(opponents):
            score = evaluate([picked[pos], picked[pos + 1]] + full_board)
            pos += 2

            if score > best:
                best, ties = score, 1
            elif score == best:
                ties += 1

        if mine > best:
            total += 1.0
        elif mine == best:
            total += 1.0 / (ties + 1)

    return total / samples


PREFLOP_SAMPLES = 1500
_PREFLOP_CACHE: dict[tuple[int, int, bool, int], float] = {}


def preflop_equity(high: int, low: int, suited: bool, opponents: int) -> float:
    """Deterministic preflop equity of a starting-hand class (cached; see
    ``save_preflop_table`` / ``load_preflop_table``)."""

    key = (high, low, suited, opponents)

    if key not in _PREFLOP_CACHE:
        first = (high - 2) * 4
        second = (low - 2) * 4 + (0 if suited else 1)
        seed = (high * 100 + low) * 20 + opponents * 2 + int(suited)
        _PREFLOP_CACHE[key] = equity(
            [first, second], [], opponents, PREFLOP_SAMPLES, random.Random(seed),
        )

    return _PREFLOP_CACHE[key]


def build_preflop_table(max_opponents: int = 8) -> None:
    for high, low, suited in sorted(PREFLOP_PERCENTILE):
        for opponents in range(1, max_opponents + 1):
            preflop_equity(high, low, suited, opponents)


def save_preflop_table(path) -> None:
    rows = [[h, l, int(s), o, v] for (h, l, s, o), v in sorted(_PREFLOP_CACHE.items())]
    Path(path).write_text(json.dumps({'samples': PREFLOP_SAMPLES, 'rows': rows}))


def load_preflop_table(path) -> bool:
    path = Path(path)

    if not path.exists():
        return False

    data = json.loads(path.read_text())

    if data.get('samples') != PREFLOP_SAMPLES:
        return False

    for h, l, s, o, v in data['rows']:
        _PREFLOP_CACHE[(h, l, bool(s), o)] = v

    return True


def equity_vs_random(hole: list[int], board: list[int], opponents: int,
                     rng: random.Random, samples: int = 200) -> float:
    if not board:
        return preflop_equity(*hand_class(*hole), min(max(opponents, 1), 8))

    return equity(hole, board, max(opponents, 1), samples, rng)
