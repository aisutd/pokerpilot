"""AI opponents for a 6-max No-Limit Texas Hold'em table.

A bot sees only its own hole cards, the board and public betting actions.
Hand strength comes from pokerkit (``calculate_hand_strength`` and
``StandardHighHand``).

Difficulty sets a bot's *skill* (how well it reads hand strength, pot odds
and opponents). Each seat also has a *personality* (looser, more aggressive,
bluffier...) so the five AI players feel different at every level.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations

from pokerkit import Card, Deck, StandardHighHand, calculate_hand_strength

DIFFICULTIES = ('easy', 'medium', 'hard')

RANK_ORDER = '23456789TJQKA'


# --------------------------------------------------------------- hand ranking

def chen_score(first: Card, second: Card) -> float:
    """Bill Chen's preflop hand-strength formula."""

    values = {'A': 10, 'K': 8, 'Q': 7, 'J': 6}
    high, low = sorted(
        (first, second), key=lambda c: RANK_ORDER.index(c.rank.value), reverse=True,
    )
    high_index = RANK_ORDER.index(high.rank.value)
    low_index = RANK_ORDER.index(low.rank.value)
    score = values.get(high.rank.value, (high_index + 2) / 2)

    if high_index == low_index:
        return max(score * 2, 5)

    if high.suit == low.suit:
        score += 2

    gap = high_index - low_index - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)

    if gap <= 1 and high_index < RANK_ORDER.index('Q'):
        score += 1

    return score


def _build_percentiles() -> dict[frozenset[Card], float]:
    """Map every starting hand to its percentile (0 = best, 1 = worst)."""

    combos = list(combinations(Deck.STANDARD, 2))
    combos.sort(key=lambda combo: chen_score(*combo), reverse=True)
    total = len(combos)

    return {frozenset(combo): (i + 1) / total for i, combo in enumerate(combos)}


HAND_PERCENTILE = _build_percentiles()


def equity_vs_random(
        hole: list[Card], board: list[Card], opponents: int, samples: int,
) -> float:
    """Share of the pot won against random hands, via pokerkit's simulator."""

    return calculate_hand_strength(
        opponents + 1,
        [hole],
        board,
        2,
        5,
        Deck.STANDARD,
        (StandardHighHand,),
        sample_count=samples,
    )


def equity_vs_ranges(
        hole: list[Card],
        board: list[Card],
        ranges: list[float],
        samples: int,
        rng: random.Random,
) -> float:
    """Share of the pot won against opponents holding hands from their
    likely ranges. ``ranges`` holds one top-fraction per opponent: hands in
    the top fraction are always accepted, others only rarely."""

    known = set(hole) | set(board)
    base_deck = [card for card in Deck.STANDARD if card not in known]
    runout_count = 5 - len(board)
    score = 0.0

    for _ in range(samples):
        deck = base_deck[:]
        rng.shuffle(deck)
        villains = []

        for top in ranges:
            pick = None

            for attempt in range(25):
                i, j = rng.sample(range(len(deck)), 2)
                candidate = (deck[i], deck[j])

                if (
                        HAND_PERCENTILE[frozenset(candidate)] <= top
                        or rng.random() < 0.05
                        or attempt == 24
                ):
                    pick = (i, j)
                    break

            i, j = sorted(pick, reverse=True)
            villains.append([deck.pop(i), deck.pop(j)])

        full_board = board + deck[:runout_count]
        mine = StandardHighHand.from_game(hole, full_board)
        theirs = [StandardHighHand.from_game(v, full_board) for v in villains]
        best = max(theirs)

        if mine > best:
            score += 1
        elif mine == best:
            score += 1 / (1 + sum(1 for hand in theirs if hand == mine))

    return score / samples


# ------------------------------------------------------------- bot settings

@dataclass(frozen=True)
class Skill:
    noise: float  # random error in reading its own hand strength
    samples: int  # Monte Carlo samples per decision
    uses_ranges: bool  # model opponents' ranges instead of random hands
    exploits: bool  # adjust bluffs and value bets to how often opponents fold
    open_ranges: tuple[float, ...]  # preflop open range by players left to act (1..5)
    limp_range: float
    reraise_range: float
    call_range: float
    all_in_call_range: float
    value_strength: float  # per-opponent strength needed to value bet
    raise_strength: float  # per-opponent strength needed to raise a bet
    call_margin: float  # extra equity demanded over pot odds (negative = loose)
    cbet: float
    semi_bluff: float
    river_bluff: float
    slowplay: float


SKILLS = {
    'easy': Skill(
        noise=0.12, samples=70, uses_ranges=False, exploits=False,
        open_ranges=(0.08, 0.07, 0.06, 0.06, 0.05), limp_range=0.65,
        reraise_range=0.025, call_range=0.55, all_in_call_range=0.22,
        value_strength=0.78, raise_strength=0.92, call_margin=-0.12,
        cbet=0.2, semi_bluff=0.03, river_bluff=0.03, slowplay=0.3,
    ),
    'medium': Skill(
        noise=0.05, samples=140, uses_ranges=False, exploits=False,
        open_ranges=(0.3, 0.25, 0.22, 0.2, 0.2), limp_range=0.38,
        reraise_range=0.05, call_range=0.25, all_in_call_range=0.08,
        value_strength=0.66, raise_strength=0.82, call_margin=0.02,
        cbet=0.5, semi_bluff=0.15, river_bluff=0.08, slowplay=0.1,
    ),
    'hard': Skill(
        noise=0.0, samples=220, uses_ranges=True, exploits=True,
        open_ranges=(0.45, 0.32, 0.25, 0.2, 0.15), limp_range=0.0,
        reraise_range=0.05, call_range=0.2, all_in_call_range=0.06,
        value_strength=0.6, raise_strength=0.78, call_margin=0.0,
        cbet=0.65, semi_bluff=0.3, river_bluff=0.2, slowplay=0.15,
    ),
}


@dataclass(frozen=True)
class Personality:
    looseness: float = 1.0  # widens or narrows starting-hand ranges
    aggression: float = 1.0  # how readily it bets and raises
    bluffing: float = 1.0
    stickiness: float = 0.0  # extra willingness to call down


# ------------------------------------------------------------------- views

@dataclass
class OpponentInfo:
    """Public information about one opponent still in the hand."""

    preflop: str  # 'raise', 'call' or 'none'
    postflop_raises: int
    vpip_rate: float
    aggression_factor: float
    fold_to_bet: float


@dataclass
class BotView:
    hole: list[Card]
    board: list[Card]
    street: int  # 0 preflop, 1 flop, 2 turn, 3 river
    to_call: int
    pot: int  # every chip in the middle, including bets on this street
    my_bet: int
    my_stack: int
    big_blind: int
    min_raise_to: int | None  # None when raising is not allowed
    max_raise_to: int | None
    opponents: list[OpponentInfo]  # opponents who have not folded
    players_behind: int  # opponents still to act after me preflop
    raises_this_street: int  # voluntary bets and raises (blinds excluded)
    limpers: int
    was_preflop_raiser: bool


# A decision is ('fold', None), ('check_call', None) or ('raise', amount_to).
Decision = tuple[str, int | None]


class Bot:
    def __init__(
            self,
            skill: Skill,
            personality: Personality,
            rng: random.Random | None = None,
    ) -> None:
        self.skill = skill
        self.personality = personality
        self.rng = rng or random.Random()

    # ------------------------------------------------------------ helpers

    @staticmethod
    def pot_odds(view: BotView) -> float:
        if view.to_call <= 0:
            return 0.0

        return view.to_call / (view.pot + view.to_call)

    def raise_to(self, view: BotView, amount: float) -> Decision:
        """Raise to ``amount`` (clamped to legal limits), else just call."""

        if view.min_raise_to is None or view.max_raise_to is None:
            return 'check_call', None

        amount = int(round(amount))
        amount = max(view.min_raise_to, min(view.max_raise_to, amount))

        # Leaving a sliver behind is pointless; move all-in instead.
        if amount >= 0.85 * view.max_raise_to:
            amount = view.max_raise_to

        return 'raise', amount

    def pot_sized(self, view: BotView, fraction: float) -> Decision:
        """Bet or raise by ``fraction`` of the pot (after calling)."""

        pot_after_call = view.pot + view.to_call

        return self.raise_to(
            view, view.my_bet + view.to_call + fraction * pot_after_call,
        )

    def fold_or_check(self, view: BotView) -> Decision:
        return ('check_call', None) if view.to_call == 0 else ('fold', None)

    def chance(self, probability: float) -> bool:
        return self.rng.random() < probability

    def bluff_scale(self, view: BotView) -> float:
        """How much to bluff: less against players who rarely fold, more
        against players who fold too often."""

        if not self.skill.exploits or not view.opponents:
            return 1.0

        stickiest = min(o.fold_to_bet for o in view.opponents)

        return min(1.6, max(0.0, (stickiest - 0.2) / 0.25))

    def facing_stations(self, view: BotView) -> bool:
        """True when every opponent left calls bets too often."""

        return self.skill.exploits and bool(view.opponents) and max(
            o.fold_to_bet for o in view.opponents
        ) < 0.32

    # ----------------------------------------------------------- decisions

    def decide(self, view: BotView) -> Decision:
        if view.street == 0:
            return self.preflop(view)

        return self.postflop(view)

    def preflop(self, view: BotView) -> Decision:
        skill = self.skill
        style = self.personality
        bb = view.big_blind
        percentile = HAND_PERCENTILE[frozenset(view.hole)]
        percentile += self.rng.gauss(0, skill.noise * 0.5)
        behind = max(1, min(view.players_behind, len(skill.open_ranges)))

        if view.raises_this_street == 0:
            open_range = skill.open_ranges[behind - 1] * style.looseness
            size = (2.5 if skill.uses_ranges else 3) * bb + bb * view.limpers

            if view.to_call == 0:  # big blind, nobody raised
                if percentile <= open_range * 0.5 and self.chance(0.8 * style.aggression):
                    return self.raise_to(view, size + bb)

                return 'check_call', None

            if percentile <= open_range:
                return self.raise_to(view, size)

            limp_range = skill.limp_range

            if skill.uses_ranges and view.players_behind == 1:
                limp_range = 0.55  # small blind completing against the big blind

            if percentile <= limp_range * style.looseness:
                return 'check_call', None

            return 'fold', None

        # Facing a raise.
        raises = view.raises_this_street
        current_bet = view.my_bet + view.to_call
        reraise_range = skill.reraise_range * style.aggression / raises

        if percentile <= reraise_range:
            return self.raise_to(view, current_bet * 3)

        if (
                skill.uses_ranges and raises == 1
                and 0.2 < percentile < 0.35
                and self.chance(0.08 * style.bluffing * self.bluff_scale(view))
        ):
            return self.raise_to(view, current_bet * 3)  # light 3-bet

        pot_odds = self.pot_odds(view)
        call_range = skill.call_range * style.looseness
        call_range *= 1 + 2 * max(0.0, 0.33 - pot_odds)
        call_range /= 1 + 0.7 * (raises - 1)

        if view.players_behind == 0 and skill.uses_ranges:
            call_range *= 1.2  # closing the action

        if view.to_call >= 0.35 * view.my_stack:
            call_range = min(call_range, skill.all_in_call_range * style.looseness)

        if percentile <= call_range:
            return 'check_call', None

        return 'fold', None

    def opponent_range(self, opponent: OpponentInfo) -> float:
        top = {'raise': 0.18, 'call': 0.4, 'none': 0.75}[opponent.preflop]
        top *= min(2.0, max(0.6, opponent.vpip_rate / 0.3))
        top *= 0.6 ** opponent.postflop_raises

        if opponent.aggression_factor > 2.5:
            top *= 1.25  # an aggressive player bets weaker hands
        elif opponent.aggression_factor < 1:
            top *= 0.8

        return min(1.0, max(0.03, top))

    def equity(self, view: BotView) -> float:
        opponents = len(view.opponents)
        samples = max(80, self.skill.samples // max(1, opponents - 1))

        if self.skill.uses_ranges:
            ranges = [self.opponent_range(o) for o in view.opponents]

            return equity_vs_ranges(view.hole, view.board, ranges, samples, self.rng)

        return equity_vs_random(view.hole, view.board, opponents, samples)

    def postflop(self, view: BotView) -> Decision:
        skill = self.skill
        style = self.personality
        opponents = max(1, len(view.opponents))
        equity = self.equity(view) + self.rng.gauss(0, skill.noise)
        equity = min(0.999, max(0.001, equity))
        # Equity against one opponent that would give this multiway equity.
        strength = equity ** (1 / opponents)
        heads_up = opponents == 1
        bluffs = self.bluff_scale(view)
        stations = self.facing_stations(view)
        value_strength = skill.value_strength - (0.06 if stations else 0.0)

        if view.to_call == 0:
            if strength > 0.85:
                if self.chance(skill.slowplay):
                    return 'check_call', None

                return self.pot_sized(view, 0.7)

            if strength > value_strength and self.chance(min(1.0, 0.85 * style.aggression)):
                return self.pot_sized(view, 0.66 if stations else 0.5)

            if (
                    view.was_preflop_raiser and view.street == 1 and opponents <= 2
                    and self.chance(skill.cbet * style.aggression * min(1.0, bluffs))
            ):
                return self.pot_sized(view, 0.4)  # continuation bet

            if (
                    heads_up and view.street < 3 and 0.3 < strength < 0.5
                    and self.chance(skill.semi_bluff * style.bluffing * bluffs)
            ):
                return self.pot_sized(view, 0.55)

            if (
                    heads_up and view.street == 3 and strength < 0.25
                    and self.chance(skill.river_bluff * style.bluffing * bluffs)
            ):
                return self.pot_sized(view, 0.75)

            return 'check_call', None

        raise_needed = skill.raise_strength + 0.06 * (view.raises_this_street - 1)

        if strength > raise_needed and self.chance(min(1.0, 0.75 * style.aggression)):
            return self.pot_sized(view, 0.8)

        if equity >= self.pot_odds(view) + skill.call_margin - style.stickiness:
            return 'check_call', None

        return 'fold', None


# ----------------------------------------------------------- the AI players

@dataclass(frozen=True)
class Character:
    name: str
    avatar: str
    color: str
    personality: Personality


CHARACTERS = (
    Character('MathNinja', '\U0001F977', 'blue',
              Personality(looseness=0.85, aggression=1.0, bluffing=0.8)),
    Character('CardShark88', '\U0001F988', 'blue',
              Personality(looseness=0.95, aggression=1.3, bluffing=1.1)),
    Character('RiverRat', '\U0001F99D', 'indigo',
              Personality(looseness=1.3, aggression=0.7, bluffing=0.6, stickiness=0.05)),
    Character('LuckyLlama', '\U0001F999', 'yellow',
              Personality(looseness=1.4, aggression=1.0, bluffing=1.2)),
    Character('TiltMaster', '\U0001F435', 'red',
              Personality(looseness=1.15, aggression=1.6, bluffing=1.5)),
)


def create_bot(difficulty: str, character: Character, rng: random.Random | None = None) -> Bot:
    if difficulty not in SKILLS:
        raise ValueError(f'Unknown difficulty: {difficulty!r}')

    return Bot(SKILLS[difficulty], character.personality, rng)
