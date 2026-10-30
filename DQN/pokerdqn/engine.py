"""A small, fast No-Limit Hold'em hand engine (integer chips).

pokerkit is the rules engine of the PokerPilot website, but its showdown
evaluation costs ~50 ms per multiway hand, far too slow for reinforcement
learning. This engine follows the same rules and is checked against pokerkit
action-by-action in ``tests/test_engine.py``.

Player indices run clockwise from the small blind: with 3+ players index 0 is
the small blind, 1 the big blind and ``n - 1`` the button. Heads-up, index 0
is the button/small blind and index 1 the big blind (this matches the
dataset's ``seat`` column, which is ``index + 1``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import evaluate

STREETS = ('Preflop', 'Flop', 'Turn', 'River')


@dataclass(frozen=True)
class ActionRecord:
    street: int
    player: int
    kind: str  # fold, check, call, bet, raise, all-in (dataset naming)
    added: int  # chips put in by this action
    to: int  # the player's street total afterwards
    aggressive: bool  # raised the amount to call (bets, raises, raising all-ins)
    faced: int  # amount the player owed before acting


class IllegalAction(ValueError):
    pass


class Hand:
    def __init__(self, stacks: list[int], deck: list[int],
                 small_blind: int = 50, big_blind: int = 100) -> None:
        n = len(stacks)

        if n < 2:
            raise ValueError('need at least two players')

        if any(stack <= 0 for stack in stacks):
            raise ValueError('every player needs chips')

        self.n = n
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stacks = list(stacks)
        self.stacks = list(stacks)  # chips behind
        # Dealt round-robin starting with the small blind, like pokerkit.
        self.hole = [[deck[i], deck[n + i]] for i in range(n)]
        self.runout = list(deck[2 * n:2 * n + 5])
        self.board: list[int] = []
        self.folded = [False] * n
        self.bets = [0] * n  # this street
        self.contrib = [0] * n  # whole hand
        self.street = 0
        self.history: list[ActionRecord] = []
        self.finished = False
        self.payoffs: list[int] | None = None
        self.showdown = False
        self.winners: list[int] = []

        self._post(0, small_blind)
        self._post(1, big_blind)
        self.max_bet = max(self.bets)
        self._reset_street_betting()
        self.actor: int | None = None
        self._start_action(0 if n == 2 else 2)

    # ------------------------------------------------------------ helpers

    def _reset_street_betting(self) -> None:
        self.acted = [False] * self.n
        # Raise bookkeeping mirrors pokerkit: the largest raise increment
        # this street, who acted since the last full raise, and the run of
        # all-in raises since then (short all-ins don't reopen the betting).
        self.largest_raise = 0
        self.acted_since_full: set[int] = set()
        self.short_all_ins: list[int] = []

    @property
    def min_increment(self) -> int:
        return max(self.largest_raise, self.big_blind)

    def _post(self, player: int, amount: int) -> None:
        amount = min(amount, self.stacks[player])
        self.stacks[player] -= amount
        self.bets[player] += amount
        self.contrib[player] += amount

    def in_hand(self) -> list[int]:
        return [i for i in range(self.n) if not self.folded[i]]

    def _needs_to_act(self, player: int) -> bool:
        return (
            not self.folded[player]
            and self.stacks[player] > 0
            and (not self.acted[player] or self.bets[player] < self.max_bet)
        )

    def _start_action(self, first: int) -> None:
        """Find the first player who must act, from ``first`` clockwise."""

        live = [i for i in self.in_hand() if self.stacks[i] > 0]

        # One player with chips and nothing to call: no betting is possible.
        if len(live) <= 1 and all(
                self.bets[i] >= self.max_bet for i in live
        ):
            self.actor = None
            self._end_street()
            return

        for step in range(self.n):
            player = (first + step) % self.n

            if self._needs_to_act(player):
                self.actor = player
                return

        self.actor = None
        self._end_street()

    def _advance(self) -> None:
        if len(self.in_hand()) == 1:
            self._finish()
            return

        start = self.actor + 1
        self.actor = None

        for step in range(self.n):
            player = (start + step) % self.n

            if self._needs_to_act(player):
                self.actor = player
                return

        self._end_street()

    def _end_street(self) -> None:
        if self.finished:
            return

        if len(self.in_hand()) == 1:
            self._finish()
            return

        self.bets = [0] * self.n

        if self.street == 3:
            self._finish()
            return

        self.street += 1
        self.board = self.runout[:{1: 3, 2: 4, 3: 5}[self.street]]
        self.max_bet = 0
        self._reset_street_betting()
        self._start_action(1 if self.n == 2 else 0)

    def _finish(self) -> None:
        self.finished = True
        self.actor = None
        alive = self.in_hand()
        final = list(self.stacks)

        if len(alive) == 1:
            final[alive[0]] += sum(self.contrib)
            self.winners = alive
        else:
            self.showdown = True
            self.board = list(self.runout)
            scores = {i: evaluate(self.hole[i] + self.board) for i in alive}
            levels = sorted({c for c in self.contrib if c > 0})
            previous = 0
            winners = set()

            for level in levels:
                layer = sum(
                    min(c, level) - min(c, previous) for c in self.contrib
                )
                eligible = [i for i in alive if self.contrib[i] >= level]
                previous = level

                if not eligible:
                    continue

                best = max(scores[i] for i in eligible)
                top = [i for i in eligible if scores[i] == best]
                share, odd = divmod(layer, len(top))

                # Odd chips go to the first winners clockwise from the small
                # blind (pokerkit's choice can differ by a chip).
                for k, i in enumerate(top):
                    final[i] += share + (1 if k < odd else 0)

                if len(eligible) > 1:
                    winners.update(top)

            self.winners = sorted(winners)

        self.stacks = final
        self.payoffs = [final[i] - self.starting_stacks[i] for i in range(self.n)]

    # ------------------------------------------------------ legal actions

    @property
    def to_call(self) -> int:
        return self.max_bet - self.bets[self.actor]

    @property
    def pot(self) -> int:
        """Every chip in the middle, including bets on this street."""

        return sum(self.contrib)

    def can_check(self) -> bool:
        return self.actor is not None and self.to_call == 0

    def can_call(self) -> bool:
        return self.actor is not None and self.to_call > 0

    def can_fold(self) -> bool:
        return self.can_call()

    def raise_range(self) -> tuple[int, int] | None:
        """(min, max) street total the actor may raise to, or None."""

        if self.actor is None:
            return None

        player = self.actor
        top = self.bets[player] + self.stacks[player]

        if top <= self.max_bet:
            return None

        if (
                self.short_all_ins
                and sum(self.short_all_ins) < self.largest_raise
                and player in self.acted_since_full
        ):
            return None

        # Raising is not allowed unless someone else could put in more than
        # the current bet (as in pokerkit).
        others = [
            i for i in self.in_hand()
            if i != player and self.bets[i] + self.stacks[i] > self.max_bet
        ]

        if not others:
            return None

        return min(self.max_bet + self.min_increment, top), top

    # ------------------------------------------------------------ actions

    def _record(self, kind: str, added: int, aggressive: bool, faced: int) -> None:
        self.history.append(ActionRecord(
            self.street, self.actor, kind, added, self.bets[self.actor], aggressive, faced,
        ))

    def fold(self) -> None:
        if not self.can_fold():
            raise IllegalAction('folding is only allowed when facing a bet')

        faced = self.to_call
        self.folded[self.actor] = True
        self.acted[self.actor] = True
        self.acted_since_full.add(self.actor)
        self._record('fold', 0, False, faced)
        self._advance()

    def check_or_call(self) -> None:
        if self.actor is None:
            raise IllegalAction('no player to act')

        player = self.actor
        faced = self.to_call
        amount = min(faced, self.stacks[player])
        self.stacks[player] -= amount
        self.bets[player] += amount
        self.contrib[player] += amount
        self.acted[player] = True
        self.acted_since_full.add(player)

        if amount == 0:
            kind = 'check'
        elif self.stacks[player] == 0:
            kind = 'all-in'
        else:
            kind = 'call'

        self._record(kind, amount, False, faced)
        self._advance()

    def raise_to(self, amount: int) -> None:
        bounds = self.raise_range()

        if bounds is None or not bounds[0] <= amount <= bounds[1]:
            raise IllegalAction(f'cannot raise to {amount} (range {bounds})')

        player = self.actor
        faced = self.to_call
        added = amount - self.bets[player]
        increment = amount - self.max_bet
        was_bet = self.max_bet == 0
        self.stacks[player] -= added
        self.bets[player] = amount
        self.contrib[player] += added

        if increment >= self.largest_raise:
            self.acted_since_full = set()

        self.acted_since_full.add(player)
        self.largest_raise = max(self.largest_raise, increment)

        if self.stacks[player]:
            self.short_all_ins.clear()
        else:
            self.short_all_ins.append(increment)

        if sum(self.short_all_ins) >= self.largest_raise:
            self.short_all_ins.clear()

        self.max_bet = amount
        self.acted[player] = True

        if self.stacks[player] == 0:
            kind = 'all-in'
        else:
            kind = 'bet' if was_bet else 'raise'

        self._record(kind, added, True, faced)
        self._advance()
