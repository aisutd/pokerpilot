"""Opponent tendencies from public information, as defined in the dataset's
feature dictionary (VPIP, PFR, aggression factor, fold to bet, ...).

Statistics only ever include *finished* hands, so a decision never sees the
outcome of the hand being played (the dataset uses "earlier games only").
Hole cards are only counted through showdown results, which are public.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import Hand


@dataclass
class PlayerStats:
    hands: int = 0
    vpip: int = 0
    pfr: int = 0
    post_aggressive: int = 0
    post_calls: int = 0
    faced_bets: int = 0
    folds_to_bet: int = 0
    flops: int = 0
    showdowns: int = 0
    showdowns_won: int = 0

    @staticmethod
    def _ratio(top: int, bottom: int) -> float | None:
        return top / bottom if bottom else None

    def summary(self) -> dict:
        return {
            'hands_seen': self.hands,
            'vpip': self._ratio(self.vpip, self.hands),
            'pfr': self._ratio(self.pfr, self.hands),
            'aggression_factor': self._ratio(self.post_aggressive, self.post_calls),
            'fold_to_bet': self._ratio(self.folds_to_bet, self.faced_bets),
            'wtsd': self._ratio(self.showdowns, self.flops),
            'won_at_showdown': self._ratio(self.showdowns_won, self.showdowns),
        }


class StatsTracker:
    """Keeps PlayerStats per player id across the hands of a session."""

    def __init__(self) -> None:
        self.players: dict[object, PlayerStats] = {}

    def get(self, player_id) -> PlayerStats:
        return self.players.setdefault(player_id, PlayerStats())

    def summary(self, player_id) -> dict:
        return self.get(player_id).summary()

    def record_hand(self, hand: Hand, ids: list) -> None:
        """Update from a finished hand; ``ids[i]`` is player i's id."""

        assert hand.finished
        preflop_fold = set()

        for record in hand.history:
            if record.street == 0 and record.kind == 'fold':
                preflop_fold.add(record.player)

        reached_flop = len(hand.board) >= 3 or hand.showdown
        vpip = set()
        pfr = set()

        for record in hand.history:
            stats = self.get(ids[record.player])

            if record.street == 0:
                if record.added > 0 or record.aggressive:
                    vpip.add(record.player)

                if record.aggressive:
                    pfr.add(record.player)

                continue

            if record.aggressive:
                stats.post_aggressive += 1
            elif record.added > 0:
                stats.post_calls += 1

            if record.faced > 0:
                stats.faced_bets += 1

                if record.kind == 'fold':
                    stats.folds_to_bet += 1

        for i in range(hand.n):
            stats = self.get(ids[i])
            stats.hands += 1
            stats.vpip += i in vpip
            stats.pfr += i in pfr

            if reached_flop and i not in preflop_fold:
                stats.flops += 1

                if hand.showdown and not hand.folded[i]:
                    stats.showdowns += 1
                    stats.showdowns_won += i in hand.winners
