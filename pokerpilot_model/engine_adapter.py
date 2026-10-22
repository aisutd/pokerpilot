"""Map the existing website PokerKit BotView to available model inputs."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from .model import load_model


class ModelBot:
    """PokerPilot classifier implementing the website engine's bot protocol."""

    def __init__(self, artifact_path: str, rng: random.Random | None = None):
        self.artifact = load_model(__import__("pathlib").Path(artifact_path))
        self.rng = rng or random.Random()

    def _features(self, view: Any) -> dict[str, Any]:
        from engine.bots import HAND_PERCENTILE, chen_score, equity_vs_random, RANK_ORDER

        hole, board = view.hole, view.board
        opponents = max(1, len(view.opponents))
        result: dict[str, Any] = {column: None for column in self.artifact["columns"]}
        result.update({
            "street": ("Preflop", "Flop", "Turn", "River")[view.street],
            "street_idx": view.street,
            "num_players": opponents + 1,
            "players_in_hand": opponents + 1,
            "players_yet_to_act": view.players_behind,
            "pot_bb": view.pot / view.big_blind,
            "to_call_bb": view.to_call / view.big_blind,
            "pot_odds": view.to_call / (view.pot + view.to_call) if view.to_call else 0,
            "facing_bet": int(view.to_call > 0),
            "facing_bet_pot_frac": view.to_call / view.pot if view.pot else 0,
            "stack_bb": view.my_stack / view.big_blind,
            "effective_stack_bb": None,
            "spr": view.my_stack / max(1, view.pot),
            "raises_this_street": view.raises_this_street,
            "callers_this_street": view.limpers,
            "is_street_aggressor": int(view.was_preflop_raiser),
            "card_high_rank": max(RANK_ORDER.index(c.rank.value) + 2 for c in hole),
            "card_low_rank": min(RANK_ORDER.index(c.rank.value) + 2 for c in hole),
            "is_pocket_pair": int(hole[0].rank == hole[1].rank),
            "is_suited": int(hole[0].suit == hole[1].suit),
            "rank_gap": abs(RANK_ORDER.index(hole[0].rank.value)
                            - RANK_ORDER.index(hole[1].rank.value)),
            "chen_score": chen_score(*hole),
            "preflop_percentile": HAND_PERCENTILE[frozenset(hole)],
            "equity_vs_random": equity_vs_random(hole, board, opponents, 100),
            "in_position": int(view.players_behind == 0),
            "position": None,
            "opp_hands_seen": 0,
            "opp_vpip": view.opponents[0].vpip_rate if view.opponents else None,
            "opp_aggression_factor": view.opponents[0].aggression_factor if view.opponents else None,
            "opp_fold_to_bet": view.opponents[0].fold_to_bet if view.opponents else None,
            "opp_stack_bb": None,
            "player_vpip": 0.3,
            "player_aggression_factor": 1.0,
        })
        return result

    def decide(self, view: Any) -> tuple[str, int | None]:
        row = pd.DataFrame([self._features(view)])
        action = self.artifact["labels"][int(self.artifact["pipeline"].predict(row)[0])]
        if action == "fold" and view.to_call == 0:
            return "check_call", None
        if action in ("bet", "raise", "all-in"):
            if view.min_raise_to is None or view.max_raise_to is None:
                return "check_call", None
            if action == "all-in":
                return "raise", view.max_raise_to
            typical_size = self.artifact.get("raise_size_median", {})
            size_bb = float(typical_size.get(action, 2.0))
            target = round(size_bb * view.big_blind)
            target = max(view.min_raise_to, min(view.max_raise_to, target))
            return "raise", target
        if action == "call":
            return "check_call", None
        if action == "check":
            return ("fold", None) if view.to_call else ("check_call", None)
        return action, None
