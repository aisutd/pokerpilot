"""
PokerPilot: use the trained model to recommend an action for one game state.

From your FastAPI backend:
    from predict import PokerPilotModel
    model = PokerPilotModel()            # load once at startup
    advice = model.recommend(game_state) # game_state is a dict

Run this file directly to see a demo:  python3 predict.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from features import ACTIONS, apply_mask, build_features

ARTIFACTS = Path(__file__).parent / "artifacts"


class PokerPilotModel:
    def __init__(self, artifacts_dir: Path = ARTIFACTS):
        self.action_model = XGBClassifier()
        self.action_model.load_model(artifacts_dir / "action_model.json")
        self.sizing_model = XGBRegressor()
        self.sizing_model.load_model(artifacts_dir / "sizing_model.json")

    def recommend(self, state: dict) -> dict:
        """state uses the same column names as poker_decisions_*.csv.
        Missing fields are fine (XGBoost handles blanks), but the more
        you fill in, the better the advice."""
        df = pd.DataFrame([state])
        X = build_features(df)
        facing = int(state.get("facing_bet", 0))

        probs = apply_mask(self.action_model.predict_proba(X), [facing])[0]
        best = ACTIONS[int(np.argmax(probs))]

        result = {
            "action": best,
            "probabilities": {a: round(float(p), 3) for a, p in zip(ACTIONS, probs)},
            "confidence": round(float(probs.max()), 3),
        }
        if best in ("bet", "raise"):
            frac = float(np.clip(self.sizing_model.predict(X)[0], 0.2, 3.0))
            pot = float(state.get("pot_bb", 0))
            result["size_pot_frac"] = round(frac, 2)
            result["size_bb"] = round(frac * pot, 1)
        return result


if __name__ == "__main__":
    model = PokerPilotModel()

    # Demo 1: replay a few real decisions from the test set
    test = pd.read_csv(Path(__file__).parent / "data" / "poker_decisions_test.csv")
    print("Real test-set decisions vs the model:\n")
    for _, row in test.sample(6, random_state=7).iterrows():
        r = model.recommend(row.to_dict())
        size = f" ({r['size_bb']} bb)" if "size_bb" in r else ""
        print(f"  {row['street']:<7} {row['hole_cards']:<6} board={str(row['board']):<15}"
              f" player did: {row['target_action']:<6} model says: {r['action']}{size}"
              f"  [{r['confidence']:.0%} sure]")

    # Demo 2: a hand typed in by hand, like the app would send
    state = {
        "stake_level": "100NL", "big_blind_usd": 1.0, "street": "Flop", "street_idx": 1,
        "position": "BTN", "num_players": 6, "players_in_hand": 2,
        "players_yet_to_act": 0, "in_position": 1,
        "pot_bb": 12.0, "to_call_bb": 0.0, "pot_odds": 0.0, "facing_bet": 0,
        "facing_bet_pot_frac": 0.0, "stack_bb": 95.0, "effective_stack_bb": 95.0,
        "spr": 7.9, "raises_this_street": 0, "callers_this_street": 0,
        "is_street_aggressor": 0, "was_prev_street_aggressor": 1,
        "own_prev_action_street": "none", "last_opp_action": "check",
        "card_high_rank": 14, "card_low_rank": 13, "is_suited": 1, "is_pocket_pair": 0,
        "rank_gap": 1, "made_hand": "Pair", "made_hand_rank": 1, "pair_type": "top pair",
        "made_hand_strength": 0.55, "equity_vs_random": 0.82, "draw_outs": 0,
        "board_suit_texture": "rainbow", "board_paired": 0, "board_high_rank": 14,
    }
    print("\nHand-typed example (AK on an Ace-high flop, opponent checked):")
    print(" ", model.recommend(state))
