"""
Shared feature code for the PokerPilot XGBoost model.

Both train.py and predict.py import from here, so the model always sees
features built the exact same way at training time and in the app.
"""

import numpy as np
import pandas as pd

# The 5 actions the model chooses between.
# "all-in" (only 69 rows) is folded into "raise" when facing a bet,
# or "bet" when not, so the model has enough examples of every class.
ACTIONS = ["fold", "check", "call", "bet", "raise"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}

# Columns marked use_as_model_input = Yes in poker_feature_dictionary.csv.
# NOTE: big_blind_usd, seat, stake_level, opp_stack_bb and opp_net_bb_last20
# are left out too. They're noise for the decision (seat number and dollar
# stakes don't change the right play), and dropping them raised validation
# accuracy from 80.7% to 81.5% and macro F1 from 0.698 to 0.714.
NUMERIC_FEATURES = [
    "street_idx", "num_players", "players_in_hand",
    "players_yet_to_act", "in_position", "pot_bb", "to_call_bb", "pot_odds",
    "facing_bet", "facing_bet_pot_frac", "stack_bb", "effective_stack_bb", "spr",
    "raises_this_street", "callers_this_street", "is_street_aggressor",
    "was_prev_street_aggressor",
    "card_high_rank", "card_low_rank", "is_pocket_pair", "is_suited", "rank_gap",
    "chen_score", "preflop_percentile", "made_hand_rank", "made_hand_strength",
    "flush_draw", "draw_outs",
    "board_paired", "board_connectedness", "board_high_rank", "equity_vs_random",
    # NOTE: the acting player's own tendency stats (player_vpip, player_pfr,
    # player_bb_per_100, ...) are left out on purpose. Keeping them would make
    # the model copy whatever style the current user already plays, instead of
    # recommending what winning players do.
    "opp_hands_seen", "opp_vpip", "opp_pfr", "opp_aggression_factor",
    "opp_fold_to_bet", "opp_wtsd", "opp_won_at_showdown", "opp_bb_per_100",
    "table_avg_vpip",
]

CATEGORICAL_FEATURES = [
    "street", "position", "own_prev_action_street",
    "last_opp_action", "made_hand", "pair_type", "straight_draw",
    "board_suit_texture",
]

# Extra features we build ourselves (feature engineering).
ENGINEERED_FEATURES = [
    "equity_minus_pot_odds",   # >0 means calling is profitable vs random hands
    "equity_x_players",        # equity adjusted for how many people are in
    "to_call_frac_stack",      # how much of our stack a call costs
    "opp_vpip_minus_table",    # is the main opponent looser than the table?
    "is_preflop",
    "strong_made_hand",        # two pair or better
    "has_big_draw",            # 8+ outs
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES

# Fixed category lists so the app and training always agree,
# even if a category is missing from a given batch.
CATEGORY_LEVELS = {
    "street": ["Preflop", "Flop", "Turn", "River"],
    "position": ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB", "BTN/SB"],
    "own_prev_action_street": ["none", "check", "call", "bet", "raise"],
    "last_opp_action": ["none", "fold", "check", "call", "bet", "raise", "all-in"],
    "made_hand": ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
                  "Flush", "Full House", "Four of a Kind", "Straight Flush"],
    "pair_type": ["none", "pocket pair", "underpair", "bottom pair", "middle pair",
                  "top pair", "overpair", "board pair only", "two pair or better"],
    "straight_draw": ["none", "gutshot", "open-ended"],
    "board_suit_texture": ["rainbow", "two-tone", "monotone"],
}


def map_target(df: pd.DataFrame) -> pd.Series:
    """Turn target_action text into a class id (0-4)."""
    action = df["target_action"].copy()
    allin = action == "all-in"
    action[allin & (df["facing_bet"] == 1)] = "raise"
    action[allin & (df["facing_bet"] == 0)] = "bet"
    return action.map(ACTION_TO_ID).astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Take raw decision rows (or one game-state dict as a 1-row frame)
    and return the exact model input table."""
    X = pd.DataFrame(index=df.index)

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan

    for col in CATEGORICAL_FEATURES:
        values = df[col] if col in df else pd.Series(np.nan, index=df.index)
        X[col] = pd.Categorical(values, categories=CATEGORY_LEVELS[col])

    eq = X["equity_vs_random"]
    X["equity_minus_pot_odds"] = eq - X["pot_odds"]
    X["equity_x_players"] = eq * X["players_in_hand"]
    X["to_call_frac_stack"] = X["to_call_bb"] / X["stack_bb"].replace(0, np.nan)
    X["opp_vpip_minus_table"] = X["opp_vpip"] - X["table_avg_vpip"]
    X["is_preflop"] = (X["street_idx"] == 0).astype(int)
    X["strong_made_hand"] = (X["made_hand_rank"] >= 2).astype(int)
    X["has_big_draw"] = (X["draw_outs"] >= 8).astype(int)

    return X[ALL_FEATURES]


def legal_mask(facing_bet) -> np.ndarray:
    """Which actions are legal (1) or not (0), per row.

    Facing a bet:      fold, call, raise
    Not facing a bet:  check, bet, raise (raise = BB raising preflop)
    """
    facing = np.asarray(facing_bet).astype(int).reshape(-1, 1)
    facing_row = np.array([1, 0, 1, 0, 1])
    free_row = np.array([0, 1, 0, 1, 1])
    return np.where(facing == 1, facing_row, free_row)


def apply_mask(probs: np.ndarray, facing_bet) -> np.ndarray:
    """Zero out illegal actions and renormalize so each row sums to 1."""
    p = probs * legal_mask(facing_bet)
    return p / p.sum(axis=1, keepdims=True)
