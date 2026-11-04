"""Simple decision baseline for recorded equity and pot-odds fields."""

import numpy as np
import pandas as pd


def baseline(frame: pd.DataFrame) -> np.ndarray:
    actions = []
    equity = pd.to_numeric(frame.equity_vs_random, errors="coerce").fillna(0.5)
    odds = pd.to_numeric(frame.pot_odds, errors="coerce").fillna(0)
    facing = pd.to_numeric(frame.facing_bet, errors="coerce").fillna(0).astype(bool)
    for hand_equity, pot_odds, is_facing in zip(equity, odds, facing):
        actions.append("call" if is_facing and hand_equity >= pot_odds
                       else "fold" if is_facing
                       else "bet" if hand_equity >= 0.68
                       else "check")
    return np.asarray(actions)
