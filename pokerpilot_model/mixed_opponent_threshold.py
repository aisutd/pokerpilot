"""Derive and report the passive-player threshold used by PokerPilot.

The live game does not need this CSV at runtime. This script reproduces the
same PFR/VIPIP-ratio rule from the supplied player summaries and is useful for
auditing or tuning the threshold against new player data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PASSIVE_RATIO_THRESHOLD = 0.65
PREVIOUS_STATS_WEIGHT = 0.90
NEW_ACTION_WEIGHT = 0.10


def score_player(row: pd.Series) -> float:
    vpip = float(row["vpip_pct"])
    pfr = float(row["pfr_pct"])
    ratio = pfr / vpip if vpip else 0.0
    ratio_score = 1.0 if ratio <= PASSIVE_RATIO_THRESHOLD else 0.0
    return round(ratio_score, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "summary_csv",
        nargs="?",
        default=Path(__file__).parent / "data" / "poker_player_summary.csv",
        type=Path,
    )
    args = parser.parse_args()

    players = pd.read_csv(args.summary_csv)
    players["passive_ratio_score"] = players.apply(score_player, axis=1)
    players["router_model"] = players["passive_ratio_score"].apply(
        lambda score: "DQN" if score >= 0.55 else "XGBoost"
    )

    print(
        "Route weights: "
        f"{PREVIOUS_STATS_WEIGHT:.0%} previous/player stats, "
        f"{NEW_ACTION_WEIGHT:.0%} new turn action"
    )
    print(f"Passive PFR/VIPIP threshold: {PASSIVE_RATIO_THRESHOLD:.2f}")
    print(players[["player_name", "play_style", "vpip_pct", "pfr_pct",
                   "passive_ratio_score", "router_model"]].to_string(index=False))


if __name__ == "__main__":
    main()
