"""Dataset loading, split validation, and dictionary-driven features."""

from pathlib import Path

import pandas as pd

TARGET = "target_action"
FORBIDDEN_PREFIXES = ("outcome_",)
FORBIDDEN = {
    "decision_id", "game_id", "decision_idx", "date", "time", "player_id",
    "player_name", "opponent_id", "hole_cards", "board", "split", TARGET,
    "target_action_amount_to_bb", "target_chips_added_bb", "target_bet_size_pot_frac",
}


def feature_columns(dictionary_path: Path) -> list[str]:
    dictionary = pd.read_csv(dictionary_path)
    selected = dictionary.loc[dictionary.use_as_model_input.eq("Yes"), "column"].tolist()
    leaked = [c for c in selected if c in FORBIDDEN or c.startswith(FORBIDDEN_PREFIXES)]
    if leaked:
        raise ValueError(f"Feature dictionary includes forbidden columns: {leaked}")
    return selected


def load_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {
        split: pd.read_csv(data_dir / f"poker_decisions_{split}.csv")
        for split in ("train", "val", "test")
    }
    for split, frame in frames.items():
        if TARGET not in frame or "game_id" not in frame:
            raise ValueError(f"{split} split is missing target_action or game_id")
        if frame.game_id.isna().any():
            raise ValueError(f"{split} split contains missing game IDs")
    names = list(frames)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = set(frames[left].game_id) & set(frames[right].game_id)
            if overlap:
                raise ValueError(f"Game leakage between {left} and {right}: {len(overlap)} IDs")
    return frames


def feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    return frame.loc[:, columns].copy()
