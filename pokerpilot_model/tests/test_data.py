from pathlib import Path

import pandas as pd
import pytest

from pokerpilot_model.data import feature_columns, feature_frame, load_splits
from pokerpilot_model.simulate import baseline

ROOT = Path(__file__).resolve().parents[2]


def test_real_splits_are_game_disjoint_and_dictionary_excludes_leaks():
    frames = load_splits(ROOT / "Dataset")
    assert [len(frames[s]) for s in ("train", "val", "test")] == [9055, 2023, 1862]
    columns = feature_columns(ROOT / "Dataset/poker_feature_dictionary.csv")
    assert "target_action" not in columns
    assert not any(name.startswith("outcome_") for name in columns)
    assert "player_vpip" in columns


def test_overlap_is_rejected():
    frame = pd.DataFrame({"game_id": [1], "target_action": ["fold"]})
    from unittest.mock import patch
    with patch("pokerpilot_model.data.pd.read_csv", side_effect=[frame, frame, frame]):
        with pytest.raises(ValueError, match="Game leakage"):
            load_splits(Path("unused"))


def test_rule_baseline_is_reproducible():
    sample = pd.DataFrame({"equity_vs_random": [0.7, 0.1, 0.8],
                           "pot_odds": [0.4, 0.2, 0], "facing_bet": [1, 1, 0]})
    assert baseline(sample).tolist() == ["call", "fold", "bet"]
    assert baseline(sample).tolist() == baseline(sample).tolist()
