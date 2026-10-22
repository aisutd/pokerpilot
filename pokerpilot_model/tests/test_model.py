from pathlib import Path

import pytest

try:
    from xgboost import XGBClassifier  # noqa: F401
except Exception as exc:
    pytest.skip(f"XGBoost runtime unavailable: {exc}", allow_module_level=True)

from pokerpilot_model.data import feature_columns, feature_frame, load_splits
from pokerpilot_model.model import build_pipeline

ROOT = Path(__file__).resolve().parents[2]


def test_small_classifier_fit_and_predict():
    frames = load_splits(ROOT / "Dataset")
    columns = feature_columns(ROOT / "Dataset/poker_feature_dictionary.csv")
    sample = frames["train"].groupby("target_action", group_keys=False).head(8)
    model = build_pipeline(columns, n_estimators=3, max_depth=2, n_jobs=1)
    model.fit(feature_frame(sample, columns), sample.target_action)
    prediction = model.predict(feature_frame(frames["val"].head(12), columns))
    assert len(prediction) == 12
