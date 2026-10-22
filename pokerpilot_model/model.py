"""Training and metrics for the dictionary-defined action classifier."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import TARGET, feature_frame, feature_columns, load_splits

CATEGORICAL = {"stake_level", "street", "position", "own_prev_action_street",
               "last_opp_action", "made_hand", "pair_type", "straight_draw",
               "board_suit_texture"}


def _transformer(columns: list[str]) -> ColumnTransformer:
    categorical = [c for c in columns if c in CATEGORICAL]
    numeric = [c for c in columns if c not in CATEGORICAL]
    return ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
    ])


def build_pipeline(columns: list[str], **model_args: object) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "XGBoost could not load; on macOS install the OpenMP runtime (libomp)"
        ) from exc
    defaults = dict(n_estimators=350, max_depth=5, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, reg_lambda=2,
                    objective="multi:softprob", eval_metric="mlogloss",
                    tree_method="hist", random_state=23, n_jobs=4)
    defaults.update(model_args)
    return Pipeline([("features", _transformer(columns)),
                     ("classifier", XGBClassifier(**defaults))])


def metrics(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels,
                                         output_dict=True, zero_division=0),
    }


def train(data_dir: Path, output_dir: Path, estimators: int = 350) -> dict:
    frames = load_splits(data_dir)
    columns = feature_columns(data_dir / "poker_feature_dictionary.csv")
    labels = sorted(frames["train"][TARGET].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_y = frames["train"][TARGET].map(label_to_id)
    best_round, best_score, val_result = 1, -1.0, None
    candidates = sorted(set((min(100, estimators), min(200, estimators), estimators)))
    for rounds in candidates:
        candidate = build_pipeline(columns, n_estimators=rounds)
        candidate.fit(feature_frame(frames["train"], columns), train_y)
        val_pred = [labels[int(index)] for index in candidate.predict(
            feature_frame(frames["val"], columns))]
        score = f1_score(frames["val"][TARGET], val_pred, labels=labels,
                         average="macro", zero_division=0)
        if score > best_score:
            best_round, best_score = rounds, score
            val_result = metrics(frames["val"][TARGET], val_pred, labels)

    # Validation selects boosting rounds; the final model refits on train+val.
    # Test is not consulted until after this selection and final fit.
    selected = build_pipeline(columns, n_estimators=best_round)
    fit_rows = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    selected.fit(feature_frame(fit_rows, columns), fit_rows[TARGET].map(label_to_id))
    test_pred = [labels[int(index)] for index in selected.predict(
        feature_frame(frames["test"], columns))]
    test_result = metrics(frames["test"][TARGET], test_pred, labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": selected, "columns": columns, "labels": labels,
                 "best_iteration": best_round,
                 "raise_size_median": {
                     action: float(fit_rows.loc[fit_rows[TARGET].eq(action),
                                                "target_action_amount_to_bb"].median())
                     for action in ("bet", "raise")
                 }},
                output_dir / "model.joblib")
    results = {
        "seed": 23, "features": columns, "labels": labels,
        "rows": {key: len(value) for key, value in frames.items()},
        "unique_games": {key: int(value.game_id.nunique()) for key, value in frames.items()},
        "validation": val_result, "selected_estimators": best_round, "test": test_result,
    }
    (output_dir / "metrics.json").write_text(__import__("json").dumps(results, indent=2))
    return results


def load_model(path: Path) -> dict:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Invalid model artifact at {path}")
    return artifact
