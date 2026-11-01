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

from .baseline import baseline
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


def _class_weights(actions: pd.Series, power: float) -> np.ndarray:
    if power == 0:
        return np.ones(len(actions), dtype=float)
    counts = actions.value_counts()
    scale = len(actions) / (len(counts) * counts)
    return actions.map(scale.pow(power)).to_numpy(dtype=float)


def train(data_dir: Path, output_dir: Path, estimators: int = 350) -> dict:
    frames = load_splits(data_dir)
    columns = feature_columns(data_dir / "poker_feature_dictionary.csv")
    labels = sorted(frames["train"][TARGET].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_y = frames["train"][TARGET].map(label_to_id)
    best_config, best_score, val_result = None, -1.0, None
    default_rounds = sorted(set((min(100, estimators), min(200, estimators), estimators)))
    candidate_configs = [
        {"n_estimators": rounds, "max_depth": 5, "class_weight_power": 0.0}
        for rounds in default_rounds
    ]
    candidate_configs.extend(
        {"n_estimators": min(rounds, estimators), "max_depth": depth,
         "class_weight_power": 0.5}
        for depth in (3, 5) for rounds in (100, 150, 200, 250)
    )
    train_x = feature_frame(frames["train"], columns)
    val_x = feature_frame(frames["val"], columns)
    for config in candidate_configs:
        candidate = build_pipeline(columns,
                                   n_estimators=config["n_estimators"],
                                   max_depth=config["max_depth"])
        candidate.fit(train_x, train_y,
                      classifier__sample_weight=_class_weights(
                          frames["train"][TARGET], config["class_weight_power"]))
        val_pred = [labels[int(index)] for index in candidate.predict(
            val_x)]
        score = f1_score(frames["val"][TARGET], val_pred, labels=labels,
                         average="macro", zero_division=0)
        if score > best_score:
            best_config, best_score = config, score
            val_result = metrics(frames["val"][TARGET], val_pred, labels)

    # Validation selects boosting rounds; the final model refits on train+val.
    # Test is not consulted until after this selection and final fit.
    assert best_config is not None
    selected = build_pipeline(columns,
                              n_estimators=best_config["n_estimators"],
                              max_depth=best_config["max_depth"])
    fit_rows = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    selected.fit(feature_frame(fit_rows, columns), fit_rows[TARGET].map(label_to_id),
                 classifier__sample_weight=_class_weights(
                     fit_rows[TARGET], best_config["class_weight_power"]))
    test_pred = [labels[int(index)] for index in selected.predict(
        feature_frame(frames["test"], columns))]
    test_result = metrics(frames["test"][TARGET], test_pred, labels)
    baseline_labels = baseline(frames["test"])
    majority_action = str(frames["train"][TARGET].mode().iloc[0])
    baselines = {
        "equity_pot_odds": metrics(frames["test"][TARGET], baseline_labels, labels),
        "majority": metrics(frames["test"][TARGET], np.repeat(majority_action, len(frames["test"])), labels),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": selected, "columns": columns, "labels": labels,
                 "best_iteration": best_config["n_estimators"],
                 "max_depth": best_config["max_depth"],
                 "class_weight_power": best_config["class_weight_power"],
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
        "validation": val_result,
        "selected_estimators": best_config["n_estimators"],
        "selected_max_depth": best_config["max_depth"],
        "selected_class_weight_power": best_config["class_weight_power"],
        "test": test_result, "baselines": baselines,
        "majority_action": majority_action,
    }
    (output_dir / "metrics.json").write_text(__import__("json").dumps(results, indent=2))
    return results


def load_model(path: Path) -> dict:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Invalid model artifact at {path}")
    return artifact
