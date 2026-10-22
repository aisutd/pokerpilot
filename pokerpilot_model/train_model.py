"""Train PokerPilot's behavior-imitation XGBoost classifier on chronological decision splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

TARGET = "target_action"


def train(dataset_dir: Path, output_dir: Path) -> dict:
    dictionary_path = dataset_dir / "poker_feature_dictionary.csv"
    dictionary = pd.read_csv(dictionary_path)
    required_meta = {"column", "use_as_model_input"}
    if not required_meta.issubset(dictionary.columns):
        raise ValueError(f"Feature dictionary must include {required_meta}")
    features = dictionary.loc[
        dictionary["use_as_model_input"].astype(str).str.strip().str.casefold().eq("yes"),
        "column",
    ].astype(str).tolist()
    if not features:
        raise ValueError("No input features marked Yes in feature dictionary")
    if TARGET in features or any(f.startswith("outcome_") or f.startswith("target_") for f in features):
        raise ValueError("Potential target/outcome leakage in input feature list")

    frames = {}
    for split in ("train", "val", "test"):
        path = dataset_dir / f"poker_decisions_{split}.csv"
        df = pd.read_csv(path, low_memory=False)
        absent = sorted(set(features + [TARGET]) - set(df.columns))
        if absent:
            raise ValueError(f"Missing columns in {path.name}: {absent}")
        df = df.dropna(subset=[TARGET]).copy()
        if df.empty:
            raise ValueError(f"{path.name} has no labeled rows")
        frames[split] = df

    train_df = frames["train"]
    categorical = [f for f in features if not pd.api.types.is_numeric_dtype(train_df[f])]
    numeric = [f for f in features if f not in categorical]

    # Consistent, explicit types across training and inference.
    def inputs(frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.reindex(columns=features).copy()
        for name in numeric:
            x[name] = pd.to_numeric(x[name], errors="coerce")
        for name in categorical:
            x[name] = x[name].astype("string").fillna("missing").astype(str)
        return x

    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="constant", fill_value=0), numeric))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    preprocessing = ColumnTransformer(transformers=transformers)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df[TARGET].astype(str))
    if len(encoder.classes_) < 2:
        raise ValueError("Training data has fewer than two action classes")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob" if len(encoder.classes_) > 2 else "binary:logistic",
        eval_metric="mlogloss" if len(encoder.classes_) > 2 else "logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )
    pipeline = Pipeline([("preprocess", preprocessing), ("model", model)])
    pipeline.fit(inputs(train_df), y_train)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "task": "Behavior imitation: predict action taken by a historical player, NOT optimal action",
        "n_features_before_encoding": len(features),
        "action_classes": encoder.classes_.tolist(),
        "split_results": {},
    }
    for split, frame in frames.items():
        y_true = frame[TARGET].astype(str)
        predictions = encoder.inverse_transform(pipeline.predict(inputs(frame)).astype(int))
        probabilities = pipeline.predict_proba(inputs(frame))
        known = y_true.isin(encoder.classes_)
        result = {
            "examples": len(frame),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "classification_report": classification_report(
                y_true, predictions, labels=list(encoder.classes_), output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_true, predictions, labels=list(encoder.classes_)
            ).tolist(),
        }
        if known.all():
            encoded_y = encoder.transform(y_true)
            result["log_loss"] = float(log_loss(encoded_y, probabilities, labels=np.arange(len(encoder.classes_))))
        else:
            result["unseen_true_labels"] = sorted(set(y_true) - set(encoder.classes_))
        metrics["split_results"][split] = result

    bundle = {"pipeline": pipeline, "label_encoder": encoder, "features": features,
              "numeric": numeric, "categorical": categorical}
    joblib.dump(bundle, output_dir / "poker_action_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {output_dir / 'poker_action_model.joblib'}")
    for split, result in metrics["split_results"].items():
        print(f"{split}: {result['examples']} examples, accuracy={result['accuracy']:.3f}")
    print("NOTE: Accuracy measures imitation of decisions, not poker profitability.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "Dataset")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    args = parser.parse_args()
    train(args.data, args.output)
