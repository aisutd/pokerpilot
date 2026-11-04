"""Five-fold chronological, game-grouped evaluation on training data."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from .data import TARGET, feature_columns, feature_frame, load_splits
from .model import _class_weights, build_pipeline

DATA = Path("pokerpilot_model/data")
OUTPUT = Path("pokerpilot_model/siri")

def main():
    df = load_splits(DATA)["train"]
    columns = feature_columns(DATA / "poker_feature_dictionary.csv")

    games = (
        df[["game_id", "date"]]
        .drop_duplicates(subset=["game_id"])
        .sort_values(["date", "game_id"])["game_id"]
        .to_numpy()
    )

    results = []
    splitter = TimeSeriesSplit(n_splits=5)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(games), 1):
        train = df[df["game_id"].isin(games[train_idx])]
        valid = df[df["game_id"].isin(games[valid_idx])]

        labels = sorted(train[TARGET].unique())
        label_ids = {name: i for i, name in enumerate(labels)}

        model = build_pipeline(
            columns, n_estimators=150, max_depth=5
        )
        model.fit(
            feature_frame(train, columns),
            train[TARGET].map(label_ids),
            classifier__sample_weight=_class_weights(
                train[TARGET], 0.5
            ),
        )

        predictions = [
            labels[int(i)]
            for i in model.predict(feature_frame(valid, columns))
        ]

        result = {
            "fold": fold,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "accuracy": accuracy_score(valid[TARGET], predictions),
            "macro_f1": f1_score(
                valid[TARGET], predictions,
                average="macro",
                labels=sorted(df[TARGET].unique()),
                zero_division=0,
            ),
        }
        results.append(result)
        print(
            f"Fold {fold}: accuracy={result['accuracy']:.4f}, "
            f"macro F1={result['macro_f1']:.4f}"
        )

    scores = pd.DataFrame(results)
    scores.to_csv(OUTPUT / "CV_RESULTS.csv", index=False)

    mean_acc = scores["accuracy"].mean()
    mean_f1 = scores["macro_f1"].mean()

    report = (
        "# Siri XGBoost — Five-Fold Cross-Validation\n\n"
        "Method: expanding chronological TimeSeriesSplit, grouped by game ID.\n"
        "Training split only; original validation and test splits excluded.\n"
        "Fixed settings: 150 estimators, max depth 5, weight power 0.5.\n\n"
        f"Mean accuracy: {mean_acc:.4%}\n\n"
        f"Mean macro F1: {mean_f1:.4f}\n\n"
        "See CV_RESULTS.csv for individual fold scores.\n"
    )
    (OUTPUT / "CV_RESULTS.md").write_text(report)

    print(f"\nMean accuracy: {mean_acc:.4%}")
    print(f"Mean macro F1: {mean_f1:.4f}")
    print("Saved CV_RESULTS.csv and CV_RESULTS.md")

if __name__ == "__main__":
    main()
