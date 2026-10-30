
"""Offline model of historical poker actions.

Commands:
    python poker_action_model.py train
    python poker_action_model.py predict
    python poker_action_model.py evaluate

Predicts historical actions, not optimal play or profitability.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, cross_validate


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ACTIONS = ["fold", "check", "call", "bet", "raise", "all-in"]

MODEL_PATH = Path(__file__).with_name("poker_action_model.joblib")
DEFAULT_CSV = Path(__file__).with_name("poker_decisions_all.csv")

RANKS = "23456789TJQKA"
SUITS = "cdhs"

CATEGORICAL = [
    "street",
    "position",
    "last_opp_action",
    "own_prev_action_street",
]

NUMERIC = [
    "pot_bb",
    "to_call_bb",
    "stack_bb",
    "players_in_hand",
    "raises_this_street",
    "callers_this_street",
    "players_yet_to_act",
    "in_position",
    "is_street_aggressor",
    "hole_high",
    "hole_low",
    "hole_pair",
    "hole_suited",
    "hole_gap",
    "hole_rank_sum",
    "board_count",
    "board_high",
    "board_paired",
    "board_max_suit",
    "shared_rank_pairs",
    "hole_board_rank_matches",
    "combined_max_suit",
    "pot_odds_computed",
    "spr_computed",
    "to_call_pot_ratio",
]

FEATURES = CATEGORICAL + NUMERIC


# --------------------------------------------------
# CARD PARSING
# --------------------------------------------------

def parse_cards(text, expected=None):
    """Convert card text like 'As Kd' into a validated list."""

    s = (
        "" if pd.isna(text) else str(text)
    ).strip().replace(",", " ").upper().replace("10", "T")

    tokens = s.split() if s else []
    cards = []

    for token in tokens:
        if not re.fullmatch(r"[2-9TJQKA][CDHS]", token):
            raise ValueError(
                f"Invalid card {token!r}. Examples: As Kd, Th 9h"
            )

        cards.append(token[0] + token[1].lower())

    if expected is not None and len(cards) != expected:
        raise ValueError(
            f"Expected {expected} cards, got {len(cards)}"
        )

    if len(set(cards)) != len(cards):
        raise ValueError("Duplicate card entered")

    return cards


def to_float(value, default=np.nan):
    try:
        if pd.isna(value) or str(value).strip() == "":
            return default
        return float(value)

    except (TypeError, ValueError):
        return default


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def make_features(row):
    """Convert one historical decision into model inputs."""

    hole = parse_cards(
        row.get("hole_cards", ""),
        expected=2
    )

    board = parse_cards(row.get("board", ""))

    if len(board) not in (0, 3, 4, 5):
        raise ValueError(
            "Board must have 0, 3, 4 or 5 cards"
        )

    if len(set(hole + board)) != len(hole + board):
        raise ValueError(
            "A card cannot appear in both hand and board"
        )

    hr = sorted(
        (RANKS.index(c[0]) + 2 for c in hole),
        reverse=True
    )

    br = [RANKS.index(c[0]) + 2 for c in board]

    all_ranks = hr + br
    all_suits = [c[1] for c in hole + board]

    counts = {
        r: all_ranks.count(r)
        for r in set(all_ranks)
    }

    pot = to_float(row.get("pot_bb"))
    call = to_float(row.get("to_call_bb"))
    stack = to_float(row.get("stack_bb"))

    feat = {
        c: str(row.get(c, "none") or "none").strip()
        for c in CATEGORICAL
    }

    original_numeric = [
        "pot_bb",
        "to_call_bb",
        "stack_bb",
        "players_in_hand",
        "raises_this_street",
        "callers_this_street",
        "players_yet_to_act",
        "in_position",
        "is_street_aggressor",
    ]

    feat.update({
        c: to_float(row.get(c))
        for c in original_numeric
    })

    # Features derived from the actual cards
    feat.update({

        "hole_high": hr[0],
        "hole_low": hr[1],

        "hole_pair": int(hr[0] == hr[1]),

        "hole_suited": int(
            hole[0][1] == hole[1][1]
        ),

        "hole_gap": hr[0] - hr[1],
        "hole_rank_sum": sum(hr),

        "board_count": len(board),

        "board_high": max(br) if br else 0,

        "board_paired": int(
            len(br) != len(set(br))
        ),

        "board_max_suit": max(
            (
                sum(c[1] == s for c in board)
                for s in SUITS
            ),
            default=0
        ),

        "shared_rank_pairs": sum(
            v * (v - 1) // 2
            for v in counts.values()
        ),

        "hole_board_rank_matches": sum(
            r in br for r in hr
        ),

        "combined_max_suit": max(
            sum(s == suit for s in all_suits)
            for suit in SUITS
        ),

        # Derived numerical ratios
        "pot_odds_computed": (
            call / (pot + call)
            if np.isfinite(pot)
            and np.isfinite(call)
            and pot + call > 0
            else np.nan
        ),

        "spr_computed": (
            stack / pot
            if np.isfinite(stack)
            and np.isfinite(pot)
            and pot > 0
            else np.nan
        ),

        "to_call_pot_ratio": (
            call / pot
            if np.isfinite(call)
            and np.isfinite(pot)
            and pot > 0
            else np.nan
        ),
    })

    return feat


def build_features(df):
    """Apply feature engineering to the full dataset."""

    records = df.to_dict("records")

    return pd.DataFrame(
        [make_features(r) for r in records],
        columns=FEATURES
    )


# --------------------------------------------------
# XGBOOST MODEL
# --------------------------------------------------

def build_pipeline():

    preprocess = ColumnTransformer([

        (
            "number",
            SimpleImputer(strategy="median"),
            NUMERIC
        ),

        (
            "category",
            Pipeline([
                (
                    "fill",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                ),
            ]),
            CATEGORICAL
        ),

    ])

    model = XGBClassifier(
        objective="multi:softprob",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        reg_lambda=2,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=4,
    )

    return Pipeline([
        ("features", preprocess),
        ("classifier", model)
    ])


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def print_metrics(model, frame, label):

    if frame.empty:
        print(f"{label}: no records")
        return

    x = build_features(frame)

    label_map = {
        a: i for i, a in enumerate(ACTIONS)
    }

    truth = frame.target_action.map(
        label_map
    ).to_numpy()

    probs = model.predict_proba(x)

    preds = np.argmax(probs, axis=1)

    accuracy = accuracy_score(truth, preds)

    loss = log_loss(
        truth,
        probs,
        labels=list(range(6))
    )

    print(f"\n{label} ({len(frame):,} decisions)")

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Log loss: {loss:.3f}")

    print(
        classification_report(
            truth,
            preds,
            labels=list(range(6)),
            target_names=ACTIONS,
            zero_division=0
        )
    )

# --------------------------------------------------
# K-FOLD CROSS-VALIDATION
# --------------------------------------------------


def run_cross_validation(csv_path):

    print("Loading dataset for cross-validation...")

    df = pd.read_csv(csv_path)

    train_df = df[df["split"] == "train"].copy()

    X = build_features(train_df)

    label_map = {
        action: i for i, action in enumerate(ACTIONS)
    }

    y = train_df["target_action"].map(label_map)
    groups = train_df["game_id"]

    cv = GroupKFold(n_splits=5)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro"
    }

    print("Running 5-Fold Cross-Validation...")

    scores = cross_validate(
        build_pipeline(),
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    # Build Markdown report
    lines = [
        "# Poker XGBoost Cross-Validation Results",
        "",
        "## Overview",
        "",
        f"- Dataset: `{csv_path.name}`",
        f"- Training decisions: {len(train_df):,}",
        "- Algorithm: XGBoost",
        "- Cross-validation: 5-Fold GroupKFold",
        "- Grouping: game_id",
        "- Averaging: Macro (except accuracy)",
        "",
        "## Results by Fold",
        "",
        "| Fold | Accuracy | Precision | Recall | F1 Score |",
        "|---|---:|---:|---:|---:|"
    ]

    for i in range(5):
        acc = scores["test_accuracy"][i]
        prec = scores["test_precision"][i]
        rec = scores["test_recall"][i]
        f1 = scores["test_f1"][i]

        lines.append(
            f"| {i+1} | {acc:.4f} | {prec:.4f} | "
            f"{rec:.4f} | {f1:.4f} |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        "| Metric | Mean | Standard Deviation |",
        "|---|---:|---:|"
    ])

    for key, name in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision (Macro)"),
        ("recall", "Recall (Macro)"),
        ("f1", "F1 Score (Macro)")
    ]:
        values = scores[f"test_{key}"]

        lines.append(
            f"| {name} | {values.mean():.4f} | "
            f"{values.std():.4f} |"
        )

    lines.extend([
        "",
        "## Metric Explanations",
        "",
        "- **Accuracy:** Percentage of all predictions that were correct.",
        "- **Precision:** Of the predictions for a particular action, "
        "how many were correct?",
        "- **Recall:** Of all actual occurrences of an action, "
        "how many did the model identify?",
        "- **F1 Score:** Harmonic mean of precision and recall.",
        "",
        "Macro averaging treats all six actions equally:",
        "fold, check, call, bet, raise, and all-in.",
        "",
        "## Notes",
        "",
        "- Cross-validation uses only the training partition.",
        "- All decisions from one game stay in the same fold.",
        "- Validation and test partitions are excluded.",
        "- Scores evaluate historical-action classification, "
        "not the profitability of poker decisions.",
        "- Standard deviation is calculated across the five folds."
    ])

    report = "\n".join(lines) + "\n"

    output_path = Path(__file__).with_name(
        "cross_validation_results.md"
    )

    output_path.write_text(
        report,
        encoding="utf-8"
    )

    print("\nCROSS-VALIDATION RESULTS\n")
    print(report)
    print(f"Markdown report saved to: {output_path}")


    print("Loading dataset for cross-validation...")

    df = pd.read_csv(csv_path)

    # Only use training data, keeping test data untouched
    train_df = df[df["split"] == "train"].copy()

    X = build_features(train_df)

    label_map = {
        action: i
        for i, action in enumerate(ACTIONS)
    }

    y = train_df["target_action"].map(label_map)

    groups = train_df["game_id"]

    cv = GroupKFold(n_splits=5)

    print("\nRunning 5-Fold Group Cross-Validation...")

    scores = cross_validate(
        build_pipeline(),
        X,
        y,
        groups=groups,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "log_loss": "neg_log_loss",
        },
        n_jobs=1
    )

    print("\nCROSS-VALIDATION RESULTS")

    for i in range(5):
        print(
            f"Fold {i+1}: "
            f"Accuracy = {scores['test_accuracy'][i]:.3f}, "
            f"F1 = {scores['test_f1_macro'][i]:.3f}"
        )

    print(
        f"\nMean Accuracy: "
        f"{scores['test_accuracy'].mean():.3f}"
    )

    print(
        f"Mean Macro F1: "
        f"{scores['test_f1_macro'].mean():.3f}"
    )

    print(
        f"Mean Log Loss: "
        f"{-scores['test_log_loss'].mean():.3f}"
    )
# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train(csv_path):

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}"
        )

    print("Loading historical decisions...")

    df = pd.read_csv(csv_path)

    required = set(CATEGORICAL) | {
        "hole_cards",
        "board",
        "target_action",
        "split",
        "pot_bb",
        "to_call_bb",
        "stack_bb",
        "players_in_hand",
        "raises_this_street",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}"
        )

    unknown = sorted(
        set(df.target_action.dropna()) - set(ACTIONS)
    )

    if unknown:
        raise ValueError(
            f"Unexpected action labels: {unknown}"
        )

    # Use the existing CSV partitions
    train_df = df[df.split == "train"].copy()
    valid_df = df[df.split == "val"].copy()
    test_df = df[df.split == "test"].copy()

    if not all((
        len(train_df),
        len(valid_df),
        len(test_df)
    )):
        raise ValueError(
            "Expected nonempty train/val/test partitions"
        )

    # Ensure the same game doesn't occur in
    # different training/evaluation partitions.
    if "game_id" in df.columns:

        groups = [
            set(d.game_id)
            for d in (train_df, valid_df, test_df)
        ]

        for i in range(3):
            for j in range(i + 1, 3):
                if groups[i] & groups[j]:
                    raise ValueError(
                        "Data leakage: game_id occurs "
                        "in multiple partitions"
                    )

    label_map = {
        a: i for i, a in enumerate(ACTIONS)
    }

    print(
        f"Training on {len(train_df):,} decisions..."
    )

    pipeline = build_pipeline()

    x_train = build_features(train_df)

    y_train = train_df.target_action.map(
        label_map
    )

    pipeline.fit(x_train, y_train)

    print_metrics(
        pipeline,
        valid_df,
        "VALIDATION"
    )

    print_metrics(
        pipeline,
        test_df,
        "HELD-OUT TEST"
    )

    # Save model locally
    joblib.dump(
        {
            "model": pipeline,
            "actions": ACTIONS,
            "csv": str(csv_path),
        },
        MODEL_PATH
    )

    print(f"\nSaved model to {MODEL_PATH}")

    print(
        "This predicts historical behavior, "
        "not expected value or profitability."
    )


# --------------------------------------------------
# SIMPLIFIED ACTION AVAILABILITY
# --------------------------------------------------

def legal_actions(to_call, stack):
    """Simplified action filter, not a full rules engine."""

    if to_call <= 0:
        if stack > 0:
            return {"check", "bet", "all-in"}
        return {"check"}

    legal = {"fold", "call"}

    if stack > to_call:
        legal.update({"raise", "all-in"})

    elif stack > 0:
        legal.add("all-in")

    return legal


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict(model, row):

    dataframe = pd.DataFrame([row])

    x = build_features(dataframe)

    raw = model.predict_proba(x)[0]

    raw = np.asarray(raw, dtype=float)

    available = legal_actions(
        float(row["to_call_bb"]),
        float(row["stack_bb"])
    )

    # Remove unavailable actions
    masked = np.array([
        raw[i] if action in available else 0.0
        for i, action in enumerate(ACTIONS)
    ])

    total = masked.sum()

    if total <= 0:
        raise ValueError(
            "Zero probability assigned to available actions"
        )

    normalized = masked / total

    print("\n" + "=" * 45)
    print("HISTORICAL ACTION PROBABILITIES")
    print("=" * 45)

    for i in np.argsort(-normalized):

        action = ACTIONS[i]

        if action in available:

            probability = normalized[i] * 100

            print(
                f"{action.upper():8s}: "
                f"{probability:6.2f}%"
            )

    print("=" * 45)

    print(
        "Estimates are not calibrated odds "
        "of winning or expected value."
    )

    print(
        "Action filtering is simplified "
        "and does not enforce every game rule."
    )


# --------------------------------------------------
# INTERACTIVE INPUT
# --------------------------------------------------

def ask(label, default, cast=str):

    while True:

        s = input(
            f"{label} [{default}]: "
        ).strip()

        try:
            return cast(s if s else default)

        except (TypeError, ValueError):
            print("Please enter a valid value.")


def interactive(model):

    print("\nHistorical Poker Action Model")

    print(
        "Card format: As Kh or Td 9d"
    )

    print(
        "Board: blank preflop, "
        "3 flop, 4 turn, 5 river."
    )

    print(
        "Amounts are in big blinds (BB)."
    )

    print(
        "Enter q at the first prompt to quit."
    )

    while True:

        try:

            hole = input(
                "\nYour two hole cards (or q): "
            ).strip()

            if hole.lower() in ("q", "quit", "exit"):
                break

            parse_cards(
                hole,
                expected=2
            )

            board = input(
                "Board cards (blank if preflop): "
            ).strip()

            board_cards = parse_cards(board)

            if len(board_cards) not in (0, 3, 4, 5):
                print(
                    "Board must have 0, 3, 4 or 5 cards."
                )
                continue

            street = {
                0: "Preflop",
                3: "Flop",
                4: "Turn",
                5: "River"
            }[len(board_cards)]

            print(f"Inferred street: {street}")

            row = {

                "hole_cards": hole,

                "board": board,

                "street": street,

                "pot_bb": ask(
                    "Pot in BB", "4.5", float
                ),

                "to_call_bb": ask(
                    "Amount to call in BB", "2.5", float
                ),

                "stack_bb": ask(
                    "Remaining stack in BB", "100", float
                ),

                "position": ask(
                    "Position (UTG, HJ, CO, BTN, SB, BB)",
                    "BTN",
                    str
                ).upper(),

                "players_in_hand": ask(
                    "Players still in hand",
                    "3",
                    int
                ),

                "last_opp_action": ask(
                    "Last opponent action",
                    "raise",
                    str
                ).lower(),

                "raises_this_street": ask(
                    "Raises this street",
                    "1",
                    int
                ),

                "callers_this_street": ask(
                    "Callers this street",
                    "0",
                    int
                ),

                "players_yet_to_act": ask(
                    "Players yet to act after you",
                    "1",
                    int
                ),

                "in_position": ask(
                    "Are you in position? (1=yes, 0=no)",
                    "0",
                    int
                ),

                "is_street_aggressor": ask(
                    "Are you the street aggressor? (1=yes, 0=no)",
                    "0",
                    int
                ),

                "own_prev_action_street": ask(
                    "Your previous action this street",
                    "none",
                    str
                ).lower(),

            }

            if (
                row["pot_bb"] <= 0
                or row["to_call_bb"] < 0
                or row["stack_bb"] <= 0
            ):
                raise ValueError(
                    "Pot and stack must be positive; "
                    "amount to call cannot be negative"
                )

            if not 2 <= row["players_in_hand"] <= 9:
                raise ValueError(
                    "Players in hand must be between 2 and 9"
                )

            if row["to_call_bb"] > row["stack_bb"]:
                print(
                    "Note: amount to call exceeds stack."
                )

            predict(model, row)

        except (ValueError, KeyError) as exc:
            print(f"Invalid scenario: {exc}")

        except (EOFError, KeyboardInterrupt):
            print()
            break


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Historical Poker Action Model"
    )

    sub = parser.add_subparsers(
        dest="command"
    )

    train_parser = sub.add_parser("train")

    train_parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV
    )

    sub.add_parser("predict")

    sub.add_parser("evaluate")

    sub.add_parser("crossval")

    args = parser.parse_args()

    if args.command == "train":

        train(args.csv)

    elif args.command == "crossval":

        run_cross_validation(DEFAULT_CSV)

    elif args.command == "predict":

        if not MODEL_PATH.exists():

            print(
                "No saved model found. "
                "Training from CSV..."
            )

            train(DEFAULT_CSV)

        artifact = joblib.load(MODEL_PATH)

        interactive(artifact["model"])

    elif args.command == "evaluate":

        if not MODEL_PATH.exists():
            print("Train the model first.")
            return

        artifact = joblib.load(MODEL_PATH)

        df = pd.read_csv(artifact["csv"])

        print_metrics(
            artifact["model"],
            df[df.split == "val"],
            "VALIDATION"
        )

        print_metrics(
            artifact["model"],
            df[df.split == "test"],
            "HELD-OUT TEST"
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
