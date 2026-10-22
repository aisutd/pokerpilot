"""Deterministic offline replay; this does not estimate counterfactual EV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .data import TARGET, feature_frame, load_splits
from .model import load_model


def baseline(frame: pd.DataFrame) -> np.ndarray:
    action = []
    equity = pd.to_numeric(frame.equity_vs_random, errors="coerce").fillna(0.5)
    odds = pd.to_numeric(frame.pot_odds, errors="coerce").fillna(0)
    facing = pd.to_numeric(frame.facing_bet, errors="coerce").fillna(0).astype(bool)
    for e, o, f in zip(equity, odds, facing):
        action.append("call" if f and e >= o else "fold" if f else "bet" if e >= 0.68 else "check")
    return np.array(action)


def replay(data_dir: Path, model_path: Path, split: str) -> dict:
    frame = load_splits(data_dir)[split]
    artifact = load_model(model_path)
    pred = [artifact["labels"][int(index)] for index in artifact["pipeline"].predict(
        feature_frame(frame, artifact["columns"]))]
    base = baseline(frame)
    labels = artifact["labels"]
    return {
        "split": split, "rows": len(frame),
        "model_accuracy": float(accuracy_score(frame[TARGET], pred)),
        "model_macro_f1": float(f1_score(frame[TARGET], pred, labels=labels, average="macro", zero_division=0)),
        "baseline_accuracy": float(accuracy_score(frame[TARGET], base)),
        "baseline_macro_f1": float(f1_score(frame[TARGET], base, labels=labels, average="macro", zero_division=0)),
        "action_counts": pd.Series(pred).value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--model", type=Path, default=Path("pokerpilot_model/artifacts/model.joblib"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    print(json.dumps(replay(args.data_dir, args.model, args.split), indent=2))


if __name__ == "__main__":
    main()
