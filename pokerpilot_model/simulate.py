"""Deterministic offline replay; this does not estimate counterfactual EV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .baseline import baseline
from .data import TARGET, feature_frame, load_splits
from .model import load_model
from .run_reports import write_run_report


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
        "action_report": classification_report(frame[TARGET], pred, labels=labels,
                                                output_dict=True, zero_division=0),
        "action_counts": pd.Series(pred).value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--model", type=Path, default=Path("pokerpilot_model/artifacts/model.joblib"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--report-dir", type=Path, default=Path("pokerpilot_model/results"))
    args = parser.parse_args()
    results = replay(args.data_dir, args.model, args.split)
    results["report_path"] = str(write_run_report(args.report_dir, args.split, results))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
