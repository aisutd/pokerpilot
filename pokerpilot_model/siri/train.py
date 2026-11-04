import argparse
import json
from pathlib import Path

from .model import train
from .run_reports import write_run_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate PokerPilot's action model")
    parser.add_argument("--data-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("pokerpilot_model/artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("pokerpilot_model/results"))
    parser.add_argument("--estimators", type=int, default=350)
    args = parser.parse_args()
    results = train(args.data_dir, args.output_dir, args.estimators)
    results["report_path"] = str(write_run_report(args.report_dir, "training", results))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
