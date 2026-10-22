import argparse
import json
from pathlib import Path

from .model import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate PokerPilot's action model")
    parser.add_argument("--data-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("pokerpilot_model/artifacts"))
    parser.add_argument("--estimators", type=int, default=350)
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.output_dir, args.estimators), indent=2))


if __name__ == "__main__":
    main()
