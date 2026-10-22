"""Run seeded complete hands with the trained classifier in the website engine."""

import argparse
import json
import random
from pathlib import Path


def run(hands: int, seed: int, model: Path) -> dict:
    from engine.game import HUMAN_SEAT, Table
    from .engine_adapter import ModelBot

    table = Table("easy", seed=seed)
    for seat, player in enumerate(table.seats):
        player.bot = ModelBot(str(model), random.Random(seed + seat))
    completed = 0
    while completed < hands:
        table.run_bots()
        if table.hand_over:
            completed += 1
            if completed < hands and not table.game_over:
                table.start_hand()
            else:
                break
        elif table.actor_seat == HUMAN_SEAT:
            table.human_action(*table.seats[HUMAN_SEAT].bot.decide(table.bot_view(HUMAN_SEAT)))
    return {"requested_hands": hands, "completed_hands": completed, "seed": seed,
            "stacks_cents": [seat.stack for seat in table.seats]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", type=Path, default=Path("pokerpilot_model/artifacts/model.joblib"))
    args = parser.parse_args()
    print(json.dumps(run(args.hands, args.seed, args.model), indent=2))


if __name__ == "__main__":
    main()
