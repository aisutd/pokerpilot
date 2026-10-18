"""Step 5: Evaluate held-out games and report bb/100 with a block-bootstrap CI."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from dqn_model_surya.env import (
    ACTION_COUNT,
    ACTION_NAMES,
    STATE_SIZE,
    PokerDQNEnv,
    load_style_distribution,
)
from dqn_model_surya.model import choose_greedy_action, load_policy


MINIMUM_HANDS = 10_000
BOOTSTRAP_SAMPLES = 10_000
BLOCK_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path(__file__).resolve().parent / 'checkpoints' / 'pokerpilot_dqn.pt',
    )
    parser.add_argument('--hands', type=int, default=MINIMUM_HANDS)
    parser.add_argument('--seed', type=int, default=2507)
    return parser.parse_args()


def evaluate(checkpoint: Path, hands: int, seed: int) -> tuple[float, float, float]:
    """Return measured bb/100 and a 95% game-block bootstrap confidence interval."""

    if hands < MINIMUM_HANDS:
        raise ValueError(f'Evaluation requires at least {MINIMUM_HANDS:,} hands.')
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f'DQN checkpoint not found: {checkpoint}. Train the agent before evaluating it.',
        )

    model, metadata = load_policy(str(checkpoint), checkpoint.stat().st_mtime_ns)
    if metadata['state_size'] != STATE_SIZE or metadata['action_count'] != ACTION_COUNT:
        raise ValueError('The checkpoint does not match the current state/action definitions.')
    if metadata.get('action_names') != list(ACTION_NAMES):
        raise ValueError('The checkpoint uses a different DQN action ordering.')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    outcomes: list[float] = []
    block_means: list[float] = []
    block_sizes: list[int] = []
    block_number = 0
    test_style_distribution = load_style_distribution('test')
    while len(outcomes) < hands:
        block_target = min(BLOCK_SIZE, hands - len(outcomes))
        env = PokerDQNEnv(
            seed=seed + block_number,
            style_split='test',
            style_distribution=test_style_distribution,
        )
        block_outcomes: list[float] = []
        while len(block_outcomes) < block_target:
            state, action_mask, reset_info = env.reset()
            skipped = reset_info['skipped_hand_rewards_bb']
            remaining = block_target - len(block_outcomes)
            block_outcomes.extend(skipped[:remaining])
            if len(block_outcomes) >= block_target:
                break

            done = False
            while not done:
                action, _ = choose_greedy_action(model, state, action_mask)
                state, action_mask, reward, done, _ = env.step(action)
            block_outcomes.append(reward)

        if block_outcomes:
            outcomes.extend(block_outcomes)
            block_means.append(float(np.mean(block_outcomes)))
            block_sizes.append(len(block_outcomes))
        block_number += 1

    bb_per_100 = 100 * float(np.mean(outcomes))
    bootstrap_rng = np.random.default_rng(seed)
    blocks = np.asarray(block_means, dtype=np.float64)
    sizes = np.asarray(block_sizes, dtype=np.float64)
    bootstrap_scores = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled = bootstrap_rng.integers(0, len(blocks), size=len(blocks))
        bootstrap_scores[sample_index] = 100 * np.average(
            blocks[sampled],
            weights=sizes[sampled],
        )
    low, high = np.percentile(bootstrap_scores, [2.5, 97.5])
    return bb_per_100, float(low), float(high)


def main() -> None:
    args = parse_args()
    result, low, high = evaluate(args.checkpoint, args.hands, args.seed)
    print(
        f'Evaluated {args.hands:,} held-out hands against test-split bot styles '
        f'(seed={args.seed}).\n'
        f'Net result: {result:.2f} bb/100 '
        f'(95% game-block bootstrap CI: [{low:.2f}, {high:.2f}]).',
    )


if __name__ == '__main__':
    main()
