"""Train the DQN.

    python scripts/train.py --stage both --out runs/main

Stages:
    offline  conservative double DQN on the dataset's logged transitions
             (train split; the val split picks the best epoch)
    online   double DQN in the simulator against a mixture of random,
             calling-station, rule-based and dataset-calibrated opponents,
             with experience replay, a target network and epsilon-greedy
             exploration; checkpoints are compared on *validation* seeds
    both     offline, then online starting from the best offline weights

Outputs in ``--out``: checkpoints (*.pt), metrics_offline.csv,
metrics_online.csv, evals_online.json and config.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokerdqn.actions import ACTION_NAMES, NUM_ACTIONS  # noqa: E402
from pokerdqn.cards import load_preflop_table  # noqa: E402
from pokerdqn.dataset import TransitionSet  # noqa: E402
from pokerdqn.dqn import (DQNAgent, DQNConfig, DQNLearner, EpsilonSchedule,  # noqa: E402
                          ReplayBuffer)
from pokerdqn.env import EnvConfig, PokerEnv  # noqa: E402
from pokerdqn.evaluation import LINEUPS, lineup_factory, load_style_data, run, summarize  # noqa: E402
from pokerdqn.features import STATE_DIM  # noqa: E402

VALIDATION_SEED = 20_000  # checkpoint selection; evaluate.py uses other seeds


class CsvLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer = None
        self.file = None

    def write(self, row: dict) -> None:
        if self.writer is None:
            self.file = self.path.open('w', newline='')
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()

        self.writer.writerow(row)
        self.file.flush()


def policy_agreement(learner: DQNLearner, data: TransitionSet) -> float:
    q = learner.q_values(data.states)
    q[~data.masks] = -np.inf
    return float(np.mean(q.argmax(axis=1) == data.actions))


def train_offline(args, out: Path) -> Path:
    data_dir = args.data
    train = TransitionSet.load(data_dir / f'transitions{args.suffix}_train.npz')
    val = TransitionSet.load(data_dir / f'transitions{args.suffix}_val.npz')
    config = DQNConfig(state_dim=STATE_DIM, lr=args.offline_lr, cql_alpha=args.cql_alpha,
                       hidden=tuple(args.hidden), batch_size=args.batch_size,
                       weight_decay=args.weight_decay)
    learner = DQNLearner(config, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    log = CsvLog(out / 'metrics_offline.csv')
    best = math.inf
    best_path = out / 'offline_best.pt'
    n = len(train)
    print(f'offline: {n} train / {len(val)} val transitions, CQL alpha {args.cql_alpha}')

    for epoch in range(1, args.offline_epochs + 1):
        order = rng.permutation(n)
        infos = []

        for start in range(0, n, config.batch_size):
            idx = order[start:start + config.batch_size]
            infos.append(learner.update({
                'states': train.states[idx], 'actions': train.actions[idx],
                'rewards': train.rewards[idx], 'next_states': train.next_states[idx],
                'dones': train.dones[idx], 'masks': train.masks[idx],
                'next_masks': train.next_masks[idx],
            }))

        v = learner.evaluate_loss(val)
        row = {
            'epoch': epoch,
            'train_loss': np.mean([i['loss'] for i in infos]),
            'train_td_loss': np.mean([i['td_loss'] for i in infos]),
            'val_loss': v['loss'],
            'val_td_loss': v['td_loss'],
            'val_cql_gap': v.get('cql_gap', 0.0),
            'val_q_mean_bb': v['q_mean'],
            'val_agreement_with_logged': policy_agreement(learner, val),
        }
        log.write(row)
        learner.save(out / 'offline_last.pt', {'stage': 'offline', 'epoch': epoch})

        if v['loss'] < best:
            best = v['loss']
            learner.save(best_path, {'stage': 'offline', 'epoch': epoch, 'val': v})

        if epoch == 1 or epoch % 5 == 0 or epoch == args.offline_epochs:
            print('  ' + ' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                                  for k, v in row.items()))

    return best_path


def evaluate_checkpoint(learner, profiles, weights, stacks, hands, seed) -> dict:
    agent = DQNAgent(learner, epsilon=0.0)
    out = {}

    for i, lineup in enumerate(LINEUPS):
        out[lineup] = summarize(run(agent, lineup, hands, seed + i, profiles, weights,
                                    stacks=stacks))

    out['mean_bb_per_100'] = float(np.mean([out[k]['bb_per_100'] for k in LINEUPS]))
    return out


def train_online(args, out: Path, init: Path | None) -> None:
    profiles, weights = load_style_data(args.data / 'style_profiles.json')
    stacks = json.loads((args.data / 'stacks_bb.json').read_text())

    if init:
        learner = DQNLearner.load(init, cql_alpha=0.0, lr=args.lr)
        print(f'online: starting from {init}')
    else:
        learner = DQNLearner(DQNConfig(state_dim=STATE_DIM, lr=args.lr, hidden=tuple(args.hidden),
                                       batch_size=args.batch_size), seed=args.seed)
        print('online: starting from random weights')

    env = PokerEnv(lineup_factory('training_mix', profiles, weights),
                   EnvConfig(num_players=6, stack_bb=stacks, seed=args.seed,
                             session_hands=args.session_hands))
    buffer = ReplayBuffer(args.buffer, STATE_DIM, seed=args.seed)
    start_eps = args.eps_start if args.eps_start is not None else (0.3 if init else 1.0)
    epsilon = EpsilonSchedule(start_eps, args.eps_end, int(args.hands * args.eps_fraction))
    agent = DQNAgent(learner, epsilon=start_eps, rng=random.Random(args.seed))
    log = CsvLog(out / 'metrics_online.csv')
    evals = []
    recent = deque(maxlen=2000)
    infos = deque(maxlen=500)
    actions = np.zeros(NUM_ACTIONS)
    steps = 0
    seen = 0
    next_log = args.log_every
    next_eval = args.eval_every
    best = -math.inf
    started = time.time()

    while len(env.results) < args.hands:
        obs = env.reset()
        done = False

        while not done:
            agent.epsilon = epsilon(len(env.results))
            action = agent.act(obs)
            state, mask = obs.vector(), obs.mask
            next_obs, reward, done, _ = env.step(action)
            buffer.add(state, action, reward,
                       None if done else next_obs.vector(), done, mask,
                       None if done else next_obs.mask)
            actions[action] += 1
            obs = next_obs
            steps += 1

            if len(buffer) >= args.warmup and steps % args.train_every == 0:
                infos.append(learner.update(buffer.sample(learner.config.batch_size)))

        hands = len(env.results)
        recent.extend(r.net_bb for r in env.results[seen:])
        seen = hands

        if hands >= next_log:
            next_log += args.log_every
            row = {
                'hands': hands, 'hero_decisions': steps, 'updates': learner.updates,
                'epsilon': round(agent.epsilon, 4),
                'loss': float(np.mean([i['loss'] for i in infos])) if infos else float('nan'),
                'q_mean_bb': float(np.mean([i['q_mean'] for i in infos])) if infos else float('nan'),
                'train_bb_per_100_recent': float(np.mean(recent) * 100),
                'seconds': round(time.time() - started, 1),
            }
            row.update({f'act_{n}': actions[i] / max(actions.sum(), 1)
                        for i, n in enumerate(ACTION_NAMES)})
            actions[:] = 0
            log.write(row)
            print(f"  hands {hands} eps {row['epsilon']:.3f} loss {row['loss']:.4f} "
                  f"q {row['q_mean_bb']:.2f} recent {row['train_bb_per_100_recent']:.1f} bb/100 "
                  f"[{row['seconds']:.0f}s]")

        if hands >= next_eval or hands >= args.hands:
            next_eval += args.eval_every
            result = evaluate_checkpoint(learner, profiles, weights, stacks,
                                         args.eval_hands, VALIDATION_SEED)
            result['hands_trained'] = hands
            evals.append(result)
            (out / 'evals_online.json').write_text(json.dumps(evals, indent=1))
            path = out / f'online_{hands}.pt'
            learner.save(path, {'stage': 'online', 'hands': hands, 'validation': result})
            print('  validation: ' + ', '.join(
                f"{k} {result[k]['bb_per_100']:+.1f}" for k in LINEUPS) +
                f" | mean {result['mean_bb_per_100']:+.1f} bb/100")

            if result['mean_bb_per_100'] > best:
                best = result['mean_bb_per_100']
                learner.save(out / 'online_best.pt', {'stage': 'online', 'hands': hands,
                                                      'validation': result})

    learner.save(out / 'online_last.pt', {'stage': 'online', 'hands': len(env.results)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('offline', 'online', 'both'), default='both')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs' / 'main')
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--reward', choices=('net', 'ev'), default='net')
    parser.add_argument('--init', type=Path, default=None, help='checkpoint to start online from')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--batch-size', type=int, default=256)
    # offline
    parser.add_argument('--offline-epochs', type=int, default=40)
    parser.add_argument('--offline-lr', type=float, default=3e-4)
    parser.add_argument('--cql-alpha', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    # online
    parser.add_argument('--hands', type=int, default=150_000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--buffer', type=int, default=200_000)
    parser.add_argument('--warmup', type=int, default=5_000)
    parser.add_argument('--train-every', type=int, default=2,
                        help='hero decisions per gradient update')
    parser.add_argument('--eps-start', type=float, default=None)
    parser.add_argument('--eps-end', type=float, default=0.05)
    parser.add_argument('--eps-fraction', type=float, default=0.4)
    parser.add_argument('--session-hands', type=int, default=200)
    parser.add_argument('--log-every', type=int, default=5_000)
    parser.add_argument('--eval-every', type=int, default=25_000)
    parser.add_argument('--eval-hands', type=int, default=2_000)
    parser.add_argument('--threads', type=int, default=4, help='torch CPU threads')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.suffix = '' if args.reward == 'net' else '_ev'
    args.out.mkdir(parents=True, exist_ok=True)

    if not (args.data / f'transitions{args.suffix}_train.npz').exists():
        sys.exit(f'{args.data} has no processed data; run scripts/preprocess.py first')

    load_preflop_table(args.data / 'preflop_equity.json')
    (args.out / 'config.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    init = args.init

    if args.stage in ('offline', 'both'):
        init = train_offline(args, args.out)

    if args.stage in ('online', 'both'):
        train_online(args, args.out, init if args.stage == 'both' or args.init else None)


if __name__ == '__main__':
    main()
