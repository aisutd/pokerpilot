"""Evaluate DQN checkpoints against baselines.

    python scripts/evaluate.py --checkpoint runs/main/online_best.pt \
        --checkpoint runs/main/offline_best.pt --hands 10000 --out results

1. Simulator: every hero (the random, calling-station and rule-based
   baselines plus each checkpoint, greedy) plays ``--hands`` 6-max hands
   against each opponent lineup on the *same* deals (test seeds, distinct
   from the validation seeds used in training). Reports bb/100 with 95% CI,
   paired DQN-minus-baseline differences, VPIP/PFR, action distributions
   and cumulative-result curves.
2. Dataset (offline): on the held-out test split, the TD loss, how often
   the greedy action equals the logged one, the correlation between
   Q(s, logged action) and the realised hand result, and the greedy action
   distribution. These describe the Q-function; they are *not* a measure
   of winnings (the logged results come from the logged players' actions).
3. Calibration: how closely the dataset-style opponents reproduce the
   dataset's VPIP/PFR per play style.

Writes ``evaluation.json``, ``evaluation.md`` and ``curves_<lineup>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokerdqn.actions import ACTION_NAMES  # noqa: E402
from pokerdqn.agents import STYLES, StyleAgent  # noqa: E402
from pokerdqn.cards import load_preflop_table  # noqa: E402
from pokerdqn.dataset import TransitionSet, find_dataset_dir, load_tables  # noqa: E402
from pokerdqn.dqn import DQNAgent, DQNLearner  # noqa: E402
from pokerdqn.evaluation import (BASELINE_HEROES, LINEUPS, baseline_hero,  # noqa: E402
                                 load_style_data, paired_difference, run, summarize)

TEST_SEED = 50_000


def offline_metrics(learner: DQNLearner, data: TransitionSet) -> dict:
    loss = learner.evaluate_loss(data)
    q = learner.q_values(data.states)
    q_logged = q[np.arange(len(q)), data.actions]
    masked_q = np.where(data.masks, q, -np.inf)
    greedy = masked_q.argmax(axis=1)
    counts = np.bincount(greedy, minlength=len(ACTION_NAMES))
    logged = np.bincount(data.actions, minlength=len(ACTION_NAMES))
    return {
        'n': len(data),
        'td_loss': loss['td_loss'],
        'agreement_with_logged': float(np.mean(greedy == data.actions)),
        'pearson_q_logged_vs_return': float(np.corrcoef(q_logged, data.returns)[0, 1]),
        'mean_q_logged_bb': float(q_logged.mean()),
        'mean_return_bb': float(data.returns.mean()),
        'mean_q_greedy_bb': float(masked_q.max(axis=1).mean()),
        'greedy_action_distribution': {n: float(c / counts.sum()) for n, c in zip(ACTION_NAMES, counts)},
        'logged_action_distribution': {n: float(c / logged.sum()) for n, c in zip(ACTION_NAMES, logged)},
    }


def style_calibration(profiles, weights, stacks, hands: int, players) -> dict:
    by_style = players.groupby('play_style')[['vpip_pct', 'pfr_pct']].mean()
    out = {}

    for i, style in enumerate(s for s in STYLES if s in profiles):
        hero = StyleAgent(style, profiles, random.Random(i))
        summary = summarize(run(hero, 'dataset_styles', hands, TEST_SEED + 100 + i,
                                profiles, weights, stacks=stacks))
        out[style] = {
            'sim_vpip': summary['vpip'], 'data_vpip': float(by_style.loc[style, 'vpip_pct']),
            'sim_pfr': summary['pfr'], 'data_pfr': float(by_style.loc[style, 'pfr_pct']),
        }

    return out


def fmt(x: float, ci: float | None = None) -> str:
    return f'{x:+.1f}' if ci is None else f'{x:+.1f} ± {ci:.1f}'


def write_markdown(report: dict, path: Path) -> None:
    lines = ['# DQN evaluation', '']
    sim = report['simulator']
    lines += [f"Simulator: {report['hands']} hands per hero per lineup, 6-max, "
              f"test seeds {TEST_SEED}+, stacks sampled from the dataset. "
              'bb/100 with 95% confidence interval.', '']
    heroes = list(next(iter(sim.values()))['heroes'])
    lines.append('| hero | ' + ' | '.join(LINEUPS) + ' |')
    lines.append('|---|' + '---|' * len(LINEUPS))

    for hero in heroes:
        cells = [fmt(sim[l]['heroes'][hero]['bb_per_100'], sim[l]['heroes'][hero]['ci95_bb_per_100'])
                 for l in LINEUPS]
        lines.append(f'| {hero} | ' + ' | '.join(cells) + ' |')

    lines += ['', 'Paired difference (DQN minus baseline, same deals), bb/100 ± 95% CI:', '']
    lines.append('| DQN | vs baseline | ' + ' | '.join(LINEUPS) + ' |')
    lines.append('|---|---|' + '---|' * len(LINEUPS))

    for hero in heroes:
        if not hero.startswith('dqn'):
            continue

        for base in BASELINE_HEROES:
            pairs = [sim[l]['paired'][hero][base] for l in LINEUPS]
            cells = [fmt(p['bb_per_100'], p['ci95_bb_per_100']) for p in pairs]
            lines.append(f'| {hero} | {base} | ' + ' | '.join(cells) + ' |')

    lines += ['', 'Hero action distribution and VPIP / PFR (dataset_styles lineup):', '']
    lines.append('| hero | VPIP | PFR | ' + ' | '.join(ACTION_NAMES) + ' |')
    lines.append('|---|---|---|' + '---|' * len(ACTION_NAMES))

    for hero in heroes:
        h = sim['dataset_styles']['heroes'][hero]
        lines.append(f"| {hero} | {h['vpip']:.2f} | {h['pfr']:.2f} | " + ' | '.join(
            f"{h['action_distribution'][a]:.2f}" for a in ACTION_NAMES) + ' |')

    if report.get('offline'):
        lines += ['', 'Dataset test split (Q-function diagnostics, not winnings):', '']
        lines.append('| checkpoint | TD loss | agree w/ logged | corr(Q, return) | mean Q logged | mean return |')
        lines.append('|---|---|---|---|---|---|')

        for name, m in report['offline'].items():
            t = m['test']
            lines.append(f"| {name} | {t['td_loss']:.4f} | {t['agreement_with_logged']:.3f} | "
                         f"{t['pearson_q_logged_vs_return']:.3f} | {t['mean_q_logged_bb']:+.2f} | "
                         f"{t['mean_return_bb']:+.2f} |")

    if report.get('style_calibration'):
        lines += ['', 'Dataset-style opponents: simulated vs logged VPIP / PFR:', '']
        lines.append('| style | sim VPIP | data VPIP | sim PFR | data PFR |')
        lines.append('|---|---|---|---|---|')

        for style, c in report['style_calibration'].items():
            lines.append(f"| {style} | {c['sim_vpip']:.2f} | {c['data_vpip']:.2f} | "
                         f"{c['sim_pfr']:.2f} | {c['data_pfr']:.2f} |")

    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, action='append', default=[],
                        help='DQN checkpoint (repeatable)')
    parser.add_argument('--hands', type=int, default=10_000)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    parser.add_argument('--calibration-hands', type=int, default=3_000)
    parser.add_argument('--skip-calibration', action='store_true')
    parser.add_argument('--threads', type=int, default=4, help='torch CPU threads')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    load_preflop_table(args.data / 'preflop_equity.json')
    profiles, weights = load_style_data(args.data / 'style_profiles.json')
    stacks = json.loads((args.data / 'stacks_bb.json').read_text())

    heroes = {name: (lambda name=name: baseline_hero(name, seed=1)) for name in BASELINE_HEROES}
    learners = {}

    for path in args.checkpoint:
        name = f'dqn:{path.parent.name}/{path.stem}'
        learners[name] = DQNLearner.load(path)
        heroes[name] = lambda learner=learners[name]: DQNAgent(learner, epsilon=0.0)

    report = {'hands': args.hands, 'test_seed': TEST_SEED, 'simulator': {}}

    for i, lineup in enumerate(LINEUPS):
        print(f'lineup {lineup}:')
        raw = {}

        for name, make in heroes.items():
            raw[name] = run(make(), lineup, args.hands, TEST_SEED + i, profiles, weights,
                            stacks=stacks)
            s = summarize(raw[name])
            print(f"  {name:32s} {s['bb_per_100']:+8.1f} ± {s['ci95_bb_per_100']:.1f} bb/100")

        paired = {
            hero: {base: paired_difference(raw[hero], raw[base]) for base in BASELINE_HEROES}
            for hero in raw if hero.startswith('dqn')
        }
        report['simulator'][lineup] = {
            'heroes': {name: summarize(results) for name, results in raw.items()},
            'paired': paired,
        }

        with (args.out / f'curves_{lineup}.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['hand'] + list(raw))
            curves = {name: np.cumsum([r.net_bb for r in results]) for name, results in raw.items()}

            for h in range(args.hands):
                writer.writerow([h + 1] + [round(float(curves[name][h]), 2) for name in raw])

    if learners:
        report['offline'] = {}
        sets = {s: TransitionSet.load(args.data / f'transitions_{s}.npz') for s in ('val', 'test')}

        for name, learner in learners.items():
            report['offline'][name] = {s: offline_metrics(learner, d) for s, d in sets.items()}

    if not args.skip_calibration:
        players = load_tables(find_dataset_dir(ROOT)).players
        report['style_calibration'] = style_calibration(profiles, weights, stacks,
                                                        args.calibration_hands, players)

    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2))
    write_markdown(report, args.out / 'evaluation.md')
    print((args.out / 'evaluation.md').read_text())


if __name__ == '__main__':
    main()
