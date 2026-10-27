"""Build the offline DQN data from the PokerPilot dataset.

    python scripts/preprocess.py [--dataset ../Dataset] [--out data/processed]

Writes to ``--out``:
    transitions_{train,val,test}.npz   offline (s, a, r, s', done, masks)
    style_profiles.json                per-style action frequencies (train only)
    stacks_bb.json                     starting stacks seen in train hands
    preflop_equity.json                cached preflop equities (169 x 1..8)
    preprocess_report.json             what was checked and how it went
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokerdqn.actions import ACTION_NAMES  # noqa: E402
from pokerdqn.agents import build_style_profiles, save_profiles  # noqa: E402
from pokerdqn.cards import (build_preflop_table, hand_features, load_preflop_table,  # noqa: E402
                            parse_cards, save_preflop_table)
from pokerdqn.dataset import (OUTCOME_COLUMNS, REPLAY_COLUMNS, build_transitions,  # noqa: E402
                              find_dataset_dir, load_tables, replay_all)
from pokerdqn.features import STATE_DIM  # noqa: E402

CARD_COLUMNS = (
    'card_high_rank', 'card_low_rank', 'is_pocket_pair', 'is_suited', 'rank_gap',
    'chen_score', 'preflop_percentile', 'made_hand_rank', 'pair_type', 'flush_draw',
    'straight_draw', 'draw_outs', 'board_paired', 'board_suit_texture',
    'board_connectedness', 'board_high_rank',
)


def card_feature_agreement(decisions) -> dict:
    """Share of rows where our card features equal the dataset's columns."""

    agree = Counter()

    for _, row in decisions.iterrows():
        board = row['board'] if isinstance(row['board'], str) else ''
        ours = hand_features(parse_cards(row['hole_cards']), parse_cards(board))

        for column in CARD_COLUMNS:
            theirs = row[column]

            if not isinstance(theirs, str) and np.isnan(theirs):
                theirs = 'none' if column == 'board_suit_texture' else 0

            try:
                same = abs(float(theirs) - float(ours[column])) < 0.002
            except ValueError:
                same = str(theirs) == str(ours[column])

            agree[column] += same

    return {column: agree[column] / len(decisions) for column in CARD_COLUMNS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=None, help='folder with the CSVs')
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--reward', choices=('net', 'ev'), default='net',
                        help="'net' = actual hand result, 'ev' = all-in-adjusted result")
    parser.add_argument('--equity-samples', type=int, default=200)
    args = parser.parse_args()

    started = time.time()
    dataset_dir = args.dataset or find_dataset_dir(ROOT)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'dataset: {dataset_dir}')

    table_path = args.out / 'preflop_equity.json'

    if not load_preflop_table(table_path):
        print('computing preflop equity table (one-off, ~2 min)...')
        build_preflop_table(8)
        save_preflop_table(table_path)

    tables = load_tables(dataset_dir)
    d = tables.decisions
    print(f'{len(d)} decisions, {d.game_id.nunique()} hands, {len(tables.hands)} player-hands')

    print('replaying every hand through the engine...')
    replays = replay_all(tables)
    ok = [r for r in replays.values() if r.ok]
    mismatches = Counter()

    for r in ok:
        mismatches.update(r.mismatches)

    replayed = sum(len(r.steps) for r in ok)
    print(f'  {len(ok)}/{len(replays)} hands replayed, payoffs match in '
          f'{sum(r.payoffs_match for r in ok)}')

    print('building transitions...')
    sets = build_transitions(tables, replays, reward=args.reward,
                             equity_samples=args.equity_samples)
    suffix = '' if args.reward == 'net' else '_ev'

    for name, data in sets.items():
        data.save(args.out / f'transitions{suffix}_{name}.npz')

    train_rows = d[d['split'] == 'train']
    profiles = build_style_profiles(train_rows, tables.hands, sets['train'].actions)
    train_hands = tables.hands[tables.hands['split'] == 'train']
    weights = train_hands['play_style'].value_counts().to_dict()
    save_profiles({'profiles': profiles, 'weights': weights,
                   'source': 'train split decisions only'}, args.out / 'style_profiles.json')
    stacks = sorted(float(s) for s in train_hands['starting_stack_bb'])
    (args.out / 'stacks_bb.json').write_text(json.dumps(stacks))

    agreement = card_feature_agreement(d)
    all_equity = np.concatenate([sets[s].equity for s in ('train', 'val', 'test')])
    all_ids = np.concatenate([sets[s].decision_ids for s in ('train', 'val', 'test')])
    theirs = d.set_index('decision_id').loc[all_ids, 'equity_vs_random'].to_numpy(dtype=float)
    equity_corr = float(np.corrcoef(all_equity, theirs)[0, 1])
    equity_mae = float(np.mean(np.abs(all_equity - theirs)))

    report = {
        'dataset_dir': str(dataset_dir),
        'state_dim': STATE_DIM,
        'reward': args.reward,
        'counts': {
            'decisions': len(d), 'hands': int(d.game_id.nunique()),
            'players': int(tables.players.shape[0]),
            'decisions_by_split': d['split'].value_counts().to_dict(),
            'hands_by_split': d.groupby('split')['game_id'].nunique().to_dict(),
            'logged_actions': d['target_action'].value_counts().to_dict(),
            'streets': d['street'].value_counts().to_dict(),
        },
        'replay': {
            'hands_ok': len(ok), 'hands_total': len(replays),
            'payoffs_match': sum(r.payoffs_match for r in ok),
            'decisions_replayed': replayed,
            'context_mismatches': {c: mismatches.get(c, 0) for c in REPLAY_COLUMNS},
            'covered_shoves_mapped_to_call': sum(r.covered_shoves for r in ok),
            'errors': [r.error for r in replays.values() if not r.ok],
        },
        'card_feature_agreement': agreement,
        'equity_vs_dataset': {'pearson': equity_corr, 'mean_abs_diff': equity_mae},
        'transitions': {
            name: {
                'n': len(data),
                'terminal': int(data.dones.sum()),
                'actions': {ACTION_NAMES[a]: int(c) for a, c in
                            zip(*np.unique(data.actions, return_counts=True))},
                'mean_return_bb': float(data.returns.mean()),
                'mean_reward_terminal_bb': float(data.rewards[data.dones].mean()),
            }
            for name, data in sets.items()
        },
        'not_used_as_input': list(OUTCOME_COLUMNS) + [
            'target_*', 'player_* (acting player identity/history)', 'hole cards of others',
        ],
        'style_profile_rows': int(len(train_rows)),
        'seconds': round(time.time() - started, 1),
    }
    (args.out / 'preprocess_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('replay', 'equity_vs_dataset', 'transitions')}, indent=2))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
