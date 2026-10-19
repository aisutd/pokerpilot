# PokerPilot DQN

Deep Q-Network experiments for PokerPilot's No-Limit Texas Hold'em decision
making. This folder is self-contained: it has its own virtual environment,
and it does not import from or modify the website (`PokerPilot-Website/`)
or other team models. It reads the team dataset in `../Dataset/`.

The pipeline:

1. **Preprocess** the dataset: replay every logged hand through the
   simulator's engine to validate the state definitions, encode each
   decision as an 87-dim state vector, and build offline transitions
   `(s, a, r, s', done)` plus opponent profiles.
2. **Train offline** (optional): conservative double DQN (CQL) on the
   logged transitions.
3. **Train online**: double DQN in a 6-max simulator against random,
   calling-station, rule-based and dataset-calibrated opponents, with a
   replay buffer, target network and epsilon-greedy exploration.
4. **Evaluate** the greedy policy against the baselines on identical
   deals, and score the Q-function on the dataset's held-out test split.

## Quick start

Python 3.11+ (developed on 3.14). From `DQN/`:

```bash
# Windows (Git Bash)
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
PY=.venv/Scripts/python

# macOS / Linux
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PY=.venv/bin/python

$PY scripts/preprocess.py                    # ~6 min; writes data/processed/
$PY scripts/train.py --stage both --out runs/offline_online      # offline + online
$PY scripts/train.py --stage online --out runs/scratch           # online only
$PY scripts/evaluate.py --hands 10000 \
    --checkpoint runs/offline_online/offline_best.pt \
    --checkpoint runs/offline_online/online_best.pt \
    --checkpoint runs/scratch/online_best.pt                    # writes results/
$PY -W error::UserWarning -m unittest discover -s tests -t .    # all tests (~5 min)
```

`--help` on each script lists every option (network size, learning rates,
CQL weight, number of hands, epsilon schedule, ...). Training is CPU-only;
`--threads` sets torch's thread count (4 was fastest here; the default of
one thread per core was about 4x slower).

## What is in the dataset

`../Dataset/` (identical to `Dataset/` on the team repo's
`PokerEngine-frontend` branch) holds 8 CSVs:

| file | rows | content |
|---|---|---|
| `poker_decisions_all.csv` | 12,940 | one row per decision: context, hand-strength features, opponent statistics, the logged action (`target_*`) and the hand result (`outcome_*`) |
| `poker_decisions_{train,val,test}.csv` | 9,055 / 2,023 / 1,862 | the same rows split chronologically by hand (games G00001-G01223 / G01224-G01485 / G01486-G01747) |
| `poker_games.csv` | 1,747 | one row per hand: board, pot, winner |
| `poker_hands_by_player.csv` | 10,000 | one row per player per hand: seat, starting stack, hole cards, per-street action strings, net result, play style |
| `poker_player_summary.csv` | 20 | per player: play style, VPIP/PFR, results |
| `poker_feature_dictionary.csv` | 81 | column descriptions and whether each may be a model input |

Things worth knowing:

* 20 players, each labelled with one of 7 play styles (Tight-Aggressive,
  Balanced Reg, Loose-Aggressive, Loose-Passive, Rock, Maniac,
  Tight-Passive). Tables have 2-9 players; stakes are 25NL-600NL, but all
  amounts are also given in big blinds (blinds are always 0.5 / 1 bb).
* 81% of decisions are preflop and 62% are folds. Postflop there are only
  2,504 decisions (1,156 flop, 824 turn, 524 river).
* The data is internally consistent: hand results are zero-sum (no rake),
  each player's `outcome_hand_net_bb` equals its `net_bb` in
  `poker_hands_by_player.csv`, and every call equals the amount owed.
* The feature dictionary marks `outcome_*` as leaking. Opponent statistics
  (`opp_vpip`, ...) use earlier games only.
* Logged bet and raise sizes, measured as `(chips - to_call) / (pot + to_call)`,
  sit in 0.34-1.18 (preflop mostly 0.6-0.86). Apart from all-ins nobody
  overbets.
* The dates (July 2026), player names and the clean style labels suggest the
  data is generated rather than taken from real play. That is an inference;
  the dataset does not say.

## Design

### State (87 floats, `pokerdqn/features.py`)

Built only from what the acting player knows when deciding:

* street, position (SB/BB/BTN/CO/HJ/LJ/early), players dealt / in the hand /
  still to act, in position
* pot, amount to call, own and effective stack, pot odds, bet faced as a
  fraction of the pot, stack-to-pot ratio (log-scaled where unbounded)
* betting this street: raises, callers since the last raise, who is the
  aggressor; aggressor on an earlier street; own previous action; last action
* own cards: ranks, pair, suited, gap, Chen score, preflop percentile;
  made-hand category, pair type, flush/straight draws and outs; board
  pairing, suit texture, connectedness, high card
* Monte Carlo equity against random hands (200 samples postflop; a cached
  1,500-sample table preflop)
* main opponent's VPIP, PFR, aggression factor, fold-to-bet and hands seen,
  table average VPIP, the opponent's stack. A missing statistic is encoded
  with a flag.

**Never used as input:** `outcome_*`, `target_*`, other players' hole
cards, unrevealed board cards, player identities/names, and the acting
player's own historical statistics (`player_*`). Including those would let
offline Q-learning tell *who* is acting, which is the behaviour policy, not
the state.

The card features are recomputed from the hole cards and board rather than
read from the CSV, so they are computed the same way for dataset rows and
simulator states. They match the dataset's columns on 99.96-100% of rows.
The only differences are 7 hands that already hold a straight or flush,
where the dataset still counts a straight draw. Our equity correlates 0.979
with the dataset's `equity_vs_random` (mean absolute difference 0.032).

### Actions (`pokerdqn/actions.py`)

Six discrete actions, masked by legality everywhere (acting, bootstrapping,
CQL):

| id | action | legal when |
|---|---|---|
| 0 | fold | facing a bet (folding when checking is free is excluded) |
| 1 | check | nothing owed |
| 2 | call | facing a bet (for less than the stack, or all-in) |
| 3 | raise_half | bet/raise by ½ pot: `to = bet + to_call + 0.5 (pot + to_call)`, clamped to the minimum raise |
| 4 | raise_pot | the same with 1 × pot |
| 5 | all_in | raising is allowed |

A size that would use the whole stack is the same as all-in and is masked.
Logged bets/raises map to the nearest size (split at 0.75 pot); a logged
all-in that doesn't exceed the amount owed is a call.

### Reward and transitions

An episode is one hand. The reward is the hero's net chips for the hand in
big blinds (blinds included), paid on the hero's last decision; earlier
steps get 0. With γ = 1 the return is exactly the hand's chip result.

**Dataset transitions** (`pokerdqn/dataset.py`): for each logged decision,
`s'` is the same player's next decision in the same hand. If there is none,
the transition is terminal and its reward is `outcome_hand_net_bb` (or
`outcome_ev_net_bb` with `--reward ev`). This gives 12,940 transitions
(9,677 terminal). They are **behaviour-policy** data: the actions are the
logged players' choices, and the agent never "chose" them. They are used
only off-policy, in offline Q-learning. No transition is made up: an
action that wasn't logged has no reward, so its Q-value is not observed.
CQL keeps those unobserved Q-values from being inflated.

**Replay validation:** every hand is replayed through the simulator's
engine using the dataset's seats, starting stacks, hole cards, board and
logged amounts. All 1,747 hands replay. Every player's final result matches
the dataset. All 16 replayed context columns (pot, to call, stacks, players
still to act, aggressor flags, last action, ...) match on all 12,940
decisions; `opp_stack_bb` differs on 3, where "main opponent" is ambiguous.
The replay also supplies each decision's exact legal-action mask. Seven
logged "all-ins" over an opponent who was already all-in for less are
really calls (the excess can never be called, and pokerkit refuses the
raise), so they are labelled as calls.

### Simulator (`pokerdqn/engine.py`, `pokerdqn/env.py`)

pokerkit, the website's rules engine, spends about 50 ms per multiway
showdown evaluating hands. That is too slow for RL, so `engine.py` is a
small integer-chip NLHE engine (blinds, min-raise, short all-in rules,
side pots) checked against pokerkit in `tests/test_engine.py`. On 1,500
random 2-9 player hands with mixed stack depths it gives the same actor,
amount to call, raise range and stacks after every action, and the same
payoffs. The one exception is which tied winner gets an odd chip
(0.01 bb).

`PokerEnv` is a gym-style environment (`reset()` / `step(a)`) for one
learning seat at a 6-max table:

* The button moves every hand, so the hero plays every position.
* Stacks are drawn from the starting stacks seen in the training hands
  (median 100 bb, 10-90%: 62-189 bb).
* Opponents persist for 200-hand sessions. Their VPIP/PFR/... statistics
  accumulate from public information only, as in the dataset.
* Deals, stacks and opponent draws come from separate random streams, so
  different heroes with the same seed play identical deals.

### Opponents and baselines (`pokerdqn/agents.py`)

* `random`: uniform over legal actions
* `calling_station`: always check/call
* `rule_based`: tight-aggressive rules of thumb (preflop percentile
  thresholds by number of raises; postflop made-hand buckets and pot odds)
* `dataset_styles`: one bot per play style. Each samples actions from that
  style's logged action frequencies in the **training split**, conditioned
  on street, bet faced, raises this street and a 5-level hand-strength
  bucket, with smoothing towards coarser situations. Lineups are drawn by
  how often each style appears in the data.

Training uses a per-seat mixture: 15% random, 10% calling station, 25%
rule-based, 50% dataset styles.

### DQN (`pokerdqn/dqn.py`)

* MLP 87 → 256 → 256 with a dueling head (6 Q-values); Adam; gradient
  clipping 10; Huber loss with δ = 100 bb. Rewards are divided by 20
  inside the learner.
* Double DQN targets: the online net picks the next action, and a Polyak
  target net (τ = 0.005) evaluates it, with masks applied.
* Experience replay: a 200k circular buffer, 5k warm-up, one batch of 256
  every 2 hero decisions.
* Epsilon-greedy over legal actions: 1.0 → 0.05 over the first 40% of hands
  from scratch, or 0.3 → 0.05 when starting from offline weights.
* Offline: the same learner plus the CQL penalty
  `α (logsumexp_legal Q(s,·) − Q(s, a_logged))` with α = 1, 40 epochs. The
  epoch is chosen by validation-split loss.
* Online checkpoints are chosen by mean bb/100 over the four lineups on
  **validation seeds** (2,000 hands each). `evaluate.py` reports on
  different **test seeds**.

RESULTS_PLACEHOLDER

## Tests

```bash
$PY -W error::UserWarning -m unittest discover -s tests -t .
```

* `test_engine.py`: engine vs pokerkit, action by action, on 1,500 random
  hands; side pots; returned uncalled bets
* `test_cards.py`: evaluator vs pokerkit's `StandardHighHand`; card
  features vs the dataset's columns; known equities
* `test_dataset.py`: every hand replays with matching results; simulator
  context = dataset columns; logged actions are legal; rewriting every
  `outcome_*`/`target_*` column leaves all states unchanged (no leakage);
  rewards only at terminal steps; `s'` is the same player's next decision;
  splits are disjoint and chronological
* `test_env.py`: action masks and sizes; episode return = chips won;
  zero-sum; the hero sees no other hole cards or future board cards;
  identical deals across heroes; button rotation; opponent statistics
* `test_dqn.py`: replay buffer; masked bootstrapping learns a two-step
  problem in which an illegal action would pay more; CQL lowers unseen
  actions; checkpoint save/load

## Layout

```
DQN/
  pokerdqn/        library: cards, engine, actions, features, stats, dataset,
                   agents, env, dqn, evaluation
  scripts/         preprocess.py, train.py, evaluate.py
  tests/           unittest suites
  data/processed/  preprocessing output (git-ignored; regenerate)
  runs/            checkpoints and training logs (git-ignored)
  results/         evaluation.md / evaluation.json / curves_*.csv
```
