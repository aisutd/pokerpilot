# DQN 5-fold cross-validation: quick results

Generated 2026-10-26. Total run time: under 2 minutes.

## Headline numbers

| | Offline DQN (CQL) | Adaptive DQN |
|---|---|---|
| **Accuracy on unseen hands** (picks the real player's action) | **76.9% ± 1.1%** | **76.3% ± 2.7%** |
| Accuracy of always folding (the bar to beat) | 62.2% | 62.2% |
| **Gain over the bar** | **+14.7 points** | **+14.1 points** |
| corr(predicted value, actual hand result) | 0.06 ± 0.05 | **0.14 ± 0.01** |
| Profitable at every table type in the simulator? | **Yes** (+12 to +405 bb/100) | **Yes** (+45 to +1,360 bb/100) |
| Beats the rule-based bot on the same cards? | No; worse at every table | No; even against the held-out-style table (−5 bb/100), behind elsewhere |

**In short:** in all 5 folds, both models predict real players' decisions much better than the trivial baseline (about 77% vs 62%). Accuracy holds steady across time periods (75–79% in every fold), so the model generalises to hands it never saw. In the poker simulator both models win money at every table type, but neither beats the hand-written rule-based bot yet. The adaptive model comes closest.

## How the test worked

- **5-fold cross-validation** over all 1,747 hands (12,940 decisions). The hands are split into 5 chronological blocks of about 350 hands each.
- **Each block is the test set once.** The next block chose the training epoch, and the remaining 3 blocks were the training data. No hand appears in two roles, and every decision is tested exactly once.
- **Every model was retrained from scratch on each fold,** so all numbers come from hands the model never trained on.

| model | how it was trained per fold |
|---|---|
| Offline DQN (CQL) | conservative double DQN on the fold's logged decisions, full 40 epochs (the best epoch was 17–37, picked on the validation block) |
| Adaptive DQN | starts from that fold's offline model, then plays **3,000** simulated hands (shortened from 150,000 to fit 5 minutes) against mixed tables and self-play, with 25% of each batch drawn from the logged data |

`± x` = standard deviation across the 5 folds.

## 1. Accuracy on held-out hands

"Accuracy" = how often the model's chosen action matches the action the real player took, on decisions from the held-out block.

| fold (test hands) | Offline DQN | Adaptive DQN | Always fold |
|---|---|---|---|
| 1 (G00001–G00350) | 75.5% | 71.8% | 61.3% |
| 2 (G00351–G00700) | 77.1% | 76.3% | 62.3% |
| 3 (G00701–G01049) | 76.4% | 77.0% | 60.5% |
| 4 (G01050–G01398) | 78.4% | 78.8% | 64.0% |
| 5 (G01399–G01747) | 77.3% | 77.9% | 62.6% |
| **Mean** | **76.9% ± 1.1%** | **76.3% ± 2.7%** | **62.2%** |

- **Both models beat the baseline in every fold,** by 11–16 points.
- **Offline is the most stable** (75.5–78.4%). It matches the original single-split result (77.9%), so that number wasn't a lucky split.
- **Adaptive is just as accurate in four of five folds.** Fold 1, the earliest block, is its weak spot (71.8%).

Other measures on the same held-out decisions:

| | Offline DQN | Adaptive DQN |
|---|---|---|
| corr(Q, actual result): does the predicted value track what really happened? | 0.059 ± 0.050 | **0.143 ± 0.011** |
| TD loss (lower is better) | **0.269 ± 0.036** | 0.324 ± 0.063 |
| Mean predicted value (bb) | +1.62 | +0.94 |
| Mean actual result (bb) | −0.17 | −0.17 |

- **The adaptive model's value estimates are more informative and far more consistent** (0.14 in every fold). The offline model's swing from −0.004 to 0.12.
- **Both models are optimistic** about their expected result (positive predicted value vs a slightly negative actual one), which is common for Q-learning.
- The correlations are low overall, which is normal: poker hand results are very noisy.

Action mix on held-out decisions:

| | fold | check | call | ½-pot raise | pot raise | all-in |
|---|---|---|---|---|---|---|
| Real players | 62.2% | 9.5% | 10.4% | 5.8% | 11.6% | 0.5% |
| Offline DQN | 68.5% | 11.8% | 10.6% | 2.5% | 6.1% | 0.4% |
| Adaptive DQN | 67.7% | 12.1% | 7.7% | 1.6% | 9.9% | 0.9% |

Both models fold a bit more than real players and raise a bit less, so their accuracy isn't just "always fold".

## 2. How well it played (simulator)

Average winnings in **big blinds per 100 hands** (bb/100), 6-player tables, 300 hands per table per fold, on test deals never used in training. `heldout styles` = opponents built from the held-out block's real players, whom no model trained against.

| player | random | calling station | rule-based | dataset styles | **held-out styles** |
|---|---|---|---|---|---|
| Random baseline | +191 | −1,617 | −500 | +24 | −415 |
| Calling-station baseline | +2,034 | −22 | −1,379 | −907 | −955 |
| Rule-based baseline | +653 | +1,648 | +282 | +170 | +50 |
| **Offline DQN** | +162 | +405 | **+45** | **+99** | **+12** |
| **Adaptive DQN** | +295 | **+1,360** | **+48** | **+108** | **+45** |

- **Both DQNs win money at all five table types,** averaged over the folds.
- **Adaptive beats offline at every table,** most clearly against calling stations (+1,360 vs +405). It learned to value-bet loose players.
- **Against the unseen held-out players, adaptive earns +45 bb/100,** close to the rule-based bot's +50.

### Head to head with the rule-based bot (same cards)

DQN winnings minus the rule-based bot's on identical deals, averaged over the folds:

| | random | calling station | rule-based | dataset styles | held-out styles |
|---|---|---|---|---|---|
| Offline DQN | −491 | −1,242 (worse in 5/5 folds) | −237 | −71 | −38 |
| Adaptive DQN | −357 | −288 | −234 | −62 | **−5** |

- **Neither DQN beats the rule-based bot yet.** The offline model is clearly behind against calling stations: significantly worse in all 5 folds.
- **The adaptive model is not significantly worse anywhere except one fold,** and it's even with the bot against the unseen held-out players.
- Tight vs loose play explains much of the gap. The offline model plays only 6–16% of hands, versus 17–41% for the rule-based bot, so it misses value against weak players. Adaptive loosens up to 56% of hands against calling stations, which is the right adjustment.

## Caveats

- **This is a shortened run.** The adaptive model trained for 3,000 simulated hands instead of 150,000, about 2% of normal. Its win rates will likely improve with full training. The original full-length adaptive model reached +28 bb/100 against the rule-based table on validation.
- **300 hands per table is a small sample.** Single-fold win rates carry 95% error bars of about ±200–1,000 bb/100, so the simulator numbers show direction, not precise values. The accuracy numbers are solid: they use every one of the 12,940 decisions.
- **Accuracy means "imitates the real players".** It doesn't mean the model plays optimally; the logged players were not perfect.
- The full-length k-fold (all four online variants, 150,000 hands each) can still be run with `python scripts/kfold.py train` followed by `evaluate`. It takes about 5–6 hours.

## Files

- Raw results: `runs/kfold_quick/results.json`
- Fold data: `data/kfold/fold1…fold5/` (built by `scripts/kfold.py prepare`)
- Per-fold training reports: `runs/outputs_archive/kfold/train_fold*_offline.md` and `runs/kfold_quick/reports/`
