# PokerPilot XGBoost Model Report

_Generated 2026-10-22 20:20_

## Cross-validation

5-fold expanding-window chronological CV grouped by game_id

Folds are used for early stopping and hyperparameter selection; final test is separate. CV scores are tuning estimates, not an unbiased test.

| fold | trial | accuracy | macro_f1 | log_loss | best_iteration | train_rows | validation_rows |
|---|---|---|---|---|---|---|---|
| 1 | 30 | 0.736 | 0.62 | 0.627 | 179 | 1834 | 773 |
| 2 | 30 | 0.773 | 0.676 | 0.568 | 302 | 3703 | 949 |
| 3 | 30 | 0.768 | 0.668 | 0.538 | 314 | 5575 | 1120 |
| 4 | 30 | 0.812 | 0.662 | 0.534 | 277 | 7433 | 872 |
| 5 | 30 | 0.805 | 0.701 | 0.501 | 472 | 9181 | 959 |

## Action model: test set results

Scored on held-out test decisions made by winning players (P03, P04, P05, P11, P12, P15, P16, P17, P18, P19). Classes: fold, check, call, bet, raise.

| metric | model (winners) | model (all players) | always fold | fold/check rule |
|---|---|---|---|---|
| accuracy | 0.812 | 0.803 | 0.631 | 0.72 |
| macro F1 | 0.719 | 0.706 | - | 0.326 |
| log loss | 0.469 | 0.487 | - | - |

## Per-class report and confusion matrix

Confusion matrix rows = what the player did, columns = model's pick.

```
              precision    recall  f1-score   support

        fold      0.929     0.907     0.918       592
       check      0.890     0.783     0.833        83
        call      0.414     0.539     0.468       102
         bet      0.679     0.826     0.745        46
       raise      0.684     0.583     0.629       115

    accuracy                          0.812       938
   macro avg      0.719     0.728     0.719       938
weighted avg      0.827     0.812     0.818       938

       fold  check  call  bet  raise
fold    537      0    48    0      7
check     0     65     0   18      0
call     23      0    55    0     24
bet       0      8     0   38      0
raise    18      0    30    0     67
```

## Accuracy by street

| street | rows | accuracy | macro_f1 |
|---|---|---|---|
| Preflop | 769 | 0.823 | 0.729 |
| Flop | 84 | 0.75 | 0.597 |
| Turn | 50 | 0.68 | 0.56 |
| River | 35 | 0.914 | 0.755 |

## Top 12 features (share of total gain)

| feature | gain_share |
|---|---|
| facing_bet | 0.203 |
| to_call_bb | 0.086 |
| board_connectedness | 0.085 |
| equity_minus_pot_odds | 0.069 |
| facing_bet_pot_frac | 0.068 |
| last_opp_action | 0.046 |
| pot_odds | 0.045 |
| chen_score | 0.042 |
| preflop_percentile | 0.037 |
| to_call_frac_stack | 0.027 |
| street_idx | 0.027 |
| pot_bb | 0.023 |

## Sizing model (bet/raise size as fraction of pot)

- MAE on test: **0.167** pot
- Baseline (always median size): 0.440 pot

## Best hyperparameters

Final model uses **310** trees.

| parameter | value |
|---|---|
| max_depth | 3 |
| learning_rate | 0.03 |
| min_child_weight | 10 |
| subsample | 1 |
| colsample_bytree | 0.85 |
| reg_lambda | 1 |
| reg_alpha | 0.1 |
| gamma | 0.1 |
| weight_power | 0.5 |

## Player skill weights

Luck-adjusted EV bb/100 on train + val hands, and the training weight each player got.

| player_id | ev_bb_per_100 | weight |
|---|---|---|
| P07 | -98.1 | 0.05 |
| P14 | -71.4 | 0.05 |
| P13 | -66.8 | 0.05 |
| P02 | -62.5 | 0.05 |
| P06 | -61.3 | 0.05 |
| P01 | -52.1 | 0.05 |
| P10 | -30.5 | 0.05 |
| P09 | -30.2 | 0.05 |
| P20 | -23.8 | 0.05 |
| P08 | -11.4 | 0.05 |
| P15 | 13 | 1.52 |
| P16 | 16.3 | 1.65 |
| P11 | 33.4 | 2.34 |
| P05 | 37.8 | 2.52 |
| P03 | 41.2 | 2.66 |
| P19 | 44.3 | 2.78 |
| P04 | 46.3 | 2.86 |
| P12 | 76.9 | 4.09 |
| P18 | 99.3 | 4.99 |
| P17 | 99.5 | 5 |
