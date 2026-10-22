# PokerPilot Model Outputs

This file records comparable evaluation outputs for PokerPilot models. Add
future models as rows in the summary table and include their per-action
metrics below, using the same held-out test split and label order.

## Evaluation setup

| Item | Value |
| --- | --- |
| Dataset | `Dataset/poker_decisions_*.csv` |
| Test split | 1,862 decisions from 262 games |
| Test dates | 2026-07-16 to 2026-07-19 |
| Actions | `all-in`, `bet`, `call`, `check`, `fold`, `raise` |
| Estimator | XGBoost 3.4.1, seed 23 |
| Selected boosting rounds | 200 (selected using validation macro F1) |
| Leakage check | No game IDs overlap across train, validation, and test |
| Model selection | Validation macro F1; test used only for final evaluation |

## Summary comparison

| Model / strategy | Test accuracy | Test macro F1 | Notes |
| --- | ---: | ---: | --- |
| PokerPilot XGBoost | **87.54%** | **0.7125** | Seed 23; 200 boosting rounds selected on validation |
| Equity and pot-odds rule | 61.55% | 0.4162 | Uses recorded equity and pot odds |
| Always predict `fold` | 63.43% | 0.1294 | Training majority-class baseline |

Against the rule baseline, XGBoost improves test accuracy by **25.99
percentage points** and macro F1 by **0.2963**. Against always-fold, it improves
accuracy by **24.11 points** and macro F1 by **0.5831**.

## Split results

| Split | Decisions | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Validation | 2,023 | 87.10% | 0.7291 |
| Test | 1,862 | 87.54% | 0.7125 |

## Test metrics by action

| Action | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `all-in` | 13 | 1.0000 | 0.2308 | 0.3750 |
| `bet` | 90 | 0.7159 | 0.7000 | 0.7079 |
| `call` | 195 | 0.7314 | 0.6564 | 0.6919 |
| `check` | 144 | 0.7792 | 0.8333 | 0.8054 |
| `fold` | 1,181 | 0.9255 | 0.9780 | 0.9510 |
| `raise` | 239 | 0.8299 | 0.6736 | 0.7436 |
| **Macro average** | **1,862** | **0.8303** | **0.6787** | **0.7125** |

The low `all-in` recall reflects the small test support (13 decisions). The
strong overall accuracy should be read alongside macro F1 because `fold` is
the dominant test label.

## Example prediction output

For the first row of the held-out test CSV, the actual action is `fold` and
the API predicts `fold`:

| Action | Probability |
| --- | ---: |
| `all-in` | 0.00003 |
| `bet` | 0.00015 |
| `call` | 0.00768 |
| `check` | 0.00025 |
| `fold` | **0.97306** |
| `raise` | 0.01882 |

## Predicted action counts

| Predicted action | Count |
| --- | ---: |
| `all-in` | 3 |
| `bet` | 88 |
| `call` | 175 |
| `check` | 154 |
| `fold` | 1,248 |
| `raise` | 194 |
| **Total** | **1,862** |

## Confusion matrix

Rows are recorded actions; columns are predicted actions.

| Actual \\ Predicted | `all-in` | `bet` | `call` | `check` | `fold` | `raise` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all-in` | 3 | 1 | 7 | 1 | 1 | 0 |
| `bet` | 0 | 63 | 0 | 27 | 0 | 0 |
| `call` | 0 | 0 | 128 | 0 | 41 | 26 |
| `check` | 0 | 24 | 0 | 120 | 0 | 0 |
| `fold` | 0 | 0 | 19 | 0 | 1,155 | 7 |
| `raise` | 0 | 0 | 21 | 6 | 51 | 161 |

## Seeded PokerKit simulation smoke run

The model was also run as all six seats in the existing PokerKit engine for
five complete hands. Two runs with seed `7` produced identical stacks:

| Seat | Final stack |
| --- | ---: |
| 0 | $211.00 |
| 1 | $197.00 |
| 2 | $198.00 |
| 3 | $193.00 |
| 4 | $207.00 |
| 5 | $194.00 |

This small run verifies engine integration and seed reproducibility. Five
hands are not enough to estimate playing strength or profitability.

## Reproduce these outputs

From the repository root, install `pokerpilot_model/requirements.txt`, then run:

```sh
python -m pokerpilot_model.train \
  --data-dir Dataset \
  --output-dir pokerpilot_model/artifacts

python -m pokerpilot_model.simulate \
  --data-dir Dataset \
  --model pokerpilot_model/artifacts/model.joblib \
  --split test

PYTHONPATH=.:PokerPilot-Website python -m pokerpilot_model.simulate_engine \
  --hands 5 \
  --seed 7 \
  --model pokerpilot_model/artifacts/model.joblib
```

Training writes detailed validation and test classification reports to
`pokerpilot_model/artifacts/metrics.json` and the fitted model to
`pokerpilot_model/artifacts/model.joblib`. The comparison above is a recorded
action classification evaluation. It does not measure counterfactual poker
profit, winnings, or full-hand performance against other players.
