# PokerPilot Model Outputs

This file records comparable evaluation outputs for PokerPilot models. Add
future models as rows in the summary table and include their per-action
metrics below, using the same held-out test split and label order.

Each training or replay run also creates a new timestamped Markdown report in
`pokerpilot_model/results/`. Earlier run files are retained for comparison.

## Evaluation setup

| Item | Value |
| --- | --- |
| Dataset | `Dataset/poker_decisions_*.csv` |
| Test split | 1,862 decisions from 262 games |
| Test dates | 2026-07-16 to 2026-07-19 |
| Actions | `all-in`, `bet`, `call`, `check`, `fold`, `raise` |
| Estimator | XGBoost 3.4.1, seed 23 |
| Selected settings | 150 rounds, depth 5, class-weight power 0.5 |
| Leakage check | No game IDs overlap across train, validation, and test |
| Model selection | Validation macro F1; test used only for final evaluation |

## Summary comparison

| Model / strategy | Test accuracy | Test macro F1 | Notes |
| --- | ---: | ---: | --- |
| PokerPilot XGBoost, class weighted | **87.43%** | **0.7415** | Seed 23; 150 rounds, depth 5 |
| PokerPilot XGBoost, previous | 87.54% | 0.7125 | Unweighted model; 200 rounds |
| Equity and pot-odds rule | 61.55% | 0.4162 | Uses recorded equity and pot odds |
| Always predict `fold` | 63.43% | 0.1294 | Training majority-class baseline |

The class-weighted model improves macro F1 by **0.0290** (2.90 percentage
points) over the previous XGBoost run, with a 0.11-point accuracy decrease.
Against the rule baseline, it improves test accuracy by **25.89 percentage
points** and macro F1 by **0.3253**.

## Split results

| Split | Decisions | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Validation | 2,023 | 87.00% | 0.7486 |
| Test | 1,862 | 87.43% | 0.7415 |

## Test metrics by action

| Action | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `all-in` | 13 | 0.6000 | 0.4615 | 0.5217 |
| `bet` | 90 | 0.7030 | 0.7889 | 0.7435 |
| `call` | 195 | 0.6734 | 0.6872 | 0.6802 |
| `check` | 144 | 0.8201 | 0.7917 | 0.8057 |
| `fold` | 1,181 | 0.9481 | 0.9594 | 0.9537 |
| `raise` | 239 | 0.7798 | 0.7113 | 0.7440 |
| **Macro average** | **1,862** | **0.7541** | **0.7333** | **0.7415** |

The `all-in` estimate is uncertain because the test split contains only 13
such decisions. Read accuracy alongside macro F1 because `fold` is the
dominant test label.

## Example prediction output

For the first row of the held-out test CSV, the actual action is `fold` and
the API predicts `fold`:

| Action | Probability |
| --- | ---: |
| `all-in` | 0.00029 |
| `bet` | 0.00079 |
| `call` | 0.02057 |
| `check` | 0.00103 |
| `fold` | **0.95208** |
| `raise` | 0.02524 |

## Predicted action counts

| Predicted action | Count |
| --- | ---: |
| `all-in` | 10 |
| `bet` | 101 |
| `call` | 199 |
| `check` | 139 |
| `fold` | 1,195 |
| `raise` | 218 |
| **Total** | **1,862** |

## Confusion matrix

Rows are recorded actions; columns are predicted actions.

| Actual \\ Predicted | `all-in` | `bet` | `call` | `check` | `fold` | `raise` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all-in` | 6 | 0 | 6 | 1 | 0 | 0 |
| `bet` | 1 | 71 | 0 | 18 | 0 | 0 |
| `call` | 2 | 0 | 134 | 0 | 27 | 32 |
| `check` | 0 | 30 | 0 | 114 | 0 | 0 |
| `fold` | 1 | 0 | 31 | 0 | 1,133 | 16 |
| `raise` | 0 | 0 | 28 | 6 | 35 | 170 |

## Seeded PokerKit simulation smoke run

The model was also run as all six seats in the existing PokerKit engine for
five complete hands. Two runs with seed `7` produced identical stacks:

| Seat | Final stack |
| --- | ---: |
| 0 | $205.00 |
| 1 | $238.53 |
| 2 | $203.00 |
| 3 | $197.00 |
| 4 | $197.00 |
| 5 | $159.47 |

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
