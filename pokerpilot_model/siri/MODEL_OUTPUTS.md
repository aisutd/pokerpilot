# PokerPilot Model Outputs

This file records comparable evaluation outputs for PokerPilot models. Add
future models as rows in the summary table and include their per-action
metrics below, using the same held-out test split and label order.

Each training or replay run also creates a new timestamped Markdown report in
`pokerpilot_model/siri/results/`. Earlier run files are retained for comparison.

## Evaluation setup

| Item | Value |
| --- | --- |
| Dataset | `Dataset/poker_decisions_*.csv` |
| Test split | 1,862 decisions from 262 games |
| Test dates | 2026-07-16 to 2026-07-19 |
| Actions | `all-in`, `bet`, `call`, `check`, `fold`, `raise` |
| Estimator | XGBoost 3.4.1, seed 23 |
| Selected settings | 100 rounds, depth 5, class-weight power 0.5, call multiplier 1.25 |
| Leakage check | No game IDs overlap across train, validation, and test |
| Model selection | Validation F1 for check/call/bet/raise; overall macro F1 breaks ties |

## Summary comparison

| Model / strategy | Test accuracy | Test macro F1 | Notes |
| --- | ---: | ---: | --- |
| PokerPilot XGBoost, target-action tuned | **87.54%** | **0.7482** | Seed 23; 100 rounds, depth 5; call multiplier 1.25 |
| PokerPilot XGBoost, previous class weighted | 87.43% | 0.7415 | 150 rounds, depth 5 |
| PokerPilot XGBoost, earlier unweighted | 87.54% | 0.7125 | 200 rounds |
| Equity and pot-odds rule | 61.55% | 0.4162 | Uses recorded equity and pot odds |
| Always predict `fold` | 63.43% | 0.1294 | Training majority-class baseline |

The target-action-tuned model improves macro F1 by **0.0067** over the previous
class-weighted model. Test F1 is in the requested 0.70–0.90 range for check
(0.8159), bet (0.7614), and raise (0.7506; grouped bet/raise: 0.7539). Call
improved from 0.6802 to **0.6875**, but remains just below 0.70; its validation
F1 is 0.7362. The test split was not used to select the model.

## Split results

| Split | Decisions | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Validation | 2,023 | 87.40% | 0.7444 |
| Test | 1,862 | 87.54% | 0.7482 |

## Test metrics by action

| Action | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `all-in` | 13 | 0.6000 | 0.4615 | 0.5217 |
| `bet` | 90 | 0.7009 | 0.8333 | 0.7614 |
| `call` | 195 | 0.6471 | 0.7333 | 0.6875 |
| `check` | 144 | 0.8496 | 0.7847 | 0.8159 |
| `fold` | 1,181 | 0.9502 | 0.9534 | 0.9518 |
| `raise` | 239 | 0.8107 | 0.6987 | 0.7506 |
| **Macro average** | **1,862** | **0.7598** | **0.7442** | **0.7482** |

The `all-in` estimate is uncertain because the test split contains only 13
such decisions. Read accuracy alongside macro F1 because `fold` is the
dominant test label.

## Example prediction output

For the first row of the held-out test CSV, the actual action is `fold` and
the API predicts `fold`:

| Action | Probability |
| --- | ---: |
| `all-in` | 0.00090 |
| `bet` | 0.00261 |
| `call` | 0.02565 |
| `check` | 0.00348 |
| `fold` | **0.94288** |
| `raise` | 0.02448 |

## Predicted action counts

| Predicted action | Count |
| --- | ---: |
| `all-in` | 10 |
| `bet` | 107 |
| `call` | 221 |
| `check` | 133 |
| `fold` | 1,185 |
| `raise` | 206 |
| **Total** | **1,862** |

## Confusion matrix

Rows are recorded actions; columns are predicted actions.

| Actual \\ Predicted | `all-in` | `bet` | `call` | `check` | `fold` | `raise` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all-in` | 6 | 1 | 6 | 0 | 0 | 0 |
| `bet` | 1 | 75 | 0 | 14 | 0 | 0 |
| `call` | 2 | 0 | 143 | 0 | 23 | 27 |
| `check` | 0 | 31 | 0 | 113 | 0 | 0 |
| `fold` | 1 | 0 | 42 | 0 | 1,126 | 12 |
| `raise` | 0 | 0 | 30 | 6 | 36 | 167 |

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
python -m pokerpilot_model.siri.train \
  --data-dir Dataset \
  --output-dir pokerpilot_model/siri/artifacts \
  --report-dir pokerpilot_model/siri/results

python -m pokerpilot_model.siri.simulate \
  --data-dir Dataset \
  --model pokerpilot_model/siri/artifacts/model.joblib \
  --split test \
  --report-dir pokerpilot_model/siri/results

PYTHONPATH=.:PokerPilot-Website python -m pokerpilot_model.siri.simulate_engine \
  --hands 5 \
  --seed 7 \
  --model pokerpilot_model/siri/artifacts/model.joblib
```

Training writes detailed validation and test classification reports to
`pokerpilot_model/siri/artifacts/metrics.json` and the fitted model to
`pokerpilot_model/siri/artifacts/model.joblib`. The comparison above is a recorded
action classification evaluation. It does not measure counterfactual poker
profit, winnings, or full-hand performance against other players.
