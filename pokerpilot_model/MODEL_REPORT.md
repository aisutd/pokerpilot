# PokerPilot Decision Model

## Overview

PokerPilot's model package trains an XGBoost classifier to predict a poker
action from the decision context. It also provides offline decision replay,
a simple rule baseline, a FastAPI inference endpoint, and an adapter for the
existing PokerKit game engine.

## Dataset and leakage controls

The package reads the chronological CSV splits in `Dataset/` and uses
`poker_feature_dictionary.csv` as the feature allowlist. It excludes IDs,
action labels and sizes, and post-hand outcomes from model inputs. It checks
that game IDs do not cross the train, validation, and test splits.

The supplied data contains:

| Split | Decisions | Games | Date range |
| --- | ---: | ---: | --- |
| Train | 9,055 | 1,223 | 2026-07-01 to 2026-07-13 |
| Validation | 2,023 | 262 | 2026-07-13 to 2026-07-16 |
| Test | 1,862 | 262 | 2026-07-16 to 2026-07-19 |

Training fits preprocessing on training data, selects the number of boosting
rounds using validation macro F1, then refits on train plus validation before
the final test evaluation.

## Baseline results

The XGBoost model could not be trained in the current macOS environment:
although the Python requirements installed, the XGBoost library requires the
OpenMP runtime `libomp`, which is not installed. Therefore, model accuracy and
model-backed PokerKit simulation results are not available yet.

The test split was evaluated with two baselines:

| Baseline | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Equity and pot-odds rule | 61.5% | 0.416 |
| Always predict `fold` | 63.4% | 0.129 |

The rule baseline uses the recorded equity and pot odds. These are comparisons
against recorded actions, not estimates of counterfactual winnings or poker
profitability.

## Setup and commands

From the repository root, create an environment and install dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r pokerpilot_model/requirements.txt
```

On macOS, install the OpenMP runtime needed by XGBoost:

```sh
brew install libomp
```

Train and evaluate the classifier:

```sh
python -m pokerpilot_model.train \
  --data-dir Dataset \
  --output-dir pokerpilot_model/artifacts
```

Replay test decisions and compare with the rule baseline:

```sh
python -m pokerpilot_model.simulate \
  --data-dir Dataset \
  --model pokerpilot_model/artifacts/model.joblib \
  --split test
```

Run the API with Uvicorn:

```sh
uvicorn pokerpilot_model.api:app --host 127.0.0.1 --port 8000
```

Use `GET /health` to check readiness and `POST /predict` with a JSON object
containing a `features` object to request an action prediction.

Run package tests:

```sh
python -m pytest pokerpilot_model/tests -q
```

The last test run reported **5 passed, 1 skipped**. The skipped test requires
the XGBoost runtime, which could not load without `libomp`.

## PokerKit engine integration

The adapter implements the existing engine bot interface and reconstructs
features available from `BotView`, including cards, betting context, and
equity. The engine does not expose every dataset feature, including some
historical player tendencies and effective stack information; these remain
missing and are imputed by the trained pipeline. Engine simulation results
should therefore be treated as experimental until tested with a fitted model.

After training, run a seeded simulation with:

```sh
PYTHONPATH=.:PokerPilot-Website python -m pokerpilot_model.simulate_engine \
  --hands 100 \
  --seed 7 \
  --model pokerpilot_model/artifacts/model.joblib
```
