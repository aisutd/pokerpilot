# PokerPilot decision model

Train an XGBoost multiclass model from the chronological decision CSVs, score
the supplied validation and test sets, replay decisions offline, serve a JSON
API, or use the model as a bot in the PokerKit website engine.

## Setup

From the repository root, use Python 3.10+:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r pokerpilot_model/requirements.txt
```

On macOS, XGBoost also requires the OpenMP runtime: `brew install libomp`.

The engine integration also needs the existing website dependencies:
`pip install -r PokerPilot-Website/requirements.txt`.

## Train and evaluate

```sh
python -m pokerpilot_model.train --data-dir Dataset --output-dir pokerpilot_model/artifacts
```

The feature allowlist is taken from `poker_feature_dictionary.csv`. IDs,
targets, and post-hand outcomes are excluded. Splits are checked for game ID
overlap. The training set fits the encoder and estimator; validation is used
for selecting the reported checkpoint count; test is scored once after that
selection. Results and the fitted pipeline are saved under `artifacts/`.

## Offline decision replay

```sh
python -m pokerpilot_model.simulate --data-dir Dataset --model pokerpilot_model/artifacts/model.joblib --split test
```

This is a deterministic decision replay against recorded labels, not a
counterfactual poker outcome or profitability estimate. The rule baseline
uses the recorded equity and pot odds and is evaluated on the same decisions.

## API

```sh
uvicorn pokerpilot_model.api:app --host 127.0.0.1 --port 8000
```

`POST /predict` accepts a JSON object of feature fields and returns the
predicted action and per-action probabilities. `/health` reports readiness.
Set `POKERPILOT_MODEL_PATH` to select a model artifact.

## PokerKit engine

```sh
PYTHONPATH=.:PokerPilot-Website python -m pokerpilot_model.simulate_engine --hands 100 --seed 7 --model pokerpilot_model/artifacts/model.joblib
```

The adapter reconstructs public state, cards, hand strength, equity, and
available opponent statistics from the engine's `BotView`. Features with no
equivalent in the engine (such as historical net results) remain missing and
are imputed by the training pipeline. This makes the integration executable,
but its engine play is an out-of-distribution smoke simulation, not a
validated estimate of poker skill. Raise sizes use the training-set median
size for that action and are clamped to legal engine limits.

## Tests

Run `pytest pokerpilot_model/tests`. Tests cover leakage/split checks,
feature building, deterministic simulation, API behavior, and the engine
adapter's legal-action fallback.
