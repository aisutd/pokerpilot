# PokerPilot XGBoost model

Predicts what a player does (**fold / check / call / bet / raise**) and, for bets and raises, **how big** (fraction of the pot).

## Run it

```
cd ~/pokerpilot-team/pokerpilot_model
pip install xgboost scikit-learn pandas joblib
python3 predict.py     # uses the already-trained model in artifacts/
python3 train.py       # retrains from data/ (about 6 minutes)
```

On a Mac, if you get an error mentioning `libomp`, run `brew install libomp`.

## Files

| File | What it does |
|---|---|
| `features.py` | Builds the model inputs. Shared by training and the app so they always match. |
| `train.py` | Tunes, trains, tests and saves both models. |
| `predict.py` | `PokerPilotModel().recommend(state)` for the FastAPI backend. |
| `artifacts/` | Saved models, test scores (`metrics.json`, `test_report.txt`), feature importance, tuning log. |
| `data/` | The team CSVs. |

## Version 2: learns from winning players

Every training decision is weighted by how much that player wins, using luck-adjusted EV (bb per 100 hands) from the train and val periods only:

- Losing players (10 of 20) count at **0.05x**, so their habits barely matter.
- Winning players count from **1.5x** (barely winning) up to **5x** (the best).
- The acting player's own style stats (`player_vpip`, `player_pfr`, ...) are no longer inputs. Otherwise the model would copy whatever style the current user already plays.

The weights are saved in `artifacts/player_weights.csv`.

## Results on the test set (winning players' decisions, later dates than training)

| | Model | Baseline |
|---|---|---|
| Accuracy | 80.3% | 63.1% always fold, 72.0% fold-or-check rule |
| Macro F1 (treats every action equally) | 0.717 | 0.326 |
| Bet sizing error | 0.162 pot | 0.440 pot always-median |

This is lower than version 1 (87.3%). That's expected: winners play different styles from each other (tight-aggressive, loose-passive, etc.), and the model no longer gets told whose style it's looking at, so it learns their blended strategy instead of copying one person.

## How it's built

- Inputs are the columns marked `use_as_model_input = Yes` in the feature dictionary (minus the acting player's own stats), plus 7 engineered ones (equity minus pot odds, cost of calling as a share of stack, opponent looseness vs the table, etc.).
- `outcome_*` and `target_*` columns are never inputs, since they reveal the answer.
- Uses the chronological train/val/test split in the data. 30 hyperparameter settings were tried on validation (judged on winners' decisions) with early stopping, then the best was retrained on train + val and scored once on test.
- Illegal actions are removed before picking: when facing a bet the model can only say fold/call/raise; otherwise check/bet/raise.
- `all-in` (69 rows) is treated as raise when facing a bet, and bet otherwise.
- `poker_hands_by_player.csv` is used to score each player's skill for the weights. `poker_games.csv` and `poker_player_summary.csv` aren't model inputs. They're mostly hand results (which would leak the answer), and the per-player stats the model needs are already in the decision files, computed from earlier games only.

## Using it in FastAPI

```python
from predict import PokerPilotModel
model = PokerPilotModel()   # load once at startup

@app.post("/api/pilot")
def pilot(state: dict):
    return model.recommend(state)
```

`state` uses the same column names as `poker_decisions_*.csv`. Missing fields are allowed, but advice gets better the more you fill in.
