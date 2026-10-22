# PokerPilot XGBoost starter model

This is a **behavior imitation** baseline: predict the player's **observed** `target_action` (fold, check, call, bet, raise, all-in) from decision context, cards, and historical opponent statistics. It **does not** learn the optimal strategy or maximize poker earnings. The dataset must be verified for provenance, poker action legality, and representation before claiming real-world performance.

## Install in repository

Copy this `pokerpilot_model` directory into the root of the `pokerpilot` GitHub repository, next to `Dataset/` and `PokerPilot-Website/`.

```text
pokerpilot/
  Dataset/
  PokerPilot-Website/
  pokerpilot_model/
    train_model.py
    api.py
    requirements.txt
```

From the **repository root**, run:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r pokerpilot_model/requirements.txt
python pokerpilot_model/train_model.py
uvicorn pokerpilot_model.api:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs to test `/predict`, `/features`, and `/health`. On your first run, train before starting the API. Run the React/Flask poker website separately as described in `PokerPilot-Website/README.md`.

## How data is used

- Train: `Dataset/poker_decisions_train.csv`
- Validation: `Dataset/poker_decisions_val.csv`
- Test: `Dataset/poker_decisions_test.csv`
- Features: read dynamically from `Dataset/poker_feature_dictionary.csv` where `use_as_model_input` is `Yes`
- Label: `target_action`
- No target columns or future outcomes are used as model features.
- Train/validation/test are not reshuffled.

Trained outputs go to `pokerpilot_model/artifacts/`:

- `poker_action_model.joblib` — preprocessing pipeline, XGBoost classifier, label encoder
- `metrics.json` — accuracy, class-level precision/recall/F1, confusion matrix, log loss

**Warning:** Model artifacts using joblib/pickle can execute code when loaded; only load your own trusted artifacts. Do not commit model binaries if large or sensitive. The training script does not guarantee decision legality; before action recommendations, filter model outputs against the set of legal actions supplied by the poker engine. For serious evaluation, test against a held-out opponent/session and simulate profit and decision quality, rather than relying on action classification accuracy alone.

## Calling the model

1. Run `GET /features` to see required column names.
2. Build a feature record for the **current** decision using only information available at that instant.
3. `POST /predict` with `{"features": {"stake_level": "...", "big_blind_usd": 2, "...": "..."}}`. The request must include **all** required feature keys; missing keys return HTTP 422.

For now, use this API as a separate model service. Integration into the main PokerPilot game requires translating the poker engine's state to the feature schema.
