"""
PokerPilot: train the XGBoost decision model on the real team data.

VERSION 2: learns to play like the WINNING players. Every decision is
weighted by how much that player wins (luck-adjusted), so losing players'
habits barely count. See section 1b.

Two models get trained:
  1. ACTION model  - predicts fold / check / call / bet / raise
  2. SIZING model  - when the answer is bet or raise, predicts how big
                     (as a fraction of the pot)

What makes it "robust":
  * Uses the chronological train / val / test split already in the data,
    so we test on hands that happened AFTER the training hands (no peeking).
  * Leaky columns (outcome_*, target_*) are never used as inputs.
  * Hyperparameters are tuned on the validation set with early stopping.
  * Class weights stop the model from just saying "fold" all the time.
  * Illegal actions are masked out (you can't check when facing a bet).
  * Final score is reported on the untouched test set, against baselines.

Run:  python3 train.py
Outputs go in ./artifacts/
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss,
                             mean_absolute_error)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from features import (ACTIONS, ACTION_TO_ID, apply_mask, build_features,
                      map_target)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "artifacts"
OUT.mkdir(exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
train = pd.read_csv(DATA / "poker_decisions_train.csv")
val = pd.read_csv(DATA / "poker_decisions_val.csv")
test = pd.read_csv(DATA / "poker_decisions_test.csv")
print(f"Rows  train={len(train)}  val={len(val)}  test={len(test)}")

# ------------------------------------------------------------------
# 1b. Skill weights: learn mostly from WINNING players
# ------------------------------------------------------------------
# Each player's luck-adjusted win rate (EV bb per 100 hands), measured only
# on train + val hands so the test period stays untouched.
# ev_net_bb removes all-in luck, so it's a fairer skill measure than net_bb.
hands = pd.read_csv(DATA / "poker_hands_by_player.csv")
hands = hands[hands["split"].isin(["train", "val"])]
skill = (hands.groupby("player_id")["ev_net_bb"].sum()
         / hands.groupby("player_id").size() * 100)
best_rate = skill.max()

LOSER_WEIGHT = 0.05   # losing players barely count (1/20th to 1/100th of a winner)


def player_weight(rate):
    if rate <= 0:
        return LOSER_WEIGHT
    return 1 + 4 * rate / best_rate      # winners: 1x (barely winning) to 5x (best)


SKILL_WEIGHT = skill.apply(player_weight)
WINNERS = set(skill[skill > 0].index)
skill_table = pd.DataFrame({"ev_bb_per_100": skill.round(1),
                            "weight": SKILL_WEIGHT.round(2)}).sort_values("ev_bb_per_100")
skill_table.to_csv(OUT / "player_weights.csv")
print("\nPlayer weights (luck-adjusted bb/100 on train+val hands):")
print(skill_table.to_string())

w_skill_tr = train["player_id"].map(SKILL_WEIGHT).fillna(LOSER_WEIGHT).values
w_skill_va = val["player_id"].map(SKILL_WEIGHT).fillna(LOSER_WEIGHT).values

X_tr, y_tr = build_features(train), map_target(train)
X_va, y_va = build_features(val), map_target(val)
X_te, y_te = build_features(test), map_target(test)
print(f"Features: {X_tr.shape[1]}")
print("Train class counts:",
      {a: int((y_tr == i).sum()) for i, a in enumerate(ACTIONS)})


def weights(y, power):
    """power=0 -> no weighting, 1 -> fully balanced, 0.5 -> in between."""
    if power == 0:
        return np.ones(len(y))
    return compute_sample_weight("balanced", y) ** power


def scores(model, X, y, facing):
    p = apply_mask(model.predict_proba(X), facing)
    pred = p.argmax(1)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, np.clip(p, 1e-9, 1), labels=range(len(ACTIONS))),
    }, p, pred


# ------------------------------------------------------------------
# 2. Tune hyperparameters on the validation set
# ------------------------------------------------------------------
search_space = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.85, 1.0],
    "reg_lambda": [0.5, 1, 3, 10],
    "reg_alpha": [0, 0.1, 1],
    "gamma": [0, 0.1, 0.5],
    "weight_power": [0, 0.5, 1.0],
}
N_TRIALS = 30


def make_clf(params, n_estimators=2000, early_stop=True):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(ACTIONS),
        n_estimators=n_estimators,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        eval_metric="mlogloss",
        early_stopping_rounds=100 if early_stop else None,
        random_state=SEED,
        n_jobs=-1,
        **{k: v for k, v in params.items() if k != "weight_power"},
    )


va_win = val["player_id"].isin(WINNERS).values
print(f"\nTuning: {N_TRIALS} random settings, picking the best on validation...")
results = []
t0 = time.time()
for trial in range(N_TRIALS):
    params = {k: rng.choice(v).item() for k, v in search_space.items()}
    clf = make_clf(params)
    clf.fit(X_tr, y_tr,
            sample_weight=weights(y_tr, params["weight_power"]) * w_skill_tr,
            eval_set=[(X_va[va_win], y_va[va_win])], verbose=False)
    # Judge each setting on how well it matches WINNING players' decisions.
    s, _, _ = scores(clf, X_va[va_win], y_va[va_win], val.loc[va_win, "facing_bet"])
    # Rank by macro F1 (treats rare actions like bet/check fairly),
    # use log loss to break ties.
    results.append({**params, **s, "best_iteration": clf.best_iteration})
    print(f"  trial {trial + 1:2d}: acc={s['accuracy']:.3f}  "
          f"macroF1={s['macro_f1']:.3f}  logloss={s['log_loss']:.3f}")

res = pd.DataFrame(results).sort_values(["macro_f1", "log_loss"],
                                        ascending=[False, True])
res.to_csv(OUT / "tuning_results.csv", index=False)
best = res.iloc[0].to_dict()
best_params = {k: best[k] for k in search_space}
for k in ["max_depth", "min_child_weight"]:
    best_params[k] = int(best_params[k])
n_trees = int(best["best_iteration"]) + 1
print(f"Tuning took {time.time() - t0:.0f}s. Best: {best_params}, trees={n_trees}")

# ------------------------------------------------------------------
# 3. Retrain on train + val with the best settings, score on TEST
# ------------------------------------------------------------------
X_full = pd.concat([X_tr, X_va])
y_full = pd.concat([y_tr, y_va])
# A bit more data -> a few more trees is usually right.
final_trees = int(n_trees * 1.15)
action_model = make_clf(best_params, n_estimators=final_trees, early_stop=False)
action_model.fit(X_full, y_full,
                 sample_weight=weights(y_full, best_params["weight_power"])
                 * np.concatenate([w_skill_tr, w_skill_va]))

# Score on TEST decisions made by winning players only: that's what we
# want the model to reproduce now. (Scores on everyone are printed too.)
all_scores, _, _ = scores(action_model, X_te, y_te, test["facing_bet"])
te_win = test["player_id"].isin(WINNERS).values
test = test[te_win].reset_index(drop=True)
X_te, y_te = X_te[te_win].reset_index(drop=True), y_te[te_win].reset_index(drop=True)
test_scores, p_te, pred_te = scores(action_model, X_te, y_te, test["facing_bet"])

# Baselines so we know the number means something
majority = np.full(len(y_te), ACTION_TO_ID["fold"])
rule = np.where(test["facing_bet"] == 1, ACTION_TO_ID["fold"], ACTION_TO_ID["check"])
baselines = {
    "always_fold_accuracy": accuracy_score(y_te, majority),
    "fold_or_check_rule_accuracy": accuracy_score(y_te, rule),
    "fold_or_check_rule_macro_f1": f1_score(y_te, rule, average="macro"),
}

print("\n====== TEST SET RESULTS (winning players' decisions, n=%d) ======" % len(y_te))
print(f"Accuracy   {test_scores['accuracy']:.3f}   "
      f"(always-fold baseline {baselines['always_fold_accuracy']:.3f}, "
      f"fold/check rule {baselines['fold_or_check_rule_accuracy']:.3f})")
print(f"Macro F1   {test_scores['macro_f1']:.3f}   "
      f"(rule baseline {baselines['fold_or_check_rule_macro_f1']:.3f})")
print(f"Log loss   {test_scores['log_loss']:.3f}")
print(f"(On ALL players' test decisions: acc={all_scores['accuracy']:.3f}, "
      f"macroF1={all_scores['macro_f1']:.3f}; lower is expected, since the "
      f"model no longer copies losing players)")
report = classification_report(y_te, pred_te, target_names=ACTIONS, digits=3)
print(report)
cm = pd.DataFrame(confusion_matrix(y_te, pred_te), index=ACTIONS, columns=ACTIONS)
print("Confusion matrix (rows = what player did, cols = model):")
print(cm)

by_street = {}
for street in ["Preflop", "Flop", "Turn", "River"]:
    m = (test["street"] == street).values
    by_street[street] = {
        "rows": int(m.sum()),
        "accuracy": accuracy_score(y_te[m], pred_te[m]),
        "macro_f1": f1_score(y_te[m], pred_te[m], average="macro"),
    }
print("\nBy street:")
for s, v in by_street.items():
    print(f"  {s:<8} n={v['rows']:<5} acc={v['accuracy']:.3f}  "
          f"macroF1={v['macro_f1']:.3f}")

imp = (pd.Series(action_model.get_booster().get_score(importance_type="gain"))
       .sort_values(ascending=False))
imp = imp / imp.sum()
imp.to_csv(OUT / "feature_importance.csv", header=["gain_share"])
print("\nTop 12 features (share of total gain):")
for name, v in imp.head(12).items():
    print(f"  {name:<28} {v:.3f}")

# ------------------------------------------------------------------
# 4. SIZING model: how big to bet/raise (fraction of pot)
# ------------------------------------------------------------------
def sizing_rows(df):
    m = df["target_action"].isin(["bet", "raise"]) & df["target_bet_size_pot_frac"].notna()
    w = df.loc[m, "player_id"].map(SKILL_WEIGHT).fillna(LOSER_WEIGHT).values
    return (build_features(df[m]), df.loc[m, "target_bet_size_pot_frac"].clip(upper=3), w)


Xs_tr, ys_tr, ws_tr = sizing_rows(train)
Xs_va, ys_va, ws_va = sizing_rows(val)
Xs_te, ys_te, _ = sizing_rows(test)          # test = winners only by now
size_params = dict(n_estimators=2000, max_depth=4, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                   tree_method="hist", enable_categorical=True,
                   max_cat_to_onehot=1, random_state=SEED, n_jobs=-1)
probe = XGBRegressor(**size_params, early_stopping_rounds=100,
                     objective="reg:absoluteerror")
probe.fit(Xs_tr, ys_tr, sample_weight=ws_tr,
          eval_set=[(Xs_va, ys_va)], sample_weight_eval_set=[ws_va], verbose=False)
size_params["n_estimators"] = int((probe.best_iteration + 1) * 1.15)
sizing_model = XGBRegressor(**size_params, objective="reg:absoluteerror")
sizing_model.fit(pd.concat([Xs_tr, Xs_va]), pd.concat([ys_tr, ys_va]),
                 sample_weight=np.concatenate([ws_tr, ws_va]))
size_mae = mean_absolute_error(ys_te, sizing_model.predict(Xs_te))
size_base = mean_absolute_error(ys_te, np.full(len(ys_te), np.median(ys_tr)))
print(f"\nSizing model MAE on test: {size_mae:.3f} pot "
      f"(baseline 'always median size': {size_base:.3f})")

# ------------------------------------------------------------------
# 5. Save everything
# ------------------------------------------------------------------
action_model.save_model(OUT / "action_model.json")
sizing_model.save_model(OUT / "sizing_model.json")
metrics = {
    "test_winning_players": test_scores,
    "test_all_players": all_scores,
    "winning_players": sorted(WINNERS),
    "baselines": baselines,
    "by_street": by_street,
    "sizing_mae_pot_frac": size_mae,
    "sizing_baseline_mae": size_base,
    "best_params": best_params,
    "n_trees": final_trees,
    "classes": ACTIONS,
}
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=float)
with open(OUT / "test_report.txt", "w") as f:
    f.write(report + "\n" + cm.to_string() + "\n")
print(f"\nSaved models and reports to {OUT}/")
