"""
Poker Pilot: basic XGBoost model that predicts FOLD, CALL, or RAISE.

IMPORTANT: The data here is FAKE (randomly generated) so you can learn how
the pieces fit together. When the team has real hand data, replace step 1
with loading that data (for example, pd.read_csv("hands.csv")).

Run it with:  python3 poker_xgboost.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

ACTIONS = ["fold", "call", "raise"]   # 0 = fold, 1 = call, 2 = raise

# ---- 1. Make fake poker data (CHANGED: was a built-in dataset) ----
# Each row is one decision a player made.
rng = np.random.default_rng(42)
n = 3000

df = pd.DataFrame({
    "hand_strength": rng.uniform(0, 1, n),          # 0 = trash, 1 = the nuts
    "position":      rng.integers(0, 6, n),         # 0 = early seat, 5 = button
    "pot_size":      rng.integers(10, 500, n),      # chips already in the pot
    "bet_to_call":   rng.integers(0, 200, n),       # chips needed to stay in
    "stack_size":    rng.integers(100, 2000, n),    # chips the player has left
    "num_players":   rng.integers(2, 7, n),         # players still in the hand
})

# A feature built from other features ("feature engineering").
# Pot odds = what fraction of the final pot you'd be paying to call.
df["pot_odds"] = df["bet_to_call"] / (df["pot_size"] + df["bet_to_call"])

# Fake "correct" answers using a simple rule, plus some randomness,
# so the model has a pattern to learn.
def pick_action(row):
    strength = row.hand_strength + row.position * 0.02 + rng.normal(0, 0.08)
    if strength > 0.75:
        return 2                      # raise
    if strength > row.pot_odds:
        return 1                      # call
    return 0                          # fold

df["action"] = df.apply(pick_action, axis=1)

X = df.drop(columns="action")   # inputs
y = df["action"]                # answer to predict
print(f"Dataset: {len(df)} decisions, {X.shape[1]} features")
print("Actions in data:", dict(zip(ACTIONS, np.bincount(y).tolist())))

# ---- 2. Split into train and test (SAME as before) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 3. Train the model (CHANGED: 3 answers instead of 2) ----
# XGBoost sees 3 classes and switches to "multi-class" mode by itself.
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="mlogloss",   # "m" = multi-class version of logloss
)
model.fit(X_train, y_train)

# ---- 4. Test it (SAME as before) ----
predictions = model.predict(X_test)
print(f"\nAccuracy on test data: {accuracy_score(y_test, predictions):.2%}")
print("\nDetailed report:")
print(classification_report(y_test, predictions, target_names=ACTIONS))

# ---- 5. Which features mattered most? (SAME as before) ----
importance = sorted(
    zip(X.columns, model.feature_importances_), key=lambda p: p[1], reverse=True
)
print("Feature importance:")
for name, score in importance:
    print(f"  {name:<15} {score:.3f}")

# ---- 6. Advise on ONE new hand (CHANGED: this is what the app would do) ----
new_hand = pd.DataFrame([{
    "hand_strength": 0.55,
    "position": 5,          # on the button
    "pot_size": 120,
    "bet_to_call": 40,
    "stack_size": 900,
    "num_players": 3,
}])
new_hand["pot_odds"] = new_hand["bet_to_call"] / (new_hand["pot_size"] + new_hand["bet_to_call"])

probs = model.predict_proba(new_hand)[0]
best = ACTIONS[int(np.argmax(probs))]
print("\nNew hand advice:")
for action, p in zip(ACTIONS, probs):
    print(f"  {action:<6} {p:.1%}")
print(f"  -> Recommended: {best.upper()}")
