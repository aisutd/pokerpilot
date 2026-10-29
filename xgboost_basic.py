"""
Basic XGBoost example (learning version).

What this does:
1. Loads a small built-in dataset (breast cancer tumors: 569 rows, 30 number columns).
2. Splits it into training data (to learn from) and test data (to check it on).
3. Trains an XGBoost model to predict: malignant (0) or benign (1).
4. Prints how accurate it is and which columns mattered most.

Run it with:  python3 xgboost_basic.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ---- 1. Load data ----
# X = the inputs (features), y = the answer we want to predict (label)
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")

# ---- 2. Split into train and test ----
# The model only sees the training set. The test set is "new" data we use
# to check if it actually learned something, instead of just memorizing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- 3. Create and train the model ----
# XGBoost builds many small decision trees, one after another.
# Each new tree tries to fix the mistakes the previous trees made.
model = XGBClassifier(
    n_estimators=100,   # how many trees to build
    max_depth=3,        # how deep each tree can go (smaller = simpler)
    learning_rate=0.1,  # how much each new tree is allowed to change things
    eval_metric="logloss",
)
model.fit(X_train, y_train)

# ---- 4. Test it ----
predictions = model.predict(X_test)
print(f"\nAccuracy on test data: {accuracy_score(y_test, predictions):.2%}")
print("\nDetailed report:")
print(classification_report(y_test, predictions, target_names=data.target_names))

# ---- 5. Which features mattered most? ----
importance = sorted(
    zip(X.columns, model.feature_importances_), key=lambda p: p[1], reverse=True
)
print("Top 5 most important features:")
for name, score in importance[:5]:
    print(f"  {name:<25} {score:.3f}")

# ---- 6. Predict a single example ----
one_row = X_test.iloc[[0]]
prob = model.predict_proba(one_row)[0]
print(f"\nExample prediction: {prob[1]:.1%} chance benign, "
      f"actual answer: {data.target_names[y_test.iloc[0]]}")
