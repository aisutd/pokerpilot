# Poker XGBoost Cross-Validation Results

## Overview

- Dataset: `poker_decisions_all.csv`
- Training decisions: 9,055
- Algorithm: XGBoost
- Cross-validation: 5-Fold GroupKFold
- Grouping: game_id
- Averaging: Macro (except accuracy)

## Results by Fold

| Fold | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| 1 | 0.7924 | 0.7151 | 0.5846 | 0.6233 |
| 2 | 0.8067 | 0.6508 | 0.5741 | 0.5982 |
| 3 | 0.8062 | 0.6442 | 0.5830 | 0.6030 |
| 4 | 0.7962 | 0.6355 | 0.5568 | 0.5812 |
| 5 | 0.8018 | 0.6155 | 0.5426 | 0.5645 |

## Summary

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 0.8007 | 0.0056 |
| Precision (Macro) | 0.6522 | 0.0336 |
| Recall (Macro) | 0.5682 | 0.0162 |
| F1 Score (Macro) | 0.5940 | 0.0200 |

## Metric Explanations

- **Accuracy:** Percentage of all predictions that were correct.
- **Precision:** Of the predictions for a particular action, how many were correct?
- **Recall:** Of all actual occurrences of an action, how many did the model identify?
- **F1 Score:** Harmonic mean of precision and recall.

Macro averaging treats all six actions equally:
fold, check, call, bet, raise, and all-in.

## Notes

- Cross-validation uses only the training partition.
- All decisions from one game stay in the same fold.
- Validation and test partitions are excluded.
- Scores evaluate historical-action classification, not the profitability of poker decisions.
- Standard deviation is calculated across the five folds.
