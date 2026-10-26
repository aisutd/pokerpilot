# PokerPilot Monte Carlo Evaluation

## Method

- Input: `poker_decisions_all.csv`; 2504 flop, turn, and river decisions inspected.
- Eligible decisions simulated: 2504.
- 500 random deals per decision; opponent cards and any remaining board cards are sampled.
- Predicted win = estimated outright win probability >= 50%.
- Five-fold GroupKFold keeps decisions from the same game together.
- A fresh 500-deal Monte Carlo simulation is run on the held-out decisions inside each fold.

## Results

| Fold | Rows | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 501 | 67.47% | 63.06% | 72.53% | 67.47% |
| 2 | 501 | 67.07% | 63.07% | 66.67% | 64.82% |
| 3 | 502 | 70.32% | 66.82% | 66.52% | 66.67% |
| 4 | 500 | 71.80% | 64.29% | 78.80% | 70.81% |
| 5 | 500 | 68.80% | 63.42% | 72.44% | 67.63% |

## Mean metrics

- **Accuracy: 69.09%**
- **Precision: 64.13%**
- **Recall: 71.39%**
- **F1: 67.48%**

## Decisions simulated by street

- Flop: 1156
- Turn: 824
- River: 524

## Important limitations

- These classification scores compare simulations with realized historical outcomes; a probabilistic prediction can be reasonable even when a single hand loses.
- Simulations are rerun inside each fold, but Monte Carlo is not a trained model; this is grouped fold-wise evaluation, not training-based cross-validation.
- Opponent hole cards are sampled uniformly, not conditioned on observed betting actions or player styles.
- River simulations have a complete five-card board; only unknown opponent hole cards are sampled.
- The 50% threshold is an evaluation choice, not a recommended poker decision rule.
