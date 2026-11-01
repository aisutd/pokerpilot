# Siri XGBoost — Five-Fold Cross-Validation

Method: expanding chronological TimeSeriesSplit, grouped by game ID.
Training split only; original validation and test splits excluded.
Fixed settings: 150 estimators, max depth 5, weight power 0.5.

Mean accuracy: 84.5785%

Mean macro F1: 0.6365

See CV_RESULTS.csv for individual fold scores.
