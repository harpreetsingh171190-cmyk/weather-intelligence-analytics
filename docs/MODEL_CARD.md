# Model Card — Weather Intelligence Prototype
**Author:** Harpreet Singh

## Intended use
Education, portfolio demonstration, and exploratory one-day-ahead regression of daily weather measurements.

## Training and evaluation
Lagged values (1, 2, 3, 7 days), prior 7-day mean and day-of-year sine/cosine features. Chronological 80/20 holdout. No random shuffling. Compare Random Forest, gradient boosting, and mean baseline. The holdout is not a substitute for repeated walk-forward backtesting.

## Limitations
Sample data is synthetic; no real-world accuracy claims. Data must be continuous daily measurements. No operational forecasts, weather warnings, uncertainty intervals, regional generalization or calibration. Holdout evaluation may be optimistic if model selection is repeated on the same test set. LSTM is optional, CPU-intensive and not guaranteed to outperform baseline.

## Computer vision
Blue and bright-gray pixel percentages are simple image color heuristics, NOT cloud segmentation, meteorological cloud-cover estimates, or validated computer vision classification.

## Governance
Record data provenance, location, time zone, licenses, quality, performance by season and weather condition; monitor drift and retrain only after validation. Do not use for safety-critical decisions.
