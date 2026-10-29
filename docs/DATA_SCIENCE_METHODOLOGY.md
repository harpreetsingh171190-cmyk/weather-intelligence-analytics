# Data Science Methodology — Harpreet Singh

## Added capabilities
- Statistical feature profiling: percentiles, skewness, missingness.
- IID bootstrap confidence intervals for descriptive means (not prediction intervals; autocorrelation can invalidate nominal coverage).
- K-means unsupervised clustering with standardization, median imputation and silhouette score.
- Population Stability Index (PSI) for chronological feature distribution comparisons. PSI is descriptive, not a statistical hypothesis test.
- Expanding-window walk-forward validation, persistence and mean baselines, Random Forest, fold-level MAE/RMSE and CSV exports.

## Reproducibility and risk
Fixed random seeds. The demo data are synthetic. No operational weather warnings or official forecasts. Temporal cross-validation does not automatically solve distribution shift, station mixing or data leakage. Clustering labels are exploratory, not validated weather categories. Do not treat bootstrap CI as predictive uncertainty. No API keys or user data are transmitted by these modules.
