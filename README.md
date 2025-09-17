## SaaS Revenue Multiple Predictor

This module trains a supervised model to predict a company's Revenue Multiple.

By default, it automatically parses and uses numeric columns from the CSV (monetary, percentages, counts, etc.). To prevent leakage, the loader excludes identifiers, links/text, the target, and any column whose name suggests it encodes valuation/multiples/price directly (e.g., `Stock Price`, `Market Cap`, `Price/Sales`, `EV/Revenue`, `Shares Outstanding`, etc.). Columns are parsed as dollars, percentages, or floats, with median imputation for sparse missing values.

Data source: the included CSV `– A free database of all SaaS businesses listed on the U.S. stock exchanges NYSE and NASDAQ.csv`.

### Approach

- Auto-parse numeric features while guarding against leakage (see above).
- Hold-out split (default 80/20) with fixed seed for reproducibility.
- Model: GradientBoostingRegressor; optional Optuna K-fold tuning.
- Optional MC-style ensemble for robustness/uncertainty.
- Metrics: MAE, RMSE, R², median AE, p90/p95 AE, MAPE; saved to JSON.
- Visuals: parity (actual vs predicted), residuals scatter, residuals histogram, top feature importances.

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
# Reproduce latest run
python saas_revenue_multiple_model.py train \
  --tune --trials 10 --cv-folds 5 \
  --mc-ensemble 5 --mc-bootstrap \
  --csv "– A free database of all SaaS businesses listed on the U.S. stock exchanges NYSE and NASDAQ.csv" \
  --model-out saas_rev_multiple_model.pkl \
  --plot saas_rev_multiple_fit.png \
  --plot-style parity --plot-label-tickers --plot-color-by none --plot-size-by none
```

Outputs:

- `saas_rev_multiple_model.pkl` — serialized model
- Plots:
  - `saas_rev_multiple_fit.png` — parity (actual vs predicted)
  - `saas_rev_multiple_fit_residuals_scatter.png`
  - `saas_rev_multiple_fit_residuals_hist.png`
  - `saas_rev_multiple_fit_feature_importance.png`
- JSON trace: `saas_model_report.json` with metrics/config

### Predict

Predict by ticker (features auto-derived from CSV):

```bash
python saas_revenue_multiple_model.py predict \
  --model saas_rev_multiple_model.pkl \
  --ticker ADBE \
  --csv "– A free database of all SaaS businesses listed on the U.S. stock exchanges NYSE and NASDAQ.csv"
```

Note: In auto-feature mode, prediction requires `--ticker` + `--csv` and raw manual values are disabled (features depend on the parsed CSV schema).

### Hyperparameter Tuning (Optuna + K-fold CV)

Enable tuning with K-fold CV:

```bash
python saas_revenue_multiple_model.py train \
  --tune --trials 50 --cv-folds 5 \
  --csv "– A free database of all SaaS businesses listed on the U.S. stock exchanges NYSE and NASDAQ.csv" \
  --model-out saas_rev_multiple_model.pkl \
  --plot saas_rev_multiple_fit.png
```

This searches over `n_estimators`, `learning_rate`, `max_depth`, and `subsample` for `GradientBoostingRegressor` using mean MAE across folds.

### Monte Carlo-style Ensemble for Uncertainty

You can train an ensemble of models via bagging to estimate uncertainty:

```bash
python saas_revenue_multiple_model.py train \
  --tune --trials 25 --cv-folds 5 \
  --mc-ensemble 15 --mc-bootstrap \
  --csv "– A free database of all SaaS businesses listed on the U.S. stock exchanges NYSE and NASDAQ.csv" \
  --model-out saas_rev_multiple_model.pkl
```

At prediction time, request mean +/- std from the ensemble:

```bash
python saas_revenue_multiple_model.py predict \
  --model saas_rev_multiple_model.pkl \
  --ticker ADBE \
  --csv "– A free database of all SaaS businesses listed on the U.S. stock exchanges NYSE and NASDAQ.csv" \
  --uncertainty
```

### Notes

- The script robustly parses '$' and commas in cash/debt columns and '%' in margin.
- Rows with missing inputs/target are dropped before splitting.
- You can adjust model choice or hyperparameters inside `saas_revenue_multiple_model.py`.


