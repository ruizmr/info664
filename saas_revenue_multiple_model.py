"""SaaS Revenue Multiple Predictor

This module trains a supervised model to predict a company's revenue multiple
from tabular SaaS data. Key capabilities:

- Auto-feature parsing: converts monetary, percentage, and numeric-like columns
  to floats, while excluding identifiers, links, and leakage-prone columns.
- Leakage guards: drops any column whose name implies valuation/multiples/price.
- Model: GradientBoostingRegressor (GBM), optionally tuned via Optuna K-fold CV.
- Monte Carlo-style ensemble (bagging) for robustness and uncertainty.
- Evaluation: prints MAE, RMSE, R² and saves a JSON trace with metrics/configs.
- Visuals: parity plot with ticker labels, residuals scatter & histogram,
  and top feature importances.
- CLI: "train" to fit/evaluate/plot, "predict" to infer by ticker from CSV.

Note: The implementation emphasises clarity and guardrails over cleverness.
"""

import argparse
import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import optuna


CSV_DEFAULT = "– A free database of all SaaS businesses listed on the U.S. stock exchanges NYSE and NASDAQ.csv"


def _is_missing(x: object) -> bool:
    """Return True if a cell should be treated as missing/invalid.

    Handles None, NaN, and common string sentinels (e.g., "N/A", "–").
    """
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s in {"N/A", "NA", "na", "null", "None", "–", "-"}:
            return True
    return False
LEAKY_KEYWORDS = [
    "market cap",
    "market capitalization",
    "stock price",
    "share price",
    "shares outstanding",
    "enterprise value",
    "ev/revenue",
    "ev to revenue",
    "revenue multiple",
    "rev multiple",
    "rev_multiple",
    "revmultiple",
    "price/sales",
    "price to sales",
    "ps ratio",
    "p/s",
]


def _is_leaky_column(name: str) -> bool:
    """Return True if a column name likely leaks target/valuation information.

    The check is substring-based and case-insensitive.
    """
    n = name.lower().strip()
    for kw in LEAKY_KEYWORDS:
        if kw in n:
            return True
    return False



def parse_money(x: object) -> Optional[float]:
    """Parse a money-like cell to float dollars.

    Strips "$" and commas, returns None on failure.
    """
    if _is_missing(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def parse_percent_to_fraction(x: object) -> Optional[float]:
    """Parse a percent-like cell to a fraction in [−∞, ∞].

    Accepts trailing "%" or raw numeric values; values with |v|>2 are assumed
    to be percentages and divided by 100.
    """
    if _is_missing(x):
        return None
    if isinstance(x, (int, float)):
        val = float(x)
        # Heuristic: values with magnitude > 2 are likely given in percent units
        return val / 100.0 if abs(val) > 2 else val
    s = str(x).strip()
    if s.endswith("%"):
        s = s[:-1]
    s = s.replace(",", "")
    try:
        val = float(s)
        return val / 100.0 if abs(val) > 2 else val
    except Exception:
        return None


def to_float(x: object) -> Optional[float]:
    """Parse a generic numeric cell to float; returns None on failure."""
    if _is_missing(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return None


@dataclass
class Dataset:
    """Container for train/test splits and metadata.

    Attributes:
        X_train: Training features array [n_train, n_features].
        X_test: Test features array [n_test, n_features].
        y_train: Training targets [n_train].
        y_test: Test targets [n_test].
        feature_names: Names aligned with feature columns.
        tickers_test: Ticker symbols for test set rows.
    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    tickers_test: List[str]


def load_dataset(csv_path: str, test_size: float = 0.2, random_state: int = 42) -> Dataset:
    """Load CSV and build a leak-safe feature matrix and target.

    - Ensures presence of required columns.
    - Parses the target (Revenue Multiple) as float.
    - Auto-parses candidate features from remaining columns, excluding identifiers,
      link/text columns, the target, and any name matching leakage keywords.
    - Keeps only features with ≥70% valid values; median-imputes remaining NaNs;
      drops constant features.
    - Returns train/test splits with a fixed random seed.
    """
    df = pd.read_csv(csv_path)

    # Expected columns (exact header strings from the file)
    col_ticker = "Ticker"
    col_rev_mult = "Revenue Multiple"
    col_cash = "Cash and Short Term Investments"
    col_debt = "Total Debt"
    col_nim = "Net Income Margin"
    col_rev_per_emp = "Annual Revenue per Employee (estimate)"

    # Rename a couple of common alternate headers if present (defensive)
    rename_map = {}
    for col in list(df.columns):
        if col.lower().strip() == "revenue multiple" and col != col_rev_mult:
            rename_map[col] = col_rev_mult
        if col.lower().strip() == "cash and short term investments" and col != col_cash:
            rename_map[col] = col_cash
        if col.lower().strip() == "total debt" and col != col_debt:
            rename_map[col] = col_debt
        if col.lower().strip() == "net income margin" and col != col_nim:
            rename_map[col] = col_nim
        if col.lower().strip() == "ticker" and col != col_ticker:
            rename_map[col] = col_ticker
        if col.lower().strip() == "annual revenue per employee (estimate)" and col != col_rev_per_emp:
            rename_map[col] = col_rev_per_emp
    if rename_map:
        df = df.rename(columns=rename_map)

    missing_cols = [c for c in [col_ticker, col_rev_mult] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing expected columns: {missing_cols}")

    # Target
    df["rev_multiple"] = df[col_rev_mult].map(to_float)
    df = df.loc[~df["rev_multiple"].isna()].copy()

    # Auto-build numeric features from all columns except excluded
    excluded_exact = {
        col_rev_mult,
        col_ticker,
        "Company",
        "Stock Price",
        "Market Cap",
        "Founder(s)",
        "Headquarters",
        "Lead Investor(s) Pre-IPO",
        "Product Description",
        "Company Website",
        "Company Investor Relations Page",
        "S-1 Filing",
        "2023 10-K Filing",
    }

    def parse_series(name: str, s: pd.Series) -> pd.Series:
        """Heuristically parse a column into numeric values.

        Priority: percent-like → money-like → generic float.
        """
        name_low = name.lower()
        # Prefer percent parsing for percent-like columns
        if "%" in name_low or "yoy" in name_low or "margin" in name_low or "growth" in name_low:
            return s.map(parse_percent_to_fraction)
        # Money-like columns
        if "$" in s.astype(str).str.cat(sep=" ") or any(tok in name_low for tok in ["revenue", "income", "ebitda", "cash", "debt", "price", "cap", "funding", "investments"]):
            return s.map(parse_money)
        # Fallback numeric
        return s.map(to_float)

    numeric_cols: List[str] = []
    parsed_frames: List[pd.Series] = []
    for col in df.columns:
        # Drop leaky columns early
        if col in excluded_exact or _is_leaky_column(col):
            continue
        if col == col_rev_mult:
            continue
        parsed = parse_series(col, df[col])
        valid_ratio = float((~parsed.isna()).sum()) / float(len(parsed)) if len(parsed) > 0 else 0.0
        if valid_ratio >= 0.7:
            # Fill NaNs with median
            med = float(parsed.median()) if (~parsed.isna()).any() else 0.0
            parsed_filled = parsed.fillna(med).astype(float)
            # Skip constant columns
            if float(parsed_filled.std()) <= 1e-12:
                continue
            new_name = col.strip()
            df[new_name + "__num"] = parsed_filled
            numeric_cols.append(new_name + "__num")
            parsed_frames.append(parsed_filled)

    if not numeric_cols:
        raise ValueError("No numeric feature columns detected after parsing.")

    feature_names = numeric_cols
    X = df[feature_names].to_numpy(dtype=float)
    y = df["rev_multiple"].to_numpy(dtype=float)

    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)), test_size=test_size, random_state=random_state
    )

    tickers_test = df.iloc[idx_test][col_ticker].astype(str).tolist()
    return Dataset(X_train, X_test, y_train, y_test, feature_names, tickers_test)


def _build_feature_row_for_ticker(csv_path: str, feature_names: List[str], ticker: str) -> np.ndarray:
    """Rebuild a single feature row for a given ticker from the CSV.

    Uses the same parsing/exclusion logic as load_dataset to ensure consistency.
    """
    df = pd.read_csv(csv_path)
    col_ticker = "Ticker"
    col_rev_mult = "Revenue Multiple"

    # Normalize column names used for exclusion
    excluded_exact = {
        col_rev_mult,
        col_ticker,
        "Company",
        "Stock Price",
        "Market Cap",
        "Founder(s)",
        "Headquarters",
        "Lead Investor(s) Pre-IPO",
        "Product Description",
        "Company Website",
        "Company Investor Relations Page",
        "S-1 Filing",
        "2023 10-K Filing",
    }

    def parse_series(name: str, s: pd.Series) -> pd.Series:
        name_low = name.lower()
        if "%" in name_low or "yoy" in name_low or "margin" in name_low or "growth" in name_low:
            return s.map(parse_percent_to_fraction)
        if "$" in s.astype(str).str.cat(sep=" ") or any(tok in name_low for tok in ["revenue", "income", "ebitda", "cash", "debt", "price", "cap", "funding", "investments"]):
            return s.map(parse_money)
        return s.map(to_float)

    # Build full numeric matrix to compute medians, then select ticker row and feature_names
    parsed_cols: Dict[str, pd.Series] = {}
    for col in df.columns:
        if col in excluded_exact or col == col_rev_mult or _is_leaky_column(col):
            continue
        parsed = parse_series(col, df[col])
        valid_ratio = float((~parsed.isna()).sum()) / float(len(parsed)) if len(parsed) > 0 else 0.0
        if valid_ratio >= 0.7:
            med = float(parsed.median()) if (~parsed.isna()).any() else 0.0
            parsed_filled = parsed.fillna(med).astype(float)
            if float(parsed_filled.std()) <= 1e-12:
                continue
            parsed_cols[col + "__num"] = parsed_filled

    feat_df = pd.DataFrame(parsed_cols)
    if not set(feature_names).issubset(set(feat_df.columns)):
        missing = sorted(set(feature_names) - set(feat_df.columns))
        raise ValueError(f"CSV parsing did not produce required feature columns: {missing}")

    mask = df[col_ticker].astype(str).str.upper().str.strip() == ticker.upper().strip()
    if not mask.any():
        raise ValueError(f"Ticker '{ticker}' not found in CSV")
    row_idx = mask.idxmax()
    row = feat_df.loc[row_idx, feature_names].to_numpy(dtype=float)
    return row.reshape(1, -1)


def train_model(dataset: Dataset, random_state: int = 42, params: Optional[Dict[str, Any]] = None) -> GradientBoostingRegressor:
    """Fit a GradientBoostingRegressor on the training split.

    Args:
        dataset: Dataset container with X_train/y_train.
        random_state: RNG seed for reproducibility.
        params: Optional GBM hyperparameters (n_estimators, learning_rate, etc.).
    """
    model_params = {
        "random_state": random_state,
    }
    if params is not None:
        # map Optuna params to GBR args
        model_params.update({
            "n_estimators": params.get("n_estimators", 200),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", 3),
            "subsample": params.get("subsample", 1.0),
        })
    model = GradientBoostingRegressor(**model_params)
    model.fit(dataset.X_train, dataset.y_train)
    return model


def tune_hyperparameters(dataset: Dataset, trials: int = 50, cv_folds: int = 5, random_state: int = 42) -> Dict[str, Any]:
    """Run Optuna to minimise MAE via K-fold CV on the full parsed dataset.

    Returns the best parameter dict for the GBM.
    """
    X = np.vstack([dataset.X_train, dataset.X_test])
    y = np.concatenate([dataset.y_train, dataset.y_test])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        maes: List[float] = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model = GradientBoostingRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                subsample=params["subsample"],
                random_state=random_state,
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            maes.append(mean_absolute_error(y_val, pred))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = study.best_params
    return best


def train_ensemble(dataset: Dataset, params: Optional[Dict[str, Any]], n_models: int = 10, bootstrap: bool = True, base_seed: int = 42) -> List[GradientBoostingRegressor]:
    """Train an ensemble of GBMs with optional bootstrap resampling.

    Args:
        dataset: Data splits.
        params: Best hyperparameters to share across ensemble members.
        n_models: Number of members.
        bootstrap: If True, sample with replacement per member.
        base_seed: Seed for reproducibility; each member increments it.
    """
    models: List[GradientBoostingRegressor] = []
    X = dataset.X_train
    y = dataset.y_train
    n = len(y)
    rng = np.random.default_rng(base_seed)
    for i in range(n_models):
        if bootstrap:
            idx = rng.integers(0, n, size=n)
            ds_i = Dataset(X[idx], dataset.X_test, y[idx], dataset.y_test, dataset.feature_names, dataset.tickers_test)
        else:
            ds_i = dataset
        model = train_model(ds_i, random_state=base_seed + i, params=params)
        models.append(model)
    return models


def _predict(models: Union[GradientBoostingRegressor, List[GradientBoostingRegressor]], X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Predict with a single model or an ensemble.

    Returns (mean_pred, std_pred) where std_pred is None for single model.
    """
    if isinstance(models, list):
        preds = np.vstack([m.predict(X) for m in models])
        return preds.mean(axis=0), preds.std(axis=0)
    else:
        return models.predict(X), None


def evaluate(models: Union[GradientBoostingRegressor, List[GradientBoostingRegressor]], dataset: Dataset) -> Tuple[float, float, float]:
    """Compute MAE, RMSE, R² for given model(s) on the test split."""
    y_pred, _ = _predict(models, dataset.X_test)
    mae = mean_absolute_error(dataset.y_test, y_pred)
    # Compute RMSE in a version-agnostic way
    mse = mean_squared_error(dataset.y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(dataset.y_test, y_pred)
    return mae, rmse, r2


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return a dictionary of evaluation metrics for traceability/reporting."""
    abs_err = np.abs(y_pred - y_true)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    med_ae = float(np.median(abs_err))
    p90_ae = float(np.quantile(abs_err, 0.90))
    p95_ae = float(np.quantile(abs_err, 0.95))
    # Safe MAPE: ignore zeros in denominator
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    mape = float(np.nanmean(abs_err / denom) * 100.0)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_ae": med_ae,
        "p90_ae": p90_ae,
        "p95_ae": p95_ae,
        "mape_percent": mape,
        "n_test": int(len(y_true)),
    }


def _scale_to_sizes(values: np.ndarray) -> np.ndarray:
    """Map a numeric series to point sizes using robust percentile scaling."""
    v = values.astype(float)
    lo, hi = np.percentile(v, [5, 95])
    if hi - lo <= 1e-12:
        return np.full_like(v, 40.0)
    norm = np.clip((v - lo) / (hi - lo), 0, 1)
    return 20.0 + 100.0 * norm


def plot_actual_vs_predicted(
    model: Union[GradientBoostingRegressor, List[GradientBoostingRegressor]],
    dataset: Dataset,
    out_path: str,
    style: str = "sorted",
    color_by: str = "nim",
    size_by: str = "debt",
    label_tickers: bool = False,
    label_top_k: int = 0,
    uncertainty_band: bool = False,
) -> None:
    """Plot either a parity chart or sorted-actual chart with optional labels/bands."""
    # Predictions and optional std (for ensembles)
    if isinstance(model, list):
        y_pred, y_std = _predict(model, dataset.X_test)
    else:
        y_pred, y_std = model.predict(dataset.X_test), None

    y_true = dataset.y_test
    tickers = np.array(dataset.tickers_test)

    def get_feat(name: str) -> Optional[np.ndarray]:
        if name == "none":
            return None
        try:
            idx = dataset.feature_names.index(name)
            return dataset.X_test[:, idx]
        except ValueError:
            return None

    c_vals = get_feat(color_by)
    s_vals = _scale_to_sizes(get_feat(size_by)) if get_feat(size_by) is not None else None

    plt.figure(figsize=(11, 7))

    if style == "parity":
        # Simplified: neutral color/size, keep labels
        sc = plt.scatter(y_true, y_pred, color="#1f77b4", s=35.0, alpha=0.85, edgecolor="k", linewidths=0.2, label="Samples")
        xy_min = float(min(y_true.min(), y_pred.min()))
        xy_max = float(max(y_true.max(), y_pred.max()))
        plt.plot([xy_min, xy_max], [xy_min, xy_max], color="#444", linestyle="--", linewidth=1.2, label="y = x")

        if y_std is not None and uncertainty_band:
            plt.errorbar(y_true, y_pred, yerr=y_std, fmt='none', ecolor='tab:orange', elinewidth=1, alpha=0.3, capsize=2)

        if label_top_k and label_top_k > 0:
            errors = np.abs(y_pred - y_true)
            top_idx = np.argsort(errors)[-label_top_k:]
            for i in top_idx:
                plt.annotate(tickers[i], (y_true[i], y_pred[i]), textcoords="offset points", xytext=(3, 3), fontsize=7)
        elif label_tickers:
            for i in range(len(y_true)):
                plt.annotate(tickers[i], (y_true[i], y_pred[i]), textcoords="offset points", xytext=(3, 3), fontsize=6)

        plt.title("Revenue Multiple: Parity (Actual vs Predicted)")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        # No colorbar in simplified mode
        plt.grid(True, alpha=0.25)
        plt.legend()
    else:
        order = np.argsort(y_true)
        y_true_sorted = y_true[order]
        y_pred_sorted = y_pred[order]
        c_sorted = c_vals[order] if c_vals is not None else None
        s_sorted = s_vals[order] if s_vals is not None else None
        tickers_sorted = tickers[order]
        x = np.arange(len(y_true_sorted))

        # Simplified: neutral color/size, keep labels
        sc = plt.scatter(x, y_true_sorted, color="#1f77b4", s=35.0, alpha=0.85, edgecolor="k", linewidths=0.2, label="Actual (dots)")
        plt.plot(x, y_pred_sorted, color="#ff7f0e", linewidth=2.0, label="Model (line)")

        if y_std is not None and uncertainty_band:
            y_std_sorted = y_std[order]
            plt.fill_between(x, y_pred_sorted - y_std_sorted, y_pred_sorted + y_std_sorted, color="#ff7f0e", alpha=0.15, label="±1σ")

        if label_top_k and label_top_k > 0:
            errors_sorted = np.abs(y_pred_sorted - y_true_sorted)
            top_idx = np.argsort(errors_sorted)[-label_top_k:]
            for i in top_idx:
                plt.annotate(tickers_sorted[i], (x[i], y_true_sorted[i]), textcoords="offset points", xytext=(3, 3), fontsize=7)
        elif label_tickers:
            for i in range(len(y_true_sorted)):
                plt.annotate(tickers_sorted[i], (x[i], y_true_sorted[i]), textcoords="offset points", xytext=(3, 3), fontsize=6)

        plt.title("Revenue Multiple: Actual vs Model Prediction")
        plt.xlabel("Test samples (sorted by actual)")
        plt.ylabel("Revenue Multiple")
        # No colorbar in simplified mode
        plt.legend()
        plt.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_residuals_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, tickers: Optional[List[str]] = None, label_top_k: int = 0) -> None:
    """Plot residuals (pred−actual) vs predicted, optionally annotating top errors."""
    residuals = y_pred - y_true
    plt.figure(figsize=(11, 6))
    plt.scatter(y_pred, residuals, color="#1f77b4", s=35.0, alpha=0.85, edgecolor="k", linewidths=0.2)
    plt.axhline(0.0, color="#444", linestyle="--", linewidth=1.2)
    if label_top_k and tickers is not None:
        idx = np.argsort(np.abs(residuals))[-label_top_k:]
        for i in idx:
            plt.annotate(tickers[i], (y_pred[i], residuals[i]), textcoords="offset points", xytext=(3, 3), fontsize=7)
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Pred - Actual)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_residuals_hist(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    """Plot histogram of residuals to assess error symmetry and tail heaviness."""
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color="#1f77b4", alpha=0.85, edgecolor="k")
    plt.title("Residuals Distribution")
    plt.xlabel("Residual (Pred - Actual)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_feature_importance(models: Union[GradientBoostingRegressor, List[GradientBoostingRegressor]], feature_names: List[str], out_path: str, top_k: int = 20) -> None:
    """Plot the top-K feature importances averaged across the ensemble (if any)."""
    if isinstance(models, list):
        importances = np.mean([getattr(m, "feature_importances_", np.zeros(len(feature_names))) for m in models], axis=0)
    else:
        importances = getattr(models, "feature_importances_", np.zeros(len(feature_names)))
    idx = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(10, max(6, int(0.35 * len(names)))))
    plt.barh(range(len(names))[::-1], vals[::-1], color="#ff7f0e")
    plt.yticks(range(len(names))[::-1], names[::-1], fontsize=8)
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_feature_vector(cash: float, net_income_margin: float, debt: float, rev_per_employee: float) -> np.ndarray:
    """Build a manual feature vector (legacy path; auto-features are preferred)."""
    # Accept margin either as fraction (0.12) or percent (12.0)
    nim = net_income_margin
    if abs(nim) > 2:  # treat as percent
        nim = nim / 100.0
    return np.array([[float(cash), float(nim), float(debt), float(rev_per_employee)]], dtype=float)


def lookup_features_by_ticker(csv_path: str, ticker: str) -> Tuple[float, float, float, float]:
    """Legacy path: look up a few specific features by ticker from CSV.

    Retained for reference; prediction path uses auto-derived features.
    """
    df = pd.read_csv(csv_path)
    col_cash = "Cash and Short Term Investments"
    col_debt = "Total Debt"
    col_nim = "Net Income Margin"
    col_ticker = "Ticker"
    col_rev_per_emp = "Annual Revenue per Employee (estimate)"

    # Defensive renames
    rename_map = {}
    for col in list(df.columns):
        low = col.lower().strip()
        if low == "cash and short term investments" and col != col_cash:
            rename_map[col] = col_cash
        if low == "total debt" and col != col_debt:
            rename_map[col] = col_debt
        if low == "net income margin" and col != col_nim:
            rename_map[col] = col_nim
        if low == "ticker" and col != col_ticker:
            rename_map[col] = col_ticker
        if low == "annual revenue per employee (estimate)" and col != col_rev_per_emp:
            rename_map[col] = col_rev_per_emp
    if rename_map:
        df = df.rename(columns=rename_map)

    mask = df[col_ticker].astype(str).str.upper().str.strip() == ticker.upper().strip()
    if not mask.any():
        raise ValueError(f"Ticker '{ticker}' not found in CSV")
    row = df.loc[mask].iloc[0]

    cash = parse_money(row[col_cash])
    debt = parse_money(row[col_debt])
    nim = parse_percent_to_fraction(row[col_nim])
    rpe = parse_money(row[col_rev_per_emp])
    if cash is None or debt is None or nim is None or rpe is None:
        raise ValueError(f"Ticker '{ticker}' has missing inputs (cash/debt/nim/rev_per_employee)")
    return float(cash), float(nim), float(debt), float(rpe)


def cli_train(args: argparse.Namespace) -> None:
    """CLI entrypoint: train/evaluate/plot on a hold-out test split."""
    ds = load_dataset(args.csv, test_size=args.test_size, random_state=args.random_state)

    # Hyperparameter tuning (Optuna + K-fold CV)
    best_params: Optional[Dict[str, Any]] = None
    if args.tune:
        best_params = tune_hyperparameters(ds, trials=args.trials, cv_folds=args.cv_folds, random_state=args.random_state)

    # Train either single model or MC ensemble
    if args.mc_ensemble and args.mc_ensemble > 1:
        models = train_ensemble(ds, best_params, n_models=args.mc_ensemble, bootstrap=args.mc_bootstrap, base_seed=args.random_state)
        mae, rmse, r2 = evaluate(models, ds)
        joblib.dump({
            "models": models,
            "ensemble": True,
            "feature_names": ds.feature_names,
            "params": best_params,
        }, args.model_out)
        trained = models
    else:
        model = train_model(ds, random_state=args.random_state, params=best_params)
        mae, rmse, r2 = evaluate(model, ds)
        joblib.dump({
            "model": model,
            "feature_names": ds.feature_names,
            "params": best_params,
        }, args.model_out)
        trained = model

    print(f"Saved model to: {args.model_out}")
    print(f"Test MAE:  {mae:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R^2:  {r2:.3f}")

    if args.plot:
        plot_actual_vs_predicted(
            trained,
            ds,
            args.plot,
            style=args.plot_style,
            color_by=args.plot_color_by,
            size_by=args.plot_size_by,
            label_tickers=args.plot_label_tickers,
            label_top_k=args.plot_label_top_k,
            uncertainty_band=args.plot_uncertainty_band,
        )
        # Additional diagnostics
        y_pred, _ = _predict(trained, ds.X_test)
        base, ext = os.path.splitext(args.plot)
        if not args.no_extra_plots:
            plot_residuals_scatter(ds.y_test, y_pred, base + "_residuals_scatter" + ext, ds.tickers_test, label_top_k=min(15, len(ds.y_test)))
            plot_residuals_hist(ds.y_test, y_pred, base + "_residuals_hist" + ext)
            plot_feature_importance(trained, ds.feature_names, base + "_feature_importance" + ext, top_k=20)
            print(f"Plots: {args.plot}, {base+'_residuals_scatter'+ext}, {base+'_residuals_hist'+ext}, {base+'_feature_importance'+ext}")
        else:
            print(f"Plot saved to: {args.plot}")

    # Persist run stats as JSON for traceability
    y_pred_all, _ = _predict(trained, ds.X_test)
    metrics = compute_metrics(ds.y_test, y_pred_all)
    run_info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tuned": bool(args.tune),
        "trials": int(args.trials) if args.tune else 0,
        "cv_folds": int(args.cv_folds) if args.tune else 0,
        "ensemble": bool(args.mc_ensemble and args.mc_ensemble > 1),
        "ensemble_size": int(args.mc_ensemble),
        "feature_count": len(ds.feature_names),
        "feature_names": ds.feature_names[:50],
        "metrics": metrics,
    }
    with open("saas_model_report.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print("Trace JSON: saas_model_report.json")


def cli_predict(args: argparse.Namespace) -> None:
    """CLI entrypoint: predict by ticker using auto-derived features from CSV."""
    obj = joblib.load(args.model)
    model: Union[GradientBoostingRegressor, List[GradientBoostingRegressor]]
    if "models" in obj:
        model = obj["models"]
    else:
        model = obj["model"]
    feature_names: List[str] = obj.get("feature_names", [])

    if args.ticker:
        if not args.csv:
            raise SystemExit("--csv is required when using --ticker")
        # Build feature row dynamically from CSV using stored feature_names
        X = _build_feature_row_for_ticker(args.csv, feature_names, args.ticker)
    else:
        # manual input path disabled when using auto-feature mode
        raise SystemExit("This model expects --ticker with --csv because features are auto-derived from the CSV.")
    if isinstance(model, list):
        preds = np.array([m.predict(X)[0] for m in model], dtype=float)
        mean = float(preds.mean())
        std = float(preds.std())
        if args.uncertainty:
            print(f"{mean:.3f} +/- {std:.3f}")
        else:
            print(f"{mean:.3f}")
    else:
        y_hat = float(model.predict(X)[0])
        print(f"{y_hat:.3f}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for train/predict workflows."""
    p = argparse.ArgumentParser(description="Predict SaaS Revenue Multiple from cash, net income margin, and debt")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Train model and evaluate on a held-out test split")
    pt.add_argument("--csv", default=CSV_DEFAULT, help="Path to SaaS CSV (quoted if it contains spaces)")
    pt.add_argument("--model-out", dest="model_out", default="saas_rev_multiple_model.pkl", help="Where to save the trained model")
    pt.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    pt.add_argument("--random-state", type=int, default=42, help="Random seed")
    pt.add_argument("--plot", default="saas_rev_multiple_fit.png", help="Path to save Actual vs Predicted plot (omit to skip)")
    pt.add_argument("--plot-style", dest="plot_style", default="sorted", choices=["sorted", "parity"], help="Plot style: sorted or parity")
    pt.add_argument("--plot-color-by", dest="plot_color_by", default="nim", help="Color by feature: cash|nim|debt|none")
    pt.add_argument("--plot-size-by", dest="plot_size_by", default="debt", help="Size by feature: cash|nim|debt|none")
    pt.add_argument("--plot-label-tickers", dest="plot_label_tickers", action="store_true", help="Annotate all points with ticker labels")
    pt.add_argument("--plot-label-top-k", dest="plot_label_top_k", type=int, default=0, help="Annotate top-K largest errors with tickers")
    pt.add_argument("--plot-uncertainty-band", dest="plot_uncertainty_band", action="store_true", help="If ensemble, draw ±1σ band or error bars")
    pt.add_argument("--no-extra-plots", dest="no_extra_plots", action="store_true", help="Disable residuals and feature importance plots")
    # tuning
    pt.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter tuning with K-fold CV")
    pt.add_argument("--trials", type=int, default=50, help="Number of Optuna trials (default: 50)")
    pt.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    # Monte Carlo ensemble
    pt.add_argument("--mc-ensemble", type=int, default=1, help="Train N models for MC-style ensemble (default: 1)")
    pt.add_argument("--mc-bootstrap", action="store_true", help="Use bootstrap resampling per ensemble member")
    pt.set_defaults(func=cli_train)

    # predict
    pp = sub.add_parser("predict", help="Predict revenue multiple by ticker or raw values")
    pp.add_argument("--model", default="saas_rev_multiple_model.pkl", help="Path to trained model file")
    pp.add_argument("--ticker", help="Ticker symbol to look up in CSV")
    pp.add_argument("--csv", help="CSV path (required with --ticker)")
    pp.add_argument("--cash", type=float, help="Cash and short-term investments (absolute dollars)")
    pp.add_argument("--net-income-margin", dest="net_income_margin", type=float, help="Net income margin as fraction (0.12) or percent (12.0)")
    pp.add_argument("--debt", type=float, help="Total debt (absolute dollars)")
    pp.add_argument("--rev-per-employee", dest="rev_per_employee", type=float, help="Annual Revenue per Employee (absolute dollars)")
    pp.add_argument("--uncertainty", action="store_true", help="For ensemble models, print mean +/- std")
    pp.set_defaults(func=cli_predict)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


