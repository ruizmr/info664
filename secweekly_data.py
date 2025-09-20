#!/usr/bin/env python3
"""Data loader utilities for SECweekly fundamentals → PyTorch-ready tensors.

Leakage-safe feature selection and optional validation target (P/S multiple).
Supports both DuckDB (`*.duckdb`) and SQLite (`*.db`) sources via DuckDB.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import duckdb  # type: ignore
except Exception as e:  # pragma: no cover
    duckdb = None  # Will raise at runtime if used without install


LEAKY_PATTERNS = (
    "price",
    "market_cap",
    "marketcap",
    "stock_price",
    "shares_outstanding",
    "shares",
)


def _connect_duckdb(db_path: str):
    if duckdb is None:
        raise RuntimeError("duckdb is required. Install with `pip install duckdb`.`)" )
    return duckdb.connect(db_path if db_path.endswith(".duckdb") else None)


def _attach_sqlite(con, sqlite_path: str) -> str:
    """Attach a SQLite database as schema `s` via DuckDB sqlite_scanner."""
    con.execute("INSTALL sqlite_scanner;")
    con.execute("LOAD sqlite_scanner;")
    con.execute(f"ATTACH '{sqlite_path}' AS s (TYPE SQLITE);")
    return "s"


def _latest_snapshot_query(schema: Optional[str] = None) -> str:
    s = (schema + ".") if schema else ""
    return f"""
    with latest as (
      select f.* from (
        select cik, max(report_date) as mr from {s}fundamentals group by cik
      ) j join {s}fundamentals f on f.cik=j.cik and f.report_date=j.mr
    )
    select l.*, c.ticker from latest l join {s}companies c using(cik)
    where l.report_date is not null
    """


def load_secweekly_dataframe(db_path: str) -> pd.DataFrame:
    """Load the latest-per-company snapshot from SEC DB into a DataFrame.

    Returns a DataFrame including identifier columns and all numeric fundamentals.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    con = _connect_duckdb(db_path)
    try:
        if db_path.endswith(".duckdb"):
            df = con.execute(_latest_snapshot_query(schema=None)).fetchdf()
        else:
            schema = _attach_sqlite(con, db_path)
            df = con.execute(_latest_snapshot_query(schema=schema)).fetchdf()
    finally:
        con.close()
    # Ensure consistent dtypes
    if "report_date" in df.columns:
        df["report_date"] = df["report_date"].astype(str)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)
    return df


def leakage_safe_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of numeric feature columns with strict leakage guards."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    blocked = set([c for c in numeric_cols for p in LEAKY_PATTERNS if p in c.lower()])
    # Always exclude identifiers / targets / stamps
    always_exclude = {"id", "fiscal_year", "fiscal_period"}
    return [c for c in numeric_cols if c not in blocked and c not in always_exclude]


@dataclass
class PreparedData:
    features: np.ndarray
    feature_names: List[str]
    ids: pd.DataFrame  # contains cik, ticker, report_date
    y_teacher: Optional[np.ndarray] = None
    y_ps_validation: Optional[np.ndarray] = None
    scaler_mean: Optional[np.ndarray] = None
    scaler_std: Optional[np.ndarray] = None


def compute_ps_validation(df: pd.DataFrame) -> np.ndarray:
    """Compute Price/Sales (MarketCap / Revenue LTM) as validation signal.

    Returns np.ndarray with NaN where unavailable.
    """
    mcap = df.get("market_cap")
    rev = df.get("revenue_ltm")
    if mcap is None or rev is None:
        return np.full(len(df), np.nan, dtype=float)
    ps = mcap.values.astype(float)
    denom = rev.values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ps / denom
    # Drop non-finite
    out[~np.isfinite(out)] = np.nan
    return out


def prepare_supervised_arrays(
    df: pd.DataFrame,
    teacher_preds: Optional[pd.DataFrame] = None,
) -> PreparedData:
    """Prepare arrays for model training/evaluation.

    teacher_preds: optional DataFrame with columns [ticker, pred] to use as KD targets.
    """
    # Identify and select features
    feat_cols = leakage_safe_feature_columns(df)
    # Remove obvious non-features even if numeric
    for c in ("cik",):
        if c in feat_cols:
            feat_cols.remove(c)
    X = df[feat_cols].copy()
    # Simple imputation: median per column
    X = X.fillna(X.median(numeric_only=True))
    Xv = X.values.astype(np.float32)
    # Standardization stats
    mean = Xv.mean(axis=0)
    std = Xv.std(axis=0)
    std[std == 0.0] = 1.0
    Xv = (Xv - mean) / std

    ids = df[[c for c in ("cik", "ticker", "report_date") if c in df.columns]].copy()

    y_teacher = None
    if teacher_preds is not None and not teacher_preds.empty:
        # Merge by ticker if available, else by cik
        cols = [c for c in ("ticker", "cik") if c in ids.columns and c in teacher_preds.columns]
        if cols:
            m = ids.join(teacher_preds.set_index(cols), on=cols, how="left")
            if "pred" in m.columns:
                arr = m["pred"].values.astype(np.float32)
                arr[~np.isfinite(arr)] = np.nan
                y_teacher = arr

    y_ps = compute_ps_validation(df)

    return PreparedData(
        features=Xv,
        feature_names=feat_cols,
        ids=ids,
        y_teacher=y_teacher,
        y_ps_validation=y_ps,
        scaler_mean=mean.astype(np.float32),
        scaler_std=std.astype(np.float32),
    )


def chronological_split(df: pd.DataFrame, frac_val: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by report_date chronologically: older → train, most recent → val."""
    if "report_date" not in df.columns:
        # Fallback to random split
        idx = np.arange(len(df))
        rs = np.random.RandomState(42)
        rs.shuffle(idx)
        cut = int(len(idx) * (1.0 - frac_val))
        return df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    d = df.dropna(subset=["report_date"]).copy()
    d["report_date"] = pd.to_datetime(d["report_date"], errors="coerce")
    d = d.sort_values("report_date")
    cut = int(len(d) * (1.0 - frac_val))
    return d.iloc[:cut], d.iloc[cut:]


