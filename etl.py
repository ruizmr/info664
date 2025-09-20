#!/usr/bin/env python3
"""SEC Fundamentals Database ETL

Builds a local SQLite database of U.S.-listed company fundamentals by pulling
structured data from the SEC XBRL APIs. Minimal, readable, leak-safe.

Usage (examples):
    python etl.py --tickers ADBE,MSFT --db saas_fundamentals.db \
      --user-agent "youremail@example.com"

    python etl.py --tickers-file tickers.txt --db saas_fundamentals.db \
      --user-agent "youremail@example.com"

Technical notes:
  - Rate-limited (≤ 10 req/s) via small sleeps.
  - Requires requests, pandas, sqlalchemy, tqdm (see requirements.txt).
"""

import argparse
import json
import os
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tqdm import tqdm


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
NASDAQ_OTHERLISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

HEADERS_TMPL = {"User-Agent": None, "Accept-Encoding": "gzip, deflate"}

# Preferred tag order per metric (smart resolver)
PREFERRED_TAGS: Dict[str, List[str]] = {
    "revenue": [
        # Prefer ASC 606 revenue tag when available, then fall back
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesAndOtherNet",
    ],
    "ebitda_direct": ["EarningsBeforeInterestTaxesDepreciationAndAmortization"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cash": ["CashCashEquivalentsAndShortTermInvestments"],
    "operating_income": ["OperatingIncomeLoss"],
    "da": ["DepreciationAndAmortization"],
    "cash_components": [
        "CashAndCashEquivalentsAtCarryingValue",
        "ShortTermInvestments",
    ],
    "debt_components": [
        "LongTermDebtNoncurrent",
        "LongTermDebtCurrent",
        "ShortTermBorrowings",
    ],
    "shares": ["CommonStockSharesOutstanding"],
}


def rate_limit_sleep(base_seconds: float = 0.12, jitter_seconds: float = 0.04) -> None:
    """Simple per-request sleep with jitter to stay under ~10 req/sec politely."""
    jitter = random.uniform(-jitter_seconds, jitter_seconds)
    time.sleep(max(0.0, base_seconds + jitter))


def pad_cik(cik: str | int) -> str:
    s = str(cik).strip()
    return s.zfill(10)


def load_ticker_map(session: requests.Session) -> Dict[str, str]:
    resp = session.get(SEC_TICKER_MAP_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    mapping: Dict[str, str] = {}
    # JSON is indexed by integers, each value has {ticker, cik_str, title}
    for _, v in data.items():
        t = str(v.get("ticker", "")).upper().strip()
        cik = pad_cik(v.get("cik_str", ""))
        if t and cik:
            mapping[t] = cik
    return mapping


def fetch_nasdaq_symbol_dirs(session: requests.Session) -> pd.DataFrame:
    """Return a DataFrame of tickers from NASDAQ Trader symbol directories.

    Columns: ticker, exchange, name, etf(bool), test(bool)
    """
    def fetch_table(url: str) -> pd.DataFrame:
        # Robust retry for flaky NASDAQ endpoints
        last_err: Optional[Exception] = None
        for attempt in range(4):
            try:
                rate_limit_sleep(0.15)
                r = session.get(url, timeout=20)
                r.raise_for_status()
                lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("File Creation Time")]
                txt = "\n".join(lines)
                df = pd.read_csv(pd.io.common.StringIO(txt), sep='|')
                return df
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        # If still failing, return empty and let caller fallback to SEC-only
        return pd.DataFrame()

    df_nas = fetch_table(NASDAQ_LISTED_URL)
    df_oth = fetch_table(NASDAQ_OTHERLISTED_URL)

    # Normalize NASDAQ-listed
    a = pd.DataFrame({
        "ticker": df_nas.get("Symbol", pd.Series(dtype=str)).astype(str).str.upper().str.strip(),
        "exchange": "NASDAQ",
        "name": df_nas.get("Security Name", "").astype(str) if not df_nas.empty else pd.Series(dtype=str),
        "etf": (df_nas.get("ETF", "N").astype(str).str.upper().eq("Y") if not df_nas.empty else pd.Series(dtype=bool)),
        "test": (df_nas.get("Test Issue", "N").astype(str).str.upper().eq("Y") if not df_nas.empty else pd.Series(dtype=bool)),
    })
    # Normalize other-listed
    b = pd.DataFrame({
        "ticker": df_oth.get("ACT Symbol", pd.Series(dtype=str)).astype(str).str.upper().str.strip(),
        "exchange": df_oth.get("Exchange", "").astype(str).str.upper().str.strip() if not df_oth.empty else pd.Series(dtype=str),
        "name": df_oth.get("Security Name", "").astype(str) if not df_oth.empty else pd.Series(dtype=str),
        "etf": (df_oth.get("ETF", "N").astype(str).str.upper().eq("Y") if not df_oth.empty else pd.Series(dtype=bool)),
        "test": (df_oth.get("Test Issue", "N").astype(str).str.upper().eq("Y") if not df_oth.empty else pd.Series(dtype=bool)),
    })
    df = pd.concat([a, b], ignore_index=True)
    df = df.dropna(subset=["ticker"])  # remove blanks
    # basic cleanup of special-symbol tickers
    df = df[~df["ticker"].str.contains(r"\^|~|\$|\.", regex=True, na=False)]
    df = df.drop_duplicates("ticker")
    return df


def build_universe(session: requests.Session, include_etfs: bool = False, exchanges: Optional[List[str]] = None) -> pd.DataFrame:
    """Merge SEC map with NASDAQ directories; filter and return a universe DF.

    Returns DF with columns: ticker, cik (may be NaN), exchange, name, etf, test
    """
    sec_map = load_ticker_map(session)
    df_sec = pd.DataFrame({"ticker": list(sec_map.keys()), "cik": list(sec_map.values())})
    df_nas = fetch_nasdaq_symbol_dirs(session)
    if df_nas.empty:
        # fallback to SEC-only universe
        u = df_sec.copy()
        u["exchange"] = "UNKNOWN"
        u["name"] = None
        u["etf"] = False
        u["test"] = False
    else:
        u = pd.merge(df_nas, df_sec, on="ticker", how="left")
    # Filter
    u = u[~u["test"].astype(bool)]
    if not include_etfs:
        u = u[~u["etf"].astype(bool)]
    if exchanges:
        ex_set = set([e.upper() for e in exchanges])
        u = u[u["exchange"].str.upper().isin(ex_set)]
    u = u.sort_values(["exchange", "ticker"]).reset_index(drop=True)
    return u


def fetch_json(session: requests.Session, url: str, max_attempts: int = 3) -> Optional[dict]:
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            rate_limit_sleep(0.12, 0.04)
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            # Exponential backoff with jitter
            time.sleep((attempt + 1) * 0.5 + random.uniform(0.0, 0.25))
    return None


def pick_units(units: Dict[str, List[dict]], prefer_order: List[str]) -> Optional[Tuple[str, List[dict]]]:
    """Return (unit_name, entries) filtered by preferred units only.

    Only accept explicitly allowed units; do not fallback to mixed/other units.
    """
    for u in prefer_order:
        if u in units:
            return u, units[u]
    return None


def _is_nondimensional(entry: dict) -> bool:
    """Return True if the fact entry appears consolidated/non-dimensional.

    Skip entries that include a segment/dimensional breakdown.
    """
    seg = entry.get("segment")
    if isinstance(seg, dict) and seg:
        return False
    if bool(entry.get("dimensional")):
        return False
    return True


def extract_series(facts: dict, tag: str, prefer_units: List[str]) -> pd.DataFrame:
    """Return a normalized DataFrame for a given tag with USD/shares only.

    Columns: [end, val, form, fy, fp, tag, unit, scale_applied, qtrs]
    """
    try:
        tag_obj = facts["facts"]["us-gaap"][tag]
    except Exception:
        return pd.DataFrame(columns=["end", "val", "form", "fy", "fp", "tag", "unit", "scale_applied"]).astype({"val": float})
    units = tag_obj.get("units", {})
    pick = pick_units(units, prefer_units)
    if not pick:
        return pd.DataFrame(columns=["end", "val", "form", "fy", "fp", "tag", "unit", "scale_applied", "qtrs"]).astype({"val": float})
    unit_name, entries = pick
    rows: List[Dict[str, Any]] = []
    for e in entries:
        # Drop dimensional/segment facts unless explicitly needed
        if not _is_nondimensional(e):
            continue
        # Parse numeric and scale
        try:
            raw_val = float(e.get("val"))
        except Exception:
            continue
        scale = e.get("scale")
        try:
            scale_int = int(scale) if scale is not None else 0
        except Exception:
            scale_int = 0
        val = raw_val * (10 ** scale_int)
        rows.append({
            "end": e.get("end"),
            "val": val,
            "form": e.get("form"),
            "fy": e.get("fy"),
            "fp": e.get("fp"),
            "tag": tag,
            "unit": unit_name,
            "scale_applied": scale_int,
            "qtrs": e.get("qtrs"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["end", "form"]).reset_index(drop=True)
        # Enforce consistent dtypes for merge keys across all series
        try:
            df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
        except Exception:
            df["fy"] = pd.Series([None] * len(df), dtype="Int64")
        try:
            df["fp"] = df["fp"].astype("string")
        except Exception:
            pass
        try:
            df["form"] = df["form"].astype("string")
        except Exception:
            pass
        try:
            df["end"] = df["end"].astype("string")
        except Exception:
            pass
        try:
            df["qtrs"] = pd.to_numeric(df["qtrs"], errors="coerce").astype("Int64")
        except Exception:
            pass
    return df


def extract_series_dei(facts: dict, tag: str, prefer_units: List[str]) -> pd.DataFrame:
    """Like extract_series but reads from the DEI namespace."""
    try:
        tag_obj = facts["facts"]["dei"][tag]
    except Exception:
        return pd.DataFrame(columns=["end", "val", "form", "fy", "fp", "tag", "unit", "scale_applied"]).astype({"val": float})
    units = tag_obj.get("units", {})
    pick = pick_units(units, prefer_units)
    if not pick:
        return pd.DataFrame(columns=["end", "val", "form", "fy", "fp", "tag", "unit", "scale_applied", "qtrs"]).astype({"val": float})
    unit_name, entries = pick
    rows: List[Dict[str, Any]] = []
    for e in entries:
        if not _is_nondimensional(e):
            continue
        try:
            raw_val = float(e.get("val"))
        except Exception:
            continue
        scale = e.get("scale")
        try:
            scale_int = int(scale) if scale is not None else 0
        except Exception:
            scale_int = 0
        val = raw_val * (10 ** scale_int)
        rows.append({
            "end": e.get("end"),
            "val": val,
            "form": e.get("form"),
            "fy": e.get("fy"),
            "fp": e.get("fp"),
            "tag": tag,
            "unit": unit_name,
            "scale_applied": scale_int,
            "qtrs": e.get("qtrs"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["end", "form"]).reset_index(drop=True)
        try:
            df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
        except Exception:
            df["fy"] = pd.Series([None] * len(df), dtype="Int64")
        try:
            df["fp"] = df["fp"].astype("string")
        except Exception:
            pass
        try:
            df["form"] = df["form"].astype("string")
        except Exception:
            pass
        try:
            df["end"] = df["end"].astype("string")
        except Exception:
            pass
        try:
            df["qtrs"] = pd.to_numeric(df["qtrs"], errors="coerce").astype("Int64")
        except Exception:
            pass
    return df


def to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Convert YTD 10-Q values to quarterly by differencing with prior quarter within the same form and fiscal year.

    If qtrs == 1 or previous quarter not found, keep the value as-is.
    """
    if df is None or df.empty:
        return df
    d = df.copy()
    if "qtrs" not in d.columns:
        d["qtrs"] = pd.Series([None] * len(d), dtype="Int64")
    try:
        d["qtrs"] = pd.to_numeric(d["qtrs"], errors="coerce").astype("Int64")
    except Exception:
        pass
    parts: List[pd.DataFrame] = []
    fp_order = ["Q1", "Q2", "Q3", "Q4"]
    for (form, fy), g in d.groupby(["form", "fy"], dropna=False):
        g = g.sort_values("end").reset_index(drop=True)
        previous_row_val: Optional[float] = None
        previous_row_qtrs: Optional[int] = None
        previous_row_fp: Optional[str] = None
        quarterly_vals: List[Optional[float]] = []
        for _, row in g.iterrows():
            current_val = float(row["val"]) if pd.notnull(row["val"]) else None
            q = None
            try:
                q = int(row.get("qtrs")) if pd.notnull(row.get("qtrs")) else None
            except Exception:
                q = None
            fp = None
            try:
                fp = str(row.get("fp")) if pd.notnull(row.get("fp")) else None
            except Exception:
                fp = None
            if current_val is None:
                quarterly_vals.append(None)
            else:
                # Determine adjacency by qtrs if present, else by fp sequence
                adjacent = False
                if q is not None and previous_row_qtrs is not None and previous_row_qtrs == (q - 1):
                    adjacent = True
                else:
                    if fp in fp_order and previous_row_fp in fp_order:
                        try:
                            adjacent = (fp_order.index(previous_row_fp) == fp_order.index(fp) - 1)
                        except Exception:
                            adjacent = False
                if adjacent and previous_row_val is not None:
                    quarterly_vals.append(current_val - previous_row_val)
                else:
                    # For Q1 or when we cannot verify adjacency, keep as-is
                    quarterly_vals.append(current_val)
            previous_row_val = current_val
            previous_row_qtrs = q
            previous_row_fp = fp
        g.loc[:, "val"] = quarterly_vals
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def latest_quarters(df: pd.DataFrame, n: int = 4, forms: Optional[List[str]] = None) -> pd.DataFrame:
    if df.empty:
        return df
    d = df
    if forms:
        d = d[d["form"].isin(forms)]
    # Prefer quarterly 10-Q if present; otherwise allow 10-K
    dq = d[d["form"] == "10-Q"]
    if not dq.empty:
        return dq.tail(n)
    if not d.empty:
        return d.tail(n)
    return df.tail(n)


def _compute_lq_annualized_ltm_yoy(q: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if q.empty:
        return None, None, None, None
    lq = float(q.tail(1)["val"].iloc[0])
    annualized = lq * 4.0
    ltm = float(q["val"].tail(4).sum()) if len(q) >= 4 else None
    yoy = None
    if len(q) >= 5:
        prev = q["val"].iloc[-5]
        if prev not in (None, 0):
            yoy = ((q["val"].iloc[-1] - prev) / prev) * 100.0
    return lq, annualized, ltm, yoy


def _merge_on_period(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(a, b, on=["end", "form", "fy", "fp"], suffixes=("_a", "_b"))


def resolve_revenue_series(facts: dict) -> pd.DataFrame:
    for tag in PREFERRED_TAGS["revenue"]:
        df = extract_series(facts, tag, ["USD"])
        if not df.empty:
            return df
    return pd.DataFrame(columns=["end", "val", "form", "fy", "fp", "tag", "unit", "scale_applied"]).astype({"val": float})


def compute_revenue_metrics(facts: dict) -> Tuple[pd.DataFrame, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:
    df_rev = resolve_revenue_series(facts)
    df_rev = to_quarterly(df_rev)
    # Use a longer window so YoY can be computed reliably
    q = latest_quarters(df_rev, 8, forms=["10-Q", "10-K"]).copy()
    return df_rev, _compute_lq_annualized_ltm_yoy(q)


def resolve_ebitda_series(facts: dict) -> pd.DataFrame:
    df_ebitda = extract_series(facts, PREFERRED_TAGS["ebitda_direct"][0], ["USD"])
    if not df_ebitda.empty:
        return to_quarterly(df_ebitda)
    df_op = extract_series(facts, PREFERRED_TAGS["operating_income"][0], ["USD"])  # keep 'val'
    df_da = extract_series(facts, PREFERRED_TAGS["da"][0], ["USD"])  # keep 'val'
    df_op = to_quarterly(df_op).rename(columns={"val": "op"})
    df_da = to_quarterly(df_da).rename(columns={"val": "da"})
    m = _merge_on_period(df_op, df_da)
    if m.empty:
        return pd.DataFrame(columns=["end", "val", "form", "fy", "fp", "tag", "unit", "scale_applied"]).astype({"val": float})
    m["val"] = m.get("op", 0).fillna(0) + m.get("da", 0).fillna(0)
    # Tag provenance: composite
    out = m[["end", "form", "fy", "fp", "val"]].copy()
    out["tag"] = "OperatingIncomeLoss+DepreciationAndAmortization"
    out["unit"] = "USD"
    out["scale_applied"] = 0
    return out


def compute_ebitda_metrics(facts: dict) -> Tuple[pd.DataFrame, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:
    df_ebitda = resolve_ebitda_series(facts)
    q = latest_quarters(df_ebitda, 8, forms=["10-Q", "10-K"]).copy()
    return df_ebitda, _compute_lq_annualized_ltm_yoy(q)


def resolve_net_income_series(facts: dict) -> pd.DataFrame:
    return extract_series(facts, PREFERRED_TAGS["net_income"][0], ["USD"])


def compute_net_income_metrics(facts: dict) -> Tuple[pd.DataFrame, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:
    df = resolve_net_income_series(facts)
    df = to_quarterly(df)
    q = latest_quarters(df, 8, forms=["10-Q", "10-K"]).copy()
    return df, _compute_lq_annualized_ltm_yoy(q)


def resolve_cash_series(facts: dict) -> pd.DataFrame:
    df_direct = extract_series(facts, PREFERRED_TAGS["cash"][0], ["USD"])
    if not df_direct.empty:
        return df_direct
    cash_only = extract_series(facts, PREFERRED_TAGS["cash_components"][0], ["USD"])  # single series
    sti_only = extract_series(facts, PREFERRED_TAGS["cash_components"][1], ["USD"])  # single series
    if cash_only.empty and sti_only.empty:
        return pd.DataFrame(columns=["end", "val", "form", "fy", "fp", "tag", "unit", "scale_applied"]).astype({"val": float})
    # Stack-and-sum by period to avoid losing data when one component is missing
    parts: List[pd.DataFrame] = []
    for d in (cash_only, sti_only):
        if not d.empty:
            parts.append(d[["end", "form", "fy", "fp", "val"]].copy())
    m = pd.concat(parts, ignore_index=True)
    g = m.groupby(["end", "form", "fy", "fp"], as_index=False)["val"].sum()
    g["tag"] = "CashAndCashEquivalentsAtCarryingValue(+ShortTermInvestments)"
    g["unit"] = "USD"
    g["scale_applied"] = 0
    return g


def resolve_debt_series(facts: dict) -> pd.DataFrame:
    components: List[pd.DataFrame] = []
    for tag in PREFERRED_TAGS["debt_components"]:
        d = extract_series(facts, tag, ["USD"])  # keeps 'val'
        if not d.empty:
            components.append(d[["end", "form", "fy", "fp", "val"]].copy())
    if not components:
        return pd.DataFrame(columns=["end", "val", "form", "fy", "fp", "tag", "unit", "scale_applied"]).astype({"val": float})
    m = pd.concat(components, ignore_index=True)
    g = m.groupby(["end", "form", "fy", "fp"], as_index=False)["val"].sum()
    g["tag"] = "+".join(PREFERRED_TAGS["debt_components"])  # provenance
    g["unit"] = "USD"
    g["scale_applied"] = 0
    return g


def resolve_shares_series(facts: dict) -> pd.DataFrame:
    df = extract_series(facts, PREFERRED_TAGS["shares"][0], ["shares"])  # prefer GAAP shares
    if not df.empty:
        return df
    # Fallback to DEI EntityCommonStockSharesOutstanding when GAAP tag is absent
    df_dei = extract_series_dei(facts, "EntityCommonStockSharesOutstanding", ["shares"])  # DEI
    return df_dei


def resolve_deferred_revenue_series(facts: dict) -> pd.DataFrame:
    # Prefer ASC 606: ContractWithCustomerLiabilityCurrent (+ Noncurrent)
    current = extract_series(facts, "ContractWithCustomerLiabilityCurrent", ["USD"]).rename(columns={"val": "cur"})
    noncurrent = extract_series(facts, "ContractWithCustomerLiabilityNoncurrent", ["USD"]).rename(columns={"val": "ncur"})
    if current.empty and noncurrent.empty:
        # Legacy tags fallback
        cur2 = extract_series(facts, "DeferredRevenueCurrent", ["USD"]).rename(columns={"val": "cur"})
        ncur2 = extract_series(facts, "DeferredRevenueNoncurrent", ["USD"]).rename(columns={"val": "ncur"})
        current, noncurrent = cur2, ncur2
        if current.empty and noncurrent.empty:
            # Last resort single total tag
            total = extract_series(facts, "DeferredRevenue", ["USD"])  # single series
            return total
    parts: List[pd.DataFrame] = []
    for d in (current, noncurrent):
        if not d.empty:
            # Normalize value column name to 'val'
            dd = d.rename(columns={"cur": "val", "ncur": "val"})
            parts.append(dd[["end", "form", "fy", "fp", "val"]].copy())
    if not parts:
        return pd.DataFrame(columns=["end", "val", "form", "fy", "fp", "tag", "unit", "scale_applied"]).astype({"val": float})
    m = pd.concat(parts, ignore_index=True)
    g = m.groupby(["end", "form", "fy", "fp"], as_index=False)["val"].sum()
    g["tag"] = "DeferredRevenue(Total)"
    g["unit"] = "USD"
    g["scale_applied"] = 0
    return g


def extract_balance_sheet(facts: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return resolve_cash_series(facts), resolve_debt_series(facts), resolve_shares_series(facts)


def _derive_stamps_from_series(series_df: pd.DataFrame) -> Dict[str, Any]:
    if series_df is None or series_df.empty:
        return {"report_date": None, "fy": None, "fp": None}
    q = latest_quarters(series_df, 1, forms=["10-Q", "10-K"]).copy()
    if q.empty:
        return {"report_date": None, "fy": None, "fp": None}
    return {"report_date": q["end"].iloc[0], "fy": q["fy"].iloc[0], "fp": q["fp"].iloc[0]}


def _compute_margin(numer_df: pd.DataFrame, denom_df: pd.DataFrame) -> Optional[float]:
    if numer_df.empty or denom_df.empty:
        return None
    # Align on exact period/form
    m = _merge_on_period(latest_quarters(numer_df, 1, forms=["10-Q", "10-K"]), latest_quarters(denom_df, 1, forms=["10-Q", "10-K"]))
    if m.empty:
        return None
    rev = float(m["val_b"].iloc[0])
    num = float(m["val_a"].iloc[0])
    if not math.isfinite(rev) or rev <= 0.0:
        return None
    margin = (num / rev) * 100.0
    if not math.isfinite(margin) or abs(margin) > 100.0:
        return None
    return margin


def latest_filings(submissions: dict, cik: str) -> Dict[str, Optional[str]]:
    urls = {"10-Q": None, "10-K": None, "S-1": None}
    try:
        recent = submissions["filings"]["recent"]
        forms = recent.get("form", [])
        accns = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        dates = recent.get("filingDate", [])
        for want in ["10-Q", "10-K", "S-1"]:
            for i, f in enumerate(forms):
                if f == want:
                    accn = accns[i].replace("-", "")
                    doc = docs[i]
                    urls[want] = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn}/{doc}"
                    break
    except Exception:
        pass
    return urls


@dataclass
class CompanyRow:
    cik: str
    ticker: str
    name: Optional[str]
    business_state: Optional[str]
    price: Optional[float]
    market_cap: Optional[float]


def ensure_schema(engine: Engine, schema_path: str) -> None:
    with open(schema_path, "r") as f:
        ddl = f.read()
    with engine.begin() as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))


def upsert_company(engine: Engine, row: CompanyRow) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO companies (cik, ticker, name, business_state, last_price, market_cap)
                VALUES (:cik, :ticker, :name, :state, :price, :mcap)
                ON CONFLICT(cik) DO UPDATE SET
                  ticker=excluded.ticker,
                  name=excluded.name,
                  business_state=excluded.business_state,
                  last_price=excluded.last_price,
                  market_cap=excluded.market_cap
                """
            ),
            {"cik": row.cik, "ticker": row.ticker, "name": row.name, "state": row.business_state, "price": row.price, "mcap": row.market_cap},
        )


def insert_fundamentals(engine: Engine, cik: str, metrics: Dict[str, Optional[float]], stamps: Dict[str, Any]) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO fundamentals (
                  cik, report_date, fiscal_year, fiscal_period,
                  revenue_lq, revenue_annualized, revenue_ltm, revenue_yoy,
                  ebitda_lq, ebitda_annualized, ebitda_ltm, ebitda_yoy, ebitda_margin,
                  net_income_lq, net_income_annualized, net_income_ltm, net_income_yoy, net_income_margin,
                  cash_sti, total_debt, shares_outstanding, stock_price, market_cap,
                  deferred_revenue
                ) VALUES (
                  :cik, :report_date, :fy, :fp,
                  :rev_lq, :rev_ann, :rev_ltm, :rev_yoy,
                  :ebitda_lq, :ebitda_ann, :ebitda_ltm, :ebitda_yoy, :ebitda_margin,
                  :ni_lq, :ni_ann, :ni_ltm, :ni_yoy, :ni_margin,
                  :cash, :debt, :shares, :price, :mcap,
                  :deferred_revenue
                )
                ON CONFLICT(cik, report_date) DO UPDATE SET
                  fiscal_year=excluded.fiscal_year,
                  fiscal_period=excluded.fiscal_period,
                  revenue_lq=excluded.revenue_lq,
                  revenue_annualized=excluded.revenue_annualized,
                  revenue_ltm=excluded.revenue_ltm,
                  revenue_yoy=excluded.revenue_yoy,
                  ebitda_lq=excluded.ebitda_lq,
                  ebitda_annualized=excluded.ebitda_annualized,
                  ebitda_ltm=excluded.ebitda_ltm,
                  ebitda_yoy=excluded.ebitda_yoy,
                  ebitda_margin=excluded.ebitda_margin,
                  net_income_lq=excluded.net_income_lq,
                  net_income_annualized=excluded.net_income_annualized,
                  net_income_ltm=excluded.net_income_ltm,
                  net_income_yoy=excluded.net_income_yoy,
                  net_income_margin=excluded.net_income_margin,
                  cash_sti=excluded.cash_sti,
                  total_debt=excluded.total_debt,
                  shares_outstanding=excluded.shares_outstanding,
                  stock_price=excluded.stock_price,
                  market_cap=excluded.market_cap,
                  deferred_revenue=excluded.deferred_revenue
                """
            ),
            {
                "cik": cik,
                "report_date": stamps.get("report_date"),
                "fy": stamps.get("fy"),
                "fp": stamps.get("fp"),
                "rev_lq": metrics.get("rev_lq"),
                "rev_ann": metrics.get("rev_annualized"),
                "rev_ltm": metrics.get("rev_ltm"),
                "rev_yoy": metrics.get("rev_yoy"),
                "ebitda_lq": metrics.get("ebitda_lq"),
                "ebitda_ann": metrics.get("ebitda_annualized"),
                "ebitda_ltm": metrics.get("ebitda_ltm"),
                "ebitda_yoy": metrics.get("ebitda_yoy"),
                "ebitda_margin": metrics.get("ebitda_margin"),
                "ni_lq": metrics.get("net_income_lq"),
                "ni_ann": metrics.get("net_income_annualized"),
                "ni_ltm": metrics.get("net_income_ltm"),
                "ni_yoy": metrics.get("net_income_yoy"),
                "ni_margin": metrics.get("net_income_margin"),
                "cash": metrics.get("cash_sti"),
                "debt": metrics.get("total_debt"),
                "shares": metrics.get("shares_outstanding"),
                "price": metrics.get("stock_price"),
                "mcap": metrics.get("market_cap"),
                "deferred_revenue": metrics.get("deferred_revenue"),
            },
        )


def upsert_filings(engine: Engine, cik: str, links: Dict[str, Optional[str]]) -> None:
    with engine.begin() as conn:
        for ftype in ["10-Q", "10-K", "S-1"]:
            url = links.get(ftype)
            if not url:
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO filings (cik, filing_type, url)
                    VALUES (:cik, :ftype, :url)
                    ON CONFLICT(cik, filing_type, url) DO NOTHING
                    """
                ),
                {"cik": cik, "ftype": ftype, "url": url},
            )


def maybe_price_and_mcap(submissions: dict, shares: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    # Stock price not directly from SEC; placeholder for optional retrieval
    # Here we leave it None, and compute market cap when price available.
    return None, None if shares is None else None


def run_etl(tickers: List[str], db_path: str, user_agent: str) -> None:
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    ensure_schema(engine, schema_path)

    sess = requests.Session()
    headers = HEADERS_TMPL.copy()
    headers["User-Agent"] = user_agent
    sess.headers.update(headers)

    print("Resolving ticker→CIK map…")
    mapping = load_ticker_map(sess)

    for ticker in tqdm(tickers, desc="ETL", unit="co"):
        t = ticker.upper().strip()
        cik = mapping.get(t)
        if not cik:
            print(f"[WARN] No CIK for ticker {t}")
            continue

        facts = fetch_json(sess, SEC_COMPANY_FACTS_URL.format(cik=cik))
        subs = fetch_json(sess, SEC_SUBMISSIONS_URL.format(cik=cik))
        if facts is None or subs is None:
            print(f"[WARN] Missing SEC data for {t}/{cik}")
            continue

        # Company metadata
        name = subs.get("name") if isinstance(subs.get("name"), str) else subs.get("entityType")
        state = None
        try:
            state = subs.get("addresses", {}).get("business", {}).get("stateOrCountry")
        except Exception:
            state = None
        state_incorp = subs.get("stateOfIncorporation")
        sic = subs.get("sic")
        sic_desc = subs.get("sicDescription")
        fy_end = subs.get("fiscalYearEnd")

        # Fundamentals - series and metrics
        df_rev, (rev_lq, rev_ann, rev_ltm, rev_yoy) = compute_revenue_metrics(facts)
        df_ebitda, (ebitda_lq, ebitda_ann, ebitda_ltm, ebitda_yoy) = compute_ebitda_metrics(facts)
        df_ni, (ni_lq, ni_ann, ni_ltm, ni_yoy) = compute_net_income_metrics(facts)
        df_cash, df_debt, df_shares = extract_balance_sheet(facts)
        df_def = resolve_deferred_revenue_series(facts)
        # Resolve latest values for balance sheet items
        def _latest_val(df: pd.DataFrame) -> Optional[float]:
            d = latest_quarters(df, 1, ["10-Q", "10-K"]).copy()
            if d.empty:
                return None
            try:
                return float(d["val"].iloc[0])
            except Exception:
                return None

        cash_val = _latest_val(df_cash)
        debt_val = _latest_val(df_debt)
        shares_val = _latest_val(df_shares)
        deferred_rev_val = _latest_val(df_def)

        price, mcap = maybe_price_and_mcap(subs, shares_val)
        if mcap is None and price is not None and shares_val is not None:
            mcap = price * shares_val

        # Margins with strict alignment and guardrails
        ebitda_margin = _compute_margin(df_ebitda, df_rev)
        ni_margin = _compute_margin(df_ni, df_rev)

        # Report stamps from chosen revenue series, with fallbacks
        stamps = _derive_stamps_from_series(df_rev)
        if stamps.get("report_date") is None:
            stamps = _derive_stamps_from_series(df_ni)
        if stamps.get("report_date") is None:
            # Fall back to operating income if available (through EBITDA components)
            df_op = extract_series(facts, PREFERRED_TAGS["operating_income"][0], ["USD"])
            stamps = _derive_stamps_from_series(df_op)
        if stamps.get("report_date") is None:
            # Last resort: latest filing date from submissions
            try:
                recent = subs.get("filings", {}).get("recent", {})
                dates = recent.get("filingDate", [])
                if dates:
                    # Use the latest available date
                    try:
                        latest_date = max(dates)
                    except Exception:
                        latest_date = dates[0]
                    stamps = {"report_date": latest_date, "fy": None, "fp": None}
            except Exception:
                pass

        # Sanitize stamp types
        rd = stamps.get("report_date")
        fy_val = stamps.get("fy")
        fp_val = stamps.get("fp")
        try:
            fy_clean: Optional[int] = int(fy_val) if fy_val is not None and str(fy_val).strip() != "" else None
        except Exception:
            fy_clean = None
        fp_clean: Optional[str] = str(fp_val) if fp_val is not None else None
        rd_clean: Optional[str] = str(rd) if rd is not None else None
        stamps = {"report_date": rd_clean, "fy": fy_clean, "fp": fp_clean}

        # Company row
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO companies (cik, ticker, name, business_state, state_incorp, sic, sic_description, fiscal_year_end, last_price, market_cap)
                    VALUES (:cik, :ticker, :name, :state, :sinc, :sic, :sicd, :fye, :price, :mcap)
                    ON CONFLICT(cik) DO UPDATE SET
                      ticker=excluded.ticker,
                      name=excluded.name,
                      business_state=excluded.business_state,
                      state_incorp=excluded.state_incorp,
                      sic=excluded.sic,
                      sic_description=excluded.sic_description,
                      fiscal_year_end=excluded.fiscal_year_end,
                      last_price=excluded.last_price,
                      market_cap=excluded.market_cap
                    """
                ),
                {"cik": cik, "ticker": t, "name": name, "state": state, "sinc": state_incorp, "sic": sic, "sicd": sic_desc, "fye": fy_end, "price": price, "mcap": mcap},
            )
        insert_fundamentals(engine, cik, {
            "rev_lq": rev_lq, "rev_annualized": rev_ann, "rev_ltm": rev_ltm, "rev_yoy": rev_yoy,
            "ebitda_lq": ebitda_lq, "ebitda_annualized": ebitda_ann, "ebitda_ltm": ebitda_ltm, "ebitda_yoy": ebitda_yoy, "ebitda_margin": ebitda_margin,
            "net_income_lq": ni_lq, "net_income_annualized": ni_ann, "net_income_ltm": ni_ltm, "net_income_yoy": ni_yoy, "net_income_margin": ni_margin,
            "cash_sti": cash_val,
            "total_debt": debt_val,
            "shares_outstanding": shares_val,
            "stock_price": price, "market_cap": mcap,
            "deferred_revenue": deferred_rev_val,
        }, stamps)

        # Traceability audit (chosen LQ rows)
        def _lq_row(df: pd.DataFrame) -> Optional[dict]:
            d = latest_quarters(df, 1, ["10-Q", "10-K"]).copy()
            if d.empty:
                return None
            row = d.tail(1).iloc[0].to_dict()
            return row

        audit_items: List[Tuple[str, Optional[dict]]] = [
            ("revenue_lq", _lq_row(df_rev)),
            ("ebitda_lq", _lq_row(df_ebitda)),
            ("net_income_lq", _lq_row(df_ni)),
            ("cash_sti", _lq_row(df_cash)),
            ("total_debt", _lq_row(df_debt)),
            ("shares_outstanding", _lq_row(df_shares)),
            ("deferred_revenue", _lq_row(df_def)),
        ]
        try:
            with engine.begin() as conn:
                for metric_name, row in audit_items:
                    if not row:
                        continue
                    conn.execute(
                        text(
                            """
                            INSERT INTO fundamentals_audit (cik, report_date, fiscal_year, fiscal_period, metric, value, source_tag, unit, scale_applied, form, fy, fp)
                            VALUES (:cik, :report_date, :fy, :fp, :metric, :value, :tag, :unit, :scale_applied, :form, :s_fy, :s_fp)
                            """
                        ),
                        {
                            "cik": cik,
                            "report_date": stamps.get("report_date"),
                            "fy": stamps.get("fy"),
                            "fp": stamps.get("fp"),
                            "metric": metric_name,
                            "value": row.get("val"),
                            "tag": row.get("tag"),
                            "unit": row.get("unit"),
                            "scale_applied": row.get("scale_applied", 0),
                            "form": row.get("form"),
                            "s_fy": row.get("fy"),
                            "s_fp": row.get("fp"),
                        },
                    )
        except Exception:
            pass
        upsert_filings(engine, cik, latest_filings(subs, cik))

        # Optional: store raw facts subset (commented out to keep DB smaller)
        # try:
        #     with engine.begin() as conn:
        #         for concept, obj in facts.get("facts", {}).get("us-gaap", {}).items():
        #             for unit, arr in obj.get("units", {}).items():
        #                 for e in arr[-4:]:  # last 4 only
        #                     conn.execute(
        #                         text("INSERT OR IGNORE INTO facts_raw (cik, concept, unit, end, fy, fp, form, val, dims) VALUES (:cik,:c,:u,:end,:fy,:fp,:f,:v,:d)"),
        #                         {"cik": cik, "c": concept, "u": unit, "end": e.get("end"), "fy": e.get("fy"), "fp": e.get("fp"), "f": e.get("form"), "v": e.get("val"), "d": json.dumps(e.get("dimensional"))},
        #                     )
        # except Exception:
        #     pass

    print("ETL complete.")


def parse_tickers_arg(t: Optional[str], file_path: Optional[str]) -> List[str]:
    tickers: List[str] = []
    if t:
        tickers.extend([x.strip() for x in t.split(",") if x.strip()])
    if file_path:
        with open(file_path, "r") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    tickers.append(s)
    # de-dup preserve order
    seen = set()
    uniq: List[str] = []
    for s in tickers:
        u = s.upper()
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(description="SEC Fundamentals ETL → SQLite")
    ap.add_argument("--tickers", help="Comma-separated tickers (e.g., ADBE,MSFT or ALL for all mapped tickers)")
    ap.add_argument("--tickers-file", help="Path to text file with one ticker per line")
    ap.add_argument("--db", default="saas_fundamentals.db", help="SQLite DB file path")
    ap.add_argument("--user-agent", required=True, help="Polite SEC header: your email")
    ap.add_argument("--build-universe", action="store_true", help="Build universe.json by merging SEC + NASDAQ directories and exit")
    ap.add_argument("--include-etfs", action="store_true", help="Include ETFs in universe")
    ap.add_argument("--exchanges", help="Comma-separated exchanges to keep (e.g., NASDAQ,NYSE)")
    args = ap.parse_args()

    if args.build_universe:
        sess = requests.Session(); h=HEADERS_TMPL.copy(); h["User-Agent"]=args.user_agent; sess.headers.update(h)
        exch = [e.strip() for e in args.exchanges.split(",")] if args.exchanges else None
        uni = build_universe(sess, include_etfs=args.include_etfs, exchanges=exch)
        uni.to_json("universe.json", orient="records", indent=2)
        print(f"universe.json written with {len(uni)} symbols.")
        return

    # Handle special ALL flag before parsing comma lists
    if args.tickers and args.tickers.strip().upper() == "ALL":
        sess = requests.Session()
        h = HEADERS_TMPL.copy(); h["User-Agent"] = args.user_agent; sess.headers.update(h)
        mapping = load_ticker_map(sess)
        tickers = sorted(mapping.keys())
    else:
        tickers = parse_tickers_arg(args.tickers, args.tickers_file)
    if not tickers:
        raise SystemExit("Provide --tickers or --tickers-file")
    run_etl(tickers, args.db, args.user_agent)


if __name__ == "__main__":
    main()


