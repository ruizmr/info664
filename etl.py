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
NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
NASDAQ_OTHERLISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

HEADERS_TMPL = {"User-Agent": None, "Accept-Encoding": "gzip, deflate"}


def rate_limit_sleep(seconds: float = 0.2) -> None:
    """Simple per-request sleep to stay under 10 requests/sec."""
    time.sleep(seconds)


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
        rate_limit_sleep(0.15)
        r = session.get(url, timeout=30)
        r.raise_for_status()
        # Some files have a footer line; pandas can parse with sep='|'
        lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("File Creation Time")]
        txt = "\n".join(lines)
        df = pd.read_csv(pd.io.common.StringIO(txt), sep='|')
        return df

    df_nas = fetch_table(NASDAQ_LISTED_URL)
    df_oth = fetch_table(NASDAQ_OTHERLISTED_URL)

    # Normalize NASDAQ-listed
    a = pd.DataFrame({
        "ticker": df_nas["Symbol"].astype(str).str.upper().str.strip(),
        "exchange": "NASDAQ",
        "name": df_nas.get("Security Name", "").astype(str),
        "etf": df_nas.get("ETF", "N").astype(str).str.upper().eq("Y"),
        "test": df_nas.get("Test Issue", "N").astype(str).str.upper().eq("Y"),
    })
    # Normalize other-listed
    b = pd.DataFrame({
        "ticker": df_oth.get("ACT Symbol", "").astype(str).str.upper().str.strip(),
        "exchange": df_oth.get("Exchange", "").astype(str).str.upper().str.strip(),
        "name": df_oth.get("Security Name", "").astype(str),
        "etf": df_oth.get("ETF", "N").astype(str).str.upper().eq("Y"),
        "test": df_oth.get("Test Issue", "N").astype(str).str.upper().eq("Y"),
    })
    df = pd.concat([a, b], ignore_index=True)
    df = df.dropna(subset=["ticker"])  # remove blanks
    df = df[~df["ticker"].str.contains("\^|~|\$|\.")]  # basic cleanup
    df = df.drop_duplicates("ticker")
    return df


def build_universe(session: requests.Session, include_etfs: bool = False, exchanges: Optional[List[str]] = None) -> pd.DataFrame:
    """Merge SEC map with NASDAQ directories; filter and return a universe DF.

    Returns DF with columns: ticker, cik (may be NaN), exchange, name, etf, test
    """
    sec_map = load_ticker_map(session)
    df_sec = pd.DataFrame({"ticker": list(sec_map.keys()), "cik": list(sec_map.values())})
    df_nas = fetch_nasdaq_symbol_dirs(session)
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


def fetch_json(session: requests.Session, url: str) -> Optional[dict]:
    try:
        rate_limit_sleep(0.15)
        r = session.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def pick_units(units: Dict[str, List[dict]], prefer_order: List[str]) -> Optional[List[dict]]:
    for u in prefer_order:
        if u in units:
            return units[u]
    # pick any
    if units:
        return next(iter(units.values()))
    return None


def extract_series(facts: dict, tag: str, prefer_units: List[str]) -> pd.DataFrame:
    """Return a DataFrame with columns [end, val, form, fy, fp] for a given tag."""
    try:
        tag_obj = facts["facts"]["us-gaap"][tag]
    except Exception:
        return pd.DataFrame(columns=["end", "val", "form", "fy", "fp"]).astype({"val": float})
    units = tag_obj.get("units", {})
    entries = pick_units(units, prefer_units) or []
    rows: List[Dict[str, Any]] = []
    for e in entries:
        end = e.get("end")
        try:
            val = float(e.get("val"))
        except Exception:
            continue
        rows.append({
            "end": end,
            "val": val,
            "form": e.get("form"),
            "fy": e.get("fy"),
            "fp": e.get("fp"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("end").reset_index(drop=True)
    return df


def latest_quarters(df: pd.DataFrame, n: int = 4, forms: Optional[List[str]] = None) -> pd.DataFrame:
    if df.empty:
        return df
    d = df
    if forms:
        d = d[d["form"].isin(forms)]
    if d.empty:
        return df.tail(n)
    return d.tail(n)


def compute_revenue_metrics(facts: dict) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    # Try Revenues, fallback to SalesRevenueNet
    df_rev = extract_series(facts, "Revenues", ["USD"])  # quarterly preferred
    if df_rev.empty:
        df_rev = extract_series(facts, "SalesRevenueNet", ["USD"])  # fallback
    q = latest_quarters(df_rev, 4, forms=["10-Q", "10-K"]).copy()
    if q.empty:
        return None, None, None, None
    lq = float(q.tail(1)["val"].iloc[0])
    annualized = lq * 4.0
    ltm = float(q["val"].tail(4).sum()) if len(q) >= 4 else None
    yoy = None
    if len(q) >= 5 and q["val"].iloc[-5] != 0:
        yoy = ((q["val"].iloc[-1] - q["val"].iloc[-5]) / q["val"].iloc[-5]) * 100.0
    return lq, annualized, ltm, yoy


def compute_ebitda_metrics(facts: dict) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    # Prefer direct EBITDA if present, else OperatingIncomeLoss + Depreciation & Amortization
    df_ebitda = extract_series(facts, "EarningsBeforeInterestTaxesDepreciationAndAmortization", ["USD"])
    if df_ebitda.empty:
        df_op = extract_series(facts, "OperatingIncomeLoss", ["USD"]).rename(columns={"val": "op"})
        df_da = extract_series(facts, "DepreciationAndAmortization", ["USD"]).rename(columns={"val": "da"})
        m = pd.merge(df_op, df_da, on=["end", "form", "fy", "fp"], how="outer")
        if m.empty:
            df_ebitda = pd.DataFrame(columns=["end", "val"])
        else:
            m["val"] = m.get("op", 0).fillna(0) + m.get("da", 0).fillna(0)
            df_ebitda = m[["end", "val", "form", "fy", "fp"]]
    q = latest_quarters(df_ebitda, 4, forms=["10-Q", "10-K"]).copy()
    if q.empty:
        return None, None, None, None
    lq = float(q.tail(1)["val"].iloc[0])
    annualized = lq * 4.0
    ltm = float(q["val"].tail(4).sum()) if len(q) >= 4 else None
    yoy = None
    if len(q) >= 5 and q["val"].iloc[-5] != 0:
        yoy = ((q["val"].iloc[-1] - q["val"].iloc[-5]) / q["val"].iloc[-5]) * 100.0
    return lq, annualized, ltm, yoy


def compute_net_income_metrics(facts: dict) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    df = extract_series(facts, "NetIncomeLoss", ["USD"])
    q = latest_quarters(df, 4, forms=["10-Q", "10-K"]).copy()
    if q.empty:
        return None, None, None, None
    lq = float(q.tail(1)["val"].iloc[0])
    annualized = lq * 4.0
    ltm = float(q["val"].tail(4).sum()) if len(q) >= 4 else None
    yoy = None
    if len(q) >= 5 and q["val"].iloc[-5] != 0:
        yoy = ((q["val"].iloc[-1] - q["val"].iloc[-5]) / q["val"].iloc[-5]) * 100.0
    return lq, annualized, ltm, yoy


def extract_balance_sheet(facts: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Cash & STI
    cash = extract_series(facts, "CashCashEquivalentsAndShortTermInvestments", ["USD"]).tail(1)
    if cash.empty:
        cash1 = extract_series(facts, "CashAndCashEquivalentsAtCarryingValue", ["USD"]).tail(1)
        cash2 = extract_series(facts, "ShortTermInvestments", ["USD"]).tail(1)
        cval = (float(cash1["val"].iloc[0]) if not cash1.empty else 0.0) + (float(cash2["val"].iloc[0]) if not cash2.empty else 0.0)
    else:
        cval = float(cash["val"].iloc[0])

    # Total debt (approx): LT debt + current debt/short-term borrowings
    lt = extract_series(facts, "LongTermDebtNoncurrent", ["USD"]).tail(1)
    cur = extract_series(facts, "LongTermDebtCurrent", ["USD"]).tail(1)
    stb = extract_series(facts, "ShortTermBorrowings", ["USD"]).tail(1)
    debt = (float(lt["val"].iloc[0]) if not lt.empty else 0.0) + (float(cur["val"].iloc[0]) if not cur.empty else 0.0) + (float(stb["val"].iloc[0]) if not stb.empty else 0.0)

    # Shares outstanding
    sh = extract_series(facts, "CommonStockSharesOutstanding", ["shares", "USD", "pure"]).tail(1)
    shares = float(sh["val"].iloc[0]) if not sh.empty else None
    return cval if cval is not None else None, debt if debt is not None else None, shares


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
                  cash_sti, total_debt, shares_outstanding, stock_price, market_cap
                ) VALUES (
                  :cik, :report_date, :fy, :fp,
                  :rev_lq, :rev_ann, :rev_ltm, :rev_yoy,
                  :ebitda_lq, :ebitda_ann, :ebitda_ltm, :ebitda_yoy, :ebitda_margin,
                  :ni_lq, :ni_ann, :ni_ltm, :ni_yoy, :ni_margin,
                  :cash, :debt, :shares, :price, :mcap
                )
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

        # Fundamentals
        rev_lq, rev_ann, rev_ltm, rev_yoy = compute_revenue_metrics(facts)
        ebitda_lq, ebitda_ann, ebitda_ltm, ebitda_yoy = compute_ebitda_metrics(facts)
        ni_lq, ni_ann, ni_ltm, ni_yoy = compute_net_income_metrics(facts)
        cash_sti, total_debt, shares = extract_balance_sheet(facts)
        price, mcap = maybe_price_and_mcap(subs, shares)
        if mcap is None and price is not None and shares is not None:
            mcap = price * shares

        # Margins
        ebitda_margin = (ebitda_lq / rev_lq * 100.0) if ebitda_lq is not None and rev_lq not in (None, 0) else None
        ni_margin = (ni_lq / rev_lq * 100.0) if ni_lq is not None and rev_lq not in (None, 0) else None

        # Timestamps (last quarterly row we used)
        stamps = {"report_date": None, "fy": None, "fp": None}
        try:
            df_rev = extract_series(facts, "Revenues", ["USD"]) or pd.DataFrame()
            q = latest_quarters(df_rev, 1, forms=["10-Q", "10-K"]).copy()
            if not q.empty:
                stamps = {"report_date": q["end"].iloc[0], "fy": q["fy"].iloc[0], "fp": q["fp"].iloc[0]}
        except Exception:
            pass

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
            "cash_sti": cash_sti, "total_debt": total_debt, "shares_outstanding": shares, "stock_price": price, "market_cap": mcap,
        }, stamps)
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

    tickers = parse_tickers_arg(args.tickers, args.tickers_file)
    if not tickers and args.tickers and args.tickers.strip().upper() == "ALL":
        # Pull all tickers from SEC mapping
        sess = requests.Session()
        h = HEADERS_TMPL.copy(); h["User-Agent"] = args.user_agent; sess.headers.update(h)
        mapping = load_ticker_map(sess)
        tickers = sorted(mapping.keys())
    if not tickers:
        raise SystemExit("Provide --tickers or --tickers-file")
    run_etl(tickers, args.db, args.user_agent)


if __name__ == "__main__":
    main()


