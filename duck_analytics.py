#!/usr/bin/env python3
"""DuckDB analytics over the existing SQLite database (zero-copy).

Examples:
  python duck_analytics.py --db saas_fundamentals.db --summary
  python duck_analytics.py --db saas_fundamentals.db --export fundamentals.parquet
"""

import argparse
import duckdb


def attach_sqlite(con: duckdb.DuckDBPyConnection, sqlite_path: str) -> None:
    con.sql("INSTALL sqlite_scanner;")
    con.sql("LOAD sqlite_scanner;")
    con.sql(f"ATTACH '{sqlite_path}' AS sqlite_db (TYPE SQLITE);")


def run_summary(con: duckdb.DuckDBPyConnection) -> None:
    print("== Tables present ==")
    print(con.sql("SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema='sqlite_db' ORDER BY 2").to_df().to_string(index=False))

    print("\n== Companies count ==")
    print(con.sql("SELECT COUNT(*) AS companies FROM sqlite_db.companies").to_df().to_string(index=False))

    print("\n== Fundamentals coverage ==")
    q_cov = """
    SELECT COUNT(*) AS n,
           ROUND(100.0*SUM(revenue_lq IS NOT NULL)/COUNT(*),1)  AS pct_rev_lq,
           ROUND(100.0*SUM(ebitda_lq IS NOT NULL)/COUNT(*),1)   AS pct_ebitda_lq,
           ROUND(100.0*SUM(net_income_lq IS NOT NULL)/COUNT(*),1) AS pct_ni_lq,
           ROUND(100.0*SUM(cash_sti IS NOT NULL)/COUNT(*),1)     AS pct_cash_sti,
           ROUND(100.0*SUM(total_debt IS NOT NULL)/COUNT(*),1)   AS pct_debt,
           ROUND(100.0*SUM(shares_outstanding IS NOT NULL)/COUNT(*),1) AS pct_shares
    FROM sqlite_db.fundamentals;
    """
    print(con.sql(q_cov).to_df().to_string(index=False))

    print("\n== Top 5 latest periods by report_date (if available) ==")
    q_dates = """
    SELECT report_date, COUNT(*) AS rows
    FROM sqlite_db.fundamentals
    WHERE report_date IS NOT NULL
    GROUP BY 1 ORDER BY report_date DESC LIMIT 5;
    """
    print(con.sql(q_dates).to_df().to_string(index=False))

    print("\n== Sample rows with non-null revenue & net income ==")
    q_sample = """
    SELECT c.ticker, f.report_date, f.revenue_lq, f.net_income_lq
    FROM sqlite_db.fundamentals f
    JOIN sqlite_db.companies c USING(cik)
    WHERE f.revenue_lq IS NOT NULL AND f.net_income_lq IS NOT NULL
    LIMIT 10;
    """
    print(con.sql(q_sample).to_df().to_string(index=False))

    print("\n== Nulls per key column (ratio) ==")
    q_nulls = """
    SELECT
      ROUND(100.0*SUM(report_date IS NULL)/COUNT(*),1) AS pct_null_report_date,
      ROUND(100.0*SUM(revenue_lq IS NULL)/COUNT(*),1) AS pct_null_revenue_lq,
      ROUND(100.0*SUM(ebitda_lq IS NULL)/COUNT(*),1) AS pct_null_ebitda_lq,
      ROUND(100.0*SUM(net_income_lq IS NULL)/COUNT(*),1) AS pct_null_net_income_lq,
      ROUND(100.0*SUM(cash_sti IS NULL)/COUNT(*),1) AS pct_null_cash_sti,
      ROUND(100.0*SUM(total_debt IS NULL)/COUNT(*),1) AS pct_null_total_debt,
      ROUND(100.0*SUM(shares_outstanding IS NULL)/COUNT(*),1) AS pct_null_shares
    FROM sqlite_db.fundamentals;
    """
    print(con.sql(q_nulls).to_df().to_string(index=False))

    print("\n== Margin distributions (histogram buckets) ==")
    q_hist = """
    WITH base AS (
      SELECT ebitda_margin AS m FROM sqlite_db.fundamentals WHERE ebitda_margin IS NOT NULL
    )
    SELECT
      CASE
        WHEN m < -100 THEN '<-100'
        WHEN m < -50 THEN '[-100,-50)'
        WHEN m < 0 THEN '[-50,0)'
        WHEN m < 20 THEN '[0,20)'
        WHEN m < 40 THEN '[20,40)'
        WHEN m < 60 THEN '[40,60)'
        WHEN m < 80 THEN '[60,80)'
        WHEN m <= 100 THEN '[80,100]'
        ELSE '>100'
      END AS bucket,
      COUNT(*) AS n
    FROM base
    GROUP BY 1 ORDER BY 1;
    """
    print(con.sql(q_hist).to_df().to_string(index=False))

    print("\n== Outlier margins (|margin|>100%) with links ==")
    q_outliers = """
    SELECT c.ticker, f.report_date, f.ebitda_margin, f.net_income_margin,
           (SELECT url FROM sqlite_db.filings WHERE cik=f.cik AND filing_type IN ('10-Q','10-K') ORDER BY id DESC LIMIT 1) AS filing_url
    FROM sqlite_db.fundamentals f
    JOIN sqlite_db.companies c USING(cik)
    WHERE (ABS(f.ebitda_margin) > 100 OR ABS(f.net_income_margin) > 100)
    ORDER BY ABS(COALESCE(f.ebitda_margin,0)) + ABS(COALESCE(f.net_income_margin,0)) DESC
    LIMIT 20;
    """
    print(con.sql(q_outliers).to_df().to_string(index=False))


def export_parquet(con: duckdb.DuckDBPyConnection, out_path: str) -> None:
    con.sql(f"COPY (SELECT * FROM sqlite_db.fundamentals) TO '{out_path}' (FORMAT PARQUET);")
    print(f"Exported fundamentals to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DuckDB analytics over SQLite (zero-copy)")
    ap.add_argument("--db", required=True, help="Path to SQLite db (e.g., saas_fundamentals.db)")
    ap.add_argument("--summary", action="store_true", help="Print coverage summary")
    ap.add_argument("--export", help="Export fundamentals to Parquet path")
    args = ap.parse_args()

    con = duckdb.connect()
    attach_sqlite(con, args.db)

    if args.summary:
        run_summary(con)
    if args.export:
        export_parquet(con, args.export)


if __name__ == "__main__":
    main()


