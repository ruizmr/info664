-- Companies static metadata
CREATE TABLE IF NOT EXISTS companies (
  cik TEXT PRIMARY KEY,
  ticker TEXT,
  name TEXT,
  business_state TEXT,
  state_incorp TEXT,
  sic TEXT,
  sic_description TEXT,
  fiscal_year_end TEXT,
  last_price REAL,
  market_cap REAL
);

CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);

-- Fundamentals: one row per cik and report date
CREATE TABLE IF NOT EXISTS fundamentals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cik TEXT NOT NULL,
  report_date TEXT,
  fiscal_year INTEGER,
  fiscal_period TEXT,

  revenue_lq REAL,
  revenue_annualized REAL,
  revenue_ltm REAL,
  revenue_yoy REAL,

  ebitda_lq REAL,
  ebitda_annualized REAL,
  ebitda_ltm REAL,
  ebitda_yoy REAL,
  ebitda_margin REAL,

  net_income_lq REAL,
  net_income_annualized REAL,
  net_income_ltm REAL,
  net_income_yoy REAL,
  net_income_margin REAL,

  gross_profit_lq REAL,
  gross_margin REAL,
  operating_income_lq REAL,
  operating_margin REAL,

  cfo_lq REAL,
  capex_lq REAL,
  fcf_lq REAL,

  cash_sti REAL,
  total_debt REAL,
  shares_outstanding REAL,
  stock_price REAL,
  market_cap REAL,

  deferred_revenue REAL,

  FOREIGN KEY (cik) REFERENCES companies(cik)
);

CREATE INDEX IF NOT EXISTS idx_fund_cik_date ON fundamentals(cik, report_date);
-- Enforce one row per (cik, report_date)
CREATE UNIQUE INDEX IF NOT EXISTS ux_fund_cik_report ON fundamentals(cik, report_date);

-- Filings (unique by cik+type+url)
CREATE TABLE IF NOT EXISTS filings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cik TEXT NOT NULL,
  filing_type TEXT NOT NULL,
  url TEXT NOT NULL,
  UNIQUE(cik, filing_type, url),
  FOREIGN KEY (cik) REFERENCES companies(cik)
);

-- Optional raw facts store (subset or all concepts)
CREATE TABLE IF NOT EXISTS facts_raw (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cik TEXT NOT NULL,
  concept TEXT NOT NULL,
  unit TEXT,
  end TEXT,
  fy INTEGER,
  fp TEXT,
  form TEXT,
  val REAL,
  dims TEXT,
  UNIQUE(cik, concept, unit, end, form, dims)
);

-- Audit records for metric extraction traceability
CREATE TABLE IF NOT EXISTS fundamentals_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cik TEXT NOT NULL,
  report_date TEXT,
  fiscal_year INTEGER,
  fiscal_period TEXT,
  metric TEXT NOT NULL,
  value REAL,
  source_tag TEXT,
  unit TEXT,
  scale_applied INTEGER,
  form TEXT,
  fy INTEGER,
  fp TEXT,
  FOREIGN KEY (cik) REFERENCES companies(cik)
);

CREATE INDEX IF NOT EXISTS idx_fundaudit_cik_date_metric ON fundamentals_audit(cik, report_date, metric);


