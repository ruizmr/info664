-- Companies static metadata
CREATE TABLE IF NOT EXISTS companies (
  cik TEXT PRIMARY KEY,
  ticker TEXT,
  name TEXT,
  business_state TEXT,
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

  cash_sti REAL,
  total_debt REAL,
  shares_outstanding REAL,
  stock_price REAL,
  market_cap REAL,

  FOREIGN KEY (cik) REFERENCES companies(cik)
);

CREATE INDEX IF NOT EXISTS idx_fund_cik_date ON fundamentals(cik, report_date);

-- Filings (unique by cik+type+url)
CREATE TABLE IF NOT EXISTS filings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cik TEXT NOT NULL,
  filing_type TEXT NOT NULL,
  url TEXT NOT NULL,
  UNIQUE(cik, filing_type, url),
  FOREIGN KEY (cik) REFERENCES companies(cik)
);


