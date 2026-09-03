# =============================================================================
# config.py
# Central configuration for the yield curve project.
# All settings live here so you only ever change one file.
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()  # reads a local .env file, if present

# -----------------------------------------------------------------------------
# FRED API KEY
# Get yours free at: https://fred.stlouisfed.org/docs/api/api_key.html
# Set it in a local .env file (see .env.example) — never hardcode it here.
# -----------------------------------------------------------------------------
FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError(
        "\n\nFRED_API_KEY not found!\n"
        "1. Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html\n"
        "2. Create a file named .env in the project root\n"
        "3. Add this line to it: FRED_API_KEY=your_key_here\n"
    )

# -----------------------------------------------------------------------------
# DATE RANGE
# We pull data from 1990 to today.
# START_DATE is the earliest date we care about.
# END_DATE = None means "pull up to today automatically."
# -----------------------------------------------------------------------------
START_DATE = "1990-01-01"
END_DATE = None  # None = today

# -----------------------------------------------------------------------------
# FRED SERIES IDs
# These are the official codes FRED uses to identify each data series.
# You can look any of these up at fred.stlouisfed.org to verify what they are.
# -----------------------------------------------------------------------------
YIELD_SERIES = {
    "DGS3MO": "3-Month Treasury Yield",
    "DGS2":   "2-Year Treasury Yield",
    "DGS5":   "5-Year Treasury Yield",
    "DGS10":  "10-Year Treasury Yield",
    "DGS30":  "30-Year Treasury Yield",
}

RECESSION_SERIES = "USREC"  # NBER recession indicator: 1 = recession, 0 = expansion

# -----------------------------------------------------------------------------
# ASSET TICKERS (for Part 4 — cross-asset regime analysis)
# These are ETF ticker symbols, the same codes you'd type into Google Finance.
# -----------------------------------------------------------------------------
ASSET_TICKERS = {
    "SPY": "S&P 500 (US Equities)",
    "TLT": "20+ Year Treasury Bonds (Long Duration)",
    "HYG": "High Yield Corporate Bonds (Junk Bonds)",
    "LQD": "Investment Grade Corporate Bonds",
}

ASSET_START_DATE = "2003-01-01"  # Earliest date all four ETFs existed

# -----------------------------------------------------------------------------
# FILE PATHS
# Using os.path so this works on Mac, Windows, and Linux without changes.
# __file__ refers to this config.py file itself.
# os.path.dirname gets the folder it lives in.
# os.path.join builds paths correctly for whatever OS you're on.
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")

# CSV file paths — data gets saved here on first run, loaded from here after
TREASURY_CSV   = os.path.join(DATA_DIR, "treasury_yields.csv")
RECESSION_CSV  = os.path.join(DATA_DIR, "nber_recessions.csv")
ASSET_CSV      = os.path.join(DATA_DIR, "asset_returns.csv")
COMBINED_CSV   = os.path.join(DATA_DIR, "combined_data.csv")

# -----------------------------------------------------------------------------
# NELSON-SIEGEL PARAMETERS
# These are the maturities (in years) corresponding to our five yield series.
# 3 months = 0.25 years.
# -----------------------------------------------------------------------------
NS_MATURITIES = [0.25, 2.0, 5.0, 10.0, 30.0]

# Initial guesses for the curve-fitting algorithm.
# beta0=5 (rates around 5% long-run), beta1=-2 (slight upward slope),
# beta2=1 (mild hump), lambda=1.5 (typical decay speed).
# These don't have to be perfect — scipy will find the best values from here.
NS_INITIAL_PARAMS = [5.0, -2.0, 1.0, 1.5]

# -----------------------------------------------------------------------------
# RECESSION MODEL PARAMETERS
# -----------------------------------------------------------------------------
# How many trading days to look forward when creating the recession label.
# 252 trading days ≈ 1 calendar year (markets are open ~252 days/year).
RECESSION_HORIZON_DAYS = 252

# Train/test split date. Everything before this = training. After = testing.
TRAIN_TEST_SPLIT = "2010-01-01"

# -----------------------------------------------------------------------------
# REGIME THRESHOLDS (in percentage points, i.e. basis points / 100)
# Steep:    2s10s spread > 1.0  (more than 100 basis points)
# Flat:     2s10s spread between -0.25 and 1.0
# Inverted: 2s10s spread < -0.25 (more than 25 basis points inverted)
# -----------------------------------------------------------------------------
REGIME_STEEP_THRESHOLD    =  1.00   # above this = Steep
REGIME_INVERTED_THRESHOLD = -0.25   # below this = Inverted
                                    # between = Flat
