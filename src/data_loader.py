# =============================================================================
# data_loader.py
# Pulls US Treasury yield data and NBER recession indicators from FRED,
# cleans them up, merges them, and saves to CSV.
#
# On the FIRST run: downloads everything from the internet and saves to CSV.
# On every run AFTER: just loads from the saved CSV (fast, no API calls).
# This is called "caching" — saving data locally so you don't re-download it.
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np

# fredapi is a Python wrapper around FRED's API.
# "Wrapper" means someone wrote code that makes talking to FRED's API easier.
from fredapi import Fred # type: ignore

# Add the src directory to Python's path so we can import config.
# sys.path is the list of folders Python searches when you write "import X".
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    FRED_API_KEY, START_DATE, END_DATE,
    YIELD_SERIES, RECESSION_SERIES,
    TREASURY_CSV, RECESSION_CSV, COMBINED_CSV,
    DATA_DIR
)


# =============================================================================
# HELPER: CHECK IF CACHED DATA EXISTS AND IS REASONABLY FRESH
# =============================================================================

def cache_exists(filepath):
    """
    Returns True if a CSV file exists at `filepath`, False otherwise.
    We use this to decide whether to download fresh data or load from disk.
    """
    return os.path.exists(filepath)


# =============================================================================
# STEP 1: PULL TREASURY YIELD DATA
# =============================================================================

def pull_treasury_yields(fred_client):
    """
    Downloads all five Treasury yield series from FRED.
    Returns a single pandas DataFrame with one column per yield series,
    indexed by date.

    What is a DataFrame?
    Think of it as a spreadsheet in Python. Rows = dates. Columns = yields.
    Index = the row labels (here, dates).
    """
    print("Downloading Treasury yield data from FRED...")

    # We'll collect each series in a dict, then combine them at the end.
    series_dict = {}

    for series_id, description in YIELD_SERIES.items():
        print(f"  Pulling {series_id} ({description})...")

        # fred_client.get_series() downloads a single data series.
        # It returns a pandas Series — like a single column of a spreadsheet.
        raw = fred_client.get_series(
            series_id,
            observation_start=START_DATE,
            observation_end=END_DATE
        )

        series_dict[series_id] = raw

    # pd.DataFrame(dict) combines multiple Series into one table.
    # Each key becomes a column name. The index (dates) is shared.
    df = pd.DataFrame(series_dict)

    print(f"  Raw data shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    return df


# =============================================================================
# STEP 2: PULL RECESSION INDICATOR
# =============================================================================

def pull_recession_indicator(fred_client):
    """
    Downloads the NBER recession indicator from FRED.
    Returns a pandas Series with monthly frequency:
      1 = the economy was in a recession that month
      0 = the economy was not in a recession that month

    NBER = National Bureau of Economic Research.
    They are the official body that declares when US recessions start and end.
    Important: they declare recessions AFTER they've already begun, often months later.
    That's a known limitation of this data — noted in the project spec.
    """
    print("Downloading NBER recession indicator from FRED...")

    recession = fred_client.get_series(
        RECESSION_SERIES,
        observation_start=START_DATE,
        observation_end=END_DATE
    )

    # Name the series so it's identifiable later
    recession.name = "RECESSION"

    print(f"  Recession data: {len(recession)} monthly observations")
    print(f"  Date range: {recession.index[0].date()} to {recession.index[-1].date()}")

    return recession


# =============================================================================
# STEP 3: CLEAN THE YIELD DATA
# =============================================================================

def clean_treasury_yields(df):
    """
    Cleans up the raw yield DataFrame:

    1. FORWARD-FILL gaps up to 3 days.
       Treasury yields aren't reported on weekends and holidays.
       Forward-fill means: if Monday is missing, use Friday's value.
       The limit=3 means: only fill up to 3 consecutive missing days.
       Beyond that, we'd be making up data, which isn't appropriate.

    2. DROP any remaining rows with missing values (NaN = Not a Number,
       Python's way of representing missing data).

    3. CONVERT from percentage points to decimals? No — we keep them as
       percentages (e.g., 4.5 means 4.5%) to match how traders talk about rates.
    """
    print("Cleaning Treasury yield data...")

    original_rows = len(df)

    # ffill = forward fill. limit=3 means fill at most 3 consecutive gaps.
    # This handles weekends (2 days) and occasional holiday gaps (up to 3 days).
    df = df.ffill(limit=3)

    # Drop any rows where ANY column still has a missing value.
    # axis=0 means "drop rows" (as opposed to axis=1 which drops columns).
    df = df.dropna(axis=0)

    rows_dropped = original_rows - len(df)
    print(f"  Rows after cleaning: {len(df)} (dropped {rows_dropped} rows with NaN)")

    return df


# =============================================================================
# STEP 4: MERGE RECESSION INDICATOR ONTO DAILY DATA
# =============================================================================

def merge_recession_indicator(daily_df, recession_series):
    """
    The recession indicator is MONTHLY but our yield data is DAILY.
    We need to align them.

    Strategy: forward-fill the monthly recession value to every day in that month.
    So if January 2001 = 1 (recession), then every trading day in January 2001 = 1.

    This is technically a slight simplification — the NBER dates recessions to
    specific days — but it's standard practice and fine for our purposes.

    pd.merge_asof: a special merge that matches each date in the daily data
    to the most recent date in the monthly data. Perfect for this use case.
    """
    print("Merging recession indicator onto daily yield data...")

    # Convert recession series to DataFrame for merging
    recession_df = recession_series.reset_index()
    recession_df.columns = ["DATE", "RECESSION"]

    # Reset the index of yields so DATE becomes a regular column
    daily_df = daily_df.reset_index()
    daily_df.columns = ["DATE"] + list(daily_df.columns[1:])

    # Ensure both DATE columns are the same type (datetime)
    daily_df["DATE"] = pd.to_datetime(daily_df["DATE"])
    recession_df["DATE"] = pd.to_datetime(recession_df["DATE"])

    # Sort both by date (required for merge_asof)
    daily_df = daily_df.sort_values("DATE")
    recession_df = recession_df.sort_values("DATE")

    # merge_asof: for each row in daily_df, find the most recent row in
    # recession_df and attach its RECESSION value. Like a VLOOKUP in Excel.
    merged = pd.merge_asof(daily_df, recession_df, on="DATE")

    # Make DATE the index again (standard for time series DataFrames)
    merged = merged.set_index("DATE")

    # Make the RECESSION column an integer (0 or 1), not a float
    merged["RECESSION"] = merged["RECESSION"].astype(int)

    print(f"  Merged data shape: {merged.shape}")
    print(f"  Recession days: {merged['RECESSION'].sum()} of {len(merged)} total days")

    return merged


# =============================================================================
# STEP 5: CALCULATE SPREADS AND INVERSION FLAGS
# =============================================================================

def add_spreads(df):
    """
    Adds the key yield spread columns and inversion boolean flags.

    SPREAD = difference between two yields.
    The most watched spreads:
      2s10s: 10-year yield minus 2-year yield
      3m10s: 10-year yield minus 3-month yield

    When either spread goes NEGATIVE, the curve is INVERTED.
    Inversion = short-term rates are HIGHER than long-term rates.
    This is abnormal and historically precedes recessions.
    """
    print("Calculating yield spreads and inversion flags...")

    # The two core spreads — these are what traders watch daily
    df["SPREAD_2S10S"] = df["DGS10"] - df["DGS2"]
    df["SPREAD_3M10S"] = df["DGS10"] - df["DGS3MO"]

    # Boolean (True/False) inversion flags
    # < 0 means the spread went negative = curve is inverted
    df["INVERTED_2S10S"] = df["SPREAD_2S10S"] < 0
    df["INVERTED_3M10S"] = df["SPREAD_3M10S"] < 0

    # Rolling 3-month change in 2s10s spread (used later in the regression model)
    # 63 trading days ≈ 3 calendar months
    # This captures MOMENTUM — is the curve flattening faster recently?
    df["SPREAD_2S10S_3M_CHANGE"] = df["SPREAD_2S10S"].diff(63)

    inversion_2s10s_pct = df["INVERTED_2S10S"].mean() * 100
    inversion_3m10s_pct = df["INVERTED_3M10S"].mean() * 100
    print(f"  2s10s inverted {inversion_2s10s_pct:.1f}% of trading days")
    print(f"  3m10s inverted {inversion_3m10s_pct:.1f}% of trading days")

    return df


# =============================================================================
# MAIN FUNCTION: ORCHESTRATES EVERYTHING
# =============================================================================

def load_data(force_refresh=False):
    """
    Main entry point. Call this from your notebook or scripts.

    If the combined CSV already exists and force_refresh=False:
        → Just load it from disk. Fast.

    If the file doesn't exist OR force_refresh=True:
        → Pull from FRED, clean, merge, calculate spreads, save to CSV.

    Parameters
    ----------
    force_refresh : bool
        Set to True if you want to re-download fresh data even if a cache exists.
        Useful when you want the most recent data after several months.

    Returns
    -------
    pd.DataFrame
        The full cleaned dataset, ready for analysis.
    """
    # Make sure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- FAST PATH: load from cache ---
    if cache_exists(COMBINED_CSV) and not force_refresh:
        print(f"Loading cached data from {COMBINED_CSV}")
        df = pd.read_csv(COMBINED_CSV, index_col="DATE", parse_dates=True)
        print(f"  Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
        return df

    # --- SLOW PATH: download from FRED ---
    print("=" * 60)
    print("No cached data found. Downloading from FRED...")
    print("This takes about 30-60 seconds. Future runs will be instant.")
    print("=" * 60)

    # Check API key
    if FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
        raise ValueError(
            "\n\nYou haven't set your FRED API key yet!\n"
            "1. Go to https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "2. Register for a free account (takes 30 seconds)\n"
            "3. Copy your API key\n"
            "4. Open src/config.py and paste it into FRED_API_KEY = '...'\n"
        )

    # Initialize the FRED client with your API key
    fred = Fred(api_key=FRED_API_KEY)

    # Run the pipeline step by step
    raw_yields    = pull_treasury_yields(fred)
    recession_raw = pull_recession_indicator(fred)
    clean_yields  = clean_treasury_yields(raw_yields)
    merged        = merge_recession_indicator(clean_yields, recession_raw)
    final         = add_spreads(merged)

    # Save intermediate files too (useful for debugging)
    clean_yields.to_csv(TREASURY_CSV)
    recession_raw.to_frame("RECESSION").to_csv(RECESSION_CSV)

    # Save the full combined dataset
    final.to_csv(COMBINED_CSV)
    print(f"\nData saved to {COMBINED_CSV}")
    print(f"Final dataset: {len(final)} rows, {len(final.columns)} columns")
    print(f"Columns: {list(final.columns)}")

    return final


# =============================================================================
# QUICK SANITY CHECK FUNCTION
# =============================================================================

def print_summary(df):
    """
    Prints a quick summary of the dataset so you can verify it looks right.
    Call this after load_data() to make sure nothing is obviously broken.
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Date range:    {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Total rows:    {len(df):,} trading days")
    print(f"Total columns: {len(df.columns)}")
    print()
    print("Latest yield curve (most recent day in dataset):")
    latest = df.iloc[-1]
    for series_id in ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"]:
        if series_id in df.columns:
            print(f"  {series_id:8s}: {latest[series_id]:.2f}%")
    print()
    print(f"2s10s spread today: {latest['SPREAD_2S10S']:.2f}%")
    print(f"3m10s spread today: {latest['SPREAD_3M10S']:.2f}%")
    inverted = latest["SPREAD_2S10S"] < 0
    print(f"Curve currently:    {'INVERTED ⚠' if inverted else 'Normal'}")
    print()

    # Count recessions in dataset
    # We find "recession start" dates by looking for days where RECESSION
    # goes from 0 to 1 (i.e., the first day of each recession period)
    recession_starts = df[
        (df["RECESSION"] == 1) & (df["RECESSION"].shift(1) == 0)
    ]
    print(f"Recession periods in dataset: {len(recession_starts)}")
    for date in recession_starts.index:
        print(f"  Recession started: {date.date()}")
    print("=" * 60)


# =============================================================================
# ALLOW RUNNING THIS FILE DIRECTLY FOR TESTING
# python src/data_loader.py
# =============================================================================

if __name__ == "__main__":
    df = load_data()
    print_summary(df)
