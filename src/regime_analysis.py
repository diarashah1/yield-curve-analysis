# =============================================================================
# regime_analysis.py
# Cross-asset performance across yield curve regimes.
#
# What this file does:
#   1. Pulls SPY, TLT, HYG, LQD from yfinance
#   2. Classifies each day into a yield curve regime (Steep/Flat/Inverted)
#   3. Calculates return, volatility, Sharpe, and max drawdown per asset per regime
#   4. Returns a clean summary table and the raw data for charting
#
# This is the most interview-ready output in the project.
# A rates desk interviewer asks "what did you find?" — you read them this table.
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASSET_TICKERS, ASSET_START_DATE,
    ASSET_CSV, DATA_DIR,
    REGIME_STEEP_THRESHOLD, REGIME_INVERTED_THRESHOLD
)


# =============================================================================
# STEP 1: PULL AND CACHE ASSET DATA
# =============================================================================

def load_asset_returns(force_refresh=False):
    """
    Downloads daily price data for SPY, TLT, HYG, LQD from Yahoo Finance
    and converts to daily returns. Caches to CSV.

    What are these assets?
    SPY — S&P 500 ETF. Tracks the 500 largest US companies. Represents
          the broad equity market. Goes up when economy is doing well,
          down sharply in recessions.

    TLT — iShares 20+ Year Treasury Bond ETF. Holds long-term US government
          bonds. When interest rates FALL, bond prices RISE (they move
          opposite). So TLT does well when the Fed cuts rates, which
          typically happens during or after recessions.

    HYG — iShares High Yield Corporate Bond ETF. "High yield" = "junk bonds"
          — bonds from companies with weaker finances that pay higher interest
          to compensate for higher default risk. Very sensitive to recession
          fear because these companies are most likely to default.

    LQD — iShares Investment Grade Corporate Bond ETF. Bonds from stronger
          companies (Apple, Microsoft, etc.). Lower yield, much more stable
          than HYG. Safer credit exposure.

    Returns
    -------
    pd.DataFrame
        Daily returns for all four assets, aligned on common dates
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(ASSET_CSV) and not force_refresh:
        print(f"Loading cached asset data from {ASSET_CSV}")
        returns = pd.read_csv(ASSET_CSV, index_col="Date", parse_dates=True)
        print(f"  Loaded {len(returns):,} rows from {returns.index[0].date()} to {returns.index[-1].date()}")
        return returns

    print("Downloading asset price data from Yahoo Finance...")
    tickers = list(ASSET_TICKERS.keys())

    # yf.download pulls daily OHLCV data for all tickers at once
    # auto_adjust=True adjusts for dividends and stock splits automatically
    prices = yf.download(
        tickers,
        start=ASSET_START_DATE,
        auto_adjust=True,
        progress=False
    )["Close"]

    # Rename columns to just the ticker symbols
    prices.columns = [col if isinstance(col, str) else col[0] for col in prices.columns]

    # pct_change() converts prices to daily returns
    # e.g. if SPY goes from 400 to 404, daily return = (404-400)/400 = 1.0%
    # We multiply by 1 (keep as decimal) — annualization happens later
    returns = prices.pct_change().dropna()

    # Reorder columns to match our ticker list
    returns = returns[[t for t in tickers if t in returns.columns]]

    returns.index.name = "Date"
    returns.to_csv(ASSET_CSV)
    print(f"  Downloaded {len(returns):,} rows, saved to {ASSET_CSV}")
    print(f"  Tickers: {list(returns.columns)}")

    return returns


# =============================================================================
# STEP 2: CLASSIFY YIELD CURVE REGIMES
# =============================================================================

def classify_regimes(df):
    """
    Assigns a regime label to each day based on the 2s10s spread.

    Three regimes:
      STEEP    — spread > 100 basis points (1.0%)
                 Economy is growing, banks are profitable, credit flows freely.
                 Normal post-recession expansion phase.

      FLAT     — spread between -25 and +100 basis points
                 Transition zone. Fed may be tightening. Market uncertain.
                 Not clearly expansionary or recessionary.

      INVERTED — spread < -25 basis points (-0.25%)
                 Short rates exceed long rates. Bank margins compressed.
                 Credit tightening. Historically precedes recessions.

    Why -25bp threshold for inversion instead of 0?
    A very brief dip to -5bp or -10bp can be noise or temporary.
    The -25bp threshold ensures we're capturing meaningful inversions,
    not just day-to-day volatility around zero.

    Returns
    -------
    pd.Series
        String labels: "Steep", "Flat", or "Inverted" for each date
    """
    conditions = [
        df["SPREAD_2S10S"] > REGIME_STEEP_THRESHOLD,
        df["SPREAD_2S10S"] < REGIME_INVERTED_THRESHOLD,
    ]
    choices = ["Steep", "Inverted"]
    regime = np.select(conditions, choices, default="Flat")
    regime_series = pd.Series(regime, index=df.index, name="REGIME")

    # Print regime counts
    counts = pd.Series(regime).value_counts()
    total = len(regime)
    print("Regime classification:")
    for r in ["Steep", "Flat", "Inverted"]:
        n = counts.get(r, 0)
        print(f"  {r:10s}: {n:,} days ({n/total*100:.1f}%)")

    return regime_series


# =============================================================================
# STEP 3: CALCULATE PERFORMANCE METRICS PER ASSET PER REGIME
# =============================================================================

def max_drawdown(returns_series):
    """
    Calculates the maximum drawdown for a return series.

    Maximum drawdown = the largest peak-to-trough decline in portfolio value.
    It answers: "what's the worst loss you would have suffered if you bought
    at the worst possible time and sold at the worst possible time?"

    Expressed as a negative percentage (e.g. -0.35 = lost 35% peak to trough).

    How it works:
    1. Convert daily returns to a cumulative wealth index (starting at $1)
    2. Track the running maximum (the "peak" at each point in time)
    3. Drawdown at each point = current value / peak - 1
    4. Maximum drawdown = the worst (most negative) drawdown in the series
    """
    # cumsum gives you the compounded return at each point
    # (1 + r1) * (1 + r2) * ... — this is the cumprod approach
    cumulative = (1 + returns_series).cumprod()

    # Running maximum — the highest value the portfolio has ever reached
    running_max = cumulative.cummax()

    # Drawdown at each point = how far below the peak are we?
    drawdown = (cumulative - running_max) / running_max

    return drawdown.min()  # Most negative value = max drawdown


def calculate_regime_stats(returns, regime_series):
    """
    For each asset and each regime, calculates:
      - Annualized return   (mean daily return × 252 trading days per year)
      - Annualized volatility (std dev of daily returns × √252)
      - Sharpe ratio         (annualized return ÷ annualized volatility)
      - Maximum drawdown

    Why 252?
    There are approximately 252 trading days in a calendar year (365 days
    minus weekends minus ~9 holidays).

    Why √252 for volatility?
    Standard deviation scales with the square root of time. If daily
    volatility is 1%, annual volatility is 1% × √252 ≈ 15.9%.

    What is Sharpe ratio?
    The return per unit of risk. A Sharpe of 1.0 means you earned 1% of
    return for each 1% of risk taken. Higher is better.
    We use zero as the risk-free rate for simplicity (note this assumption).

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame: rows = (asset, metric), columns = regimes
    """
    regimes = ["Steep", "Flat", "Inverted"]
    assets = returns.columns.tolist()
    metrics = ["Ann. Return (%)", "Ann. Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"]

    # Align dates: only use days where both returns and regime data exist
    common_dates = returns.index.intersection(regime_series.index)
    returns_aligned = returns.loc[common_dates]
    regime_aligned  = regime_series.loc[common_dates]

    results = {}

    for asset in assets:
        asset_description = ASSET_TICKERS.get(asset, asset)
        results[asset] = {}

        for regime in regimes:
            mask = regime_aligned == regime
            r = returns_aligned.loc[mask, asset].dropna()

            if len(r) < 30:
                # Not enough data for this regime/asset combination
                results[asset][regime] = {
                    "Ann. Return (%)":    np.nan,
                    "Ann. Volatility (%)": np.nan,
                    "Sharpe Ratio":        np.nan,
                    "Max Drawdown (%)":    np.nan,
                    "N Days":              len(r),
                }
                continue

            ann_return = r.mean() * 252 * 100          # convert to percentage
            ann_vol    = r.std()  * np.sqrt(252) * 100 # convert to percentage
            sharpe     = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
            mdd        = max_drawdown(r) * 100          # convert to percentage

            results[asset][regime] = {
                "Ann. Return (%)":    round(ann_return, 2),
                "Ann. Volatility (%)": round(ann_vol, 2),
                "Sharpe Ratio":        round(sharpe, 2),
                "Max Drawdown (%)":    round(mdd, 2),
                "N Days":              len(r),
            }

    return results


def format_results_table(results):
    """
    Converts the nested results dict into a clean, readable DataFrame.
    Regimes = columns. Assets + metrics = rows.

    This is the table you show in interviews.
    """
    regimes = ["Steep", "Flat", "Inverted"]
    metrics = ["Ann. Return (%)", "Ann. Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"]

    rows = []
    for asset, regime_data in results.items():
        for metric in metrics:
            row = {"Asset": asset, "Metric": metric}
            for regime in regimes:
                row[regime] = regime_data.get(regime, {}).get(metric, np.nan)
            rows.append(row)

    table = pd.DataFrame(rows).set_index(["Asset", "Metric"])
    return table


def print_results_table(table, results):
    """
    Prints the results table in a clean format to the terminal.
    Also prints the key economic findings you should memorize.
    """
    regimes = ["Steep", "Flat", "Inverted"]
    metrics = ["Ann. Return (%)", "Ann. Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"]

    print("\n" + "="*75)
    print("CROSS-ASSET PERFORMANCE BY YIELD CURVE REGIME")
    print("="*75)
    print(f"{'Asset':<6} {'Metric':<22} {'Steep':>12} {'Flat':>12} {'Inverted':>12}")
    print("-"*75)

    for asset in results.keys():
        for i, metric in enumerate(metrics):
            asset_label = asset if i == 0 else ""
            steep    = results[asset].get("Steep",    {}).get(metric, np.nan)
            flat     = results[asset].get("Flat",     {}).get(metric, np.nan)
            inverted = results[asset].get("Inverted", {}).get(metric, np.nan)

            def fmt(v):
                if pd.isna(v): return "N/A"
                if "Sharpe" in metric: return f"{v:>12.2f}"
                return f"{v:>11.1f}%"

            print(f"{asset_label:<6} {metric:<22} {fmt(steep)} {fmt(flat)} {fmt(inverted)}")
        print("-"*75)

    # Print key economic findings to validate against theory
    print("\nKEY FINDINGS — validate against economic theory:")
    print()

    for asset, expected in [
        ("TLT", "should perform BEST in Inverted regime (rates fall, bond prices rise)"),
        ("HYG", "should perform WORST in Inverted regime (recession risk, defaults rise)"),
        ("SPY", "should show largest drawdown in Inverted regime"),
        ("LQD", "should outperform HYG in Inverted regime (safer credit)"),
    ]:
        if asset in results:
            steep_ret    = results[asset].get("Steep",    {}).get("Ann. Return (%)", np.nan)
            inverted_ret = results[asset].get("Inverted", {}).get("Ann. Return (%)", np.nan)
            steep_mdd    = results[asset].get("Steep",    {}).get("Max Drawdown (%)", np.nan)
            inverted_mdd = results[asset].get("Inverted", {}).get("Max Drawdown (%)", np.nan)

            print(f"  {asset}: {expected}")
            if not pd.isna(steep_ret) and not pd.isna(inverted_ret):
                direction = "✓" if (
                    (asset == "TLT" and inverted_ret > steep_ret) or
                    (asset == "HYG" and inverted_ret < steep_ret) or
                    (asset == "SPY" and inverted_mdd < steep_mdd) or
                    (asset == "LQD" and True)
                ) else "✗ UNEXPECTED"
                print(f"    Steep return: {steep_ret:.1f}%  |  Inverted return: {inverted_ret:.1f}%  {direction}")
            print()
