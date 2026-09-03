# =============================================================================
# 04_cross_asset_regimes.py
# Cross-asset regime analysis — full analysis script.
#
# Run this AFTER 03_recession_model.py.
# Produces:
#   1. Summary table: annualized return, vol, Sharpe, max drawdown
#      by asset (SPY/TLT/HYG/LQD) and regime (Steep/Flat/Inverted)
#   2. Bar chart: annualized returns by asset and regime
#   3. Drawdown chart: max drawdown by asset and regime
#   4. Regime timeline: what regime were we in each year?
#
# Run from terminal: python notebooks/04_cross_asset_regimes.py
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_data
from regime_analysis import (
    load_asset_returns,
    classify_regimes,
    calculate_regime_stats,
    format_results_table,
    print_results_table,
)
from config import FIGURES_DIR, TABLES_DIR

sns.set_theme(style="whitegrid", font_scale=1.1)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Color scheme for three regimes — used consistently across all charts
REGIME_COLORS = {
    "Steep":    "#2166ac",   # blue  — expansionary
    "Flat":     "#f4a582",   # orange — transitional
    "Inverted": "#d6604d",   # red   — warning
}


# =============================================================================
# CELL 1: Load data
# =============================================================================

print("Loading yield curve data...")
df = load_data()
print(f"Loaded {len(df):,} rows")

print("\nLoading asset return data...")
returns = load_asset_returns()
print(f"Asset data: {len(returns):,} rows, tickers: {list(returns.columns)}")


# =============================================================================
# CELL 2: Classify regimes
# =============================================================================

print("\n" + "="*65)
print("Classifying yield curve regimes (2s10s spread thresholds)")
print("="*65)
regime_series = classify_regimes(df)

# Quick check: print current regime
current_spread = df["SPREAD_2S10S"].iloc[-1]
current_regime = regime_series.iloc[-1]
print(f"\nCurrent 2s10s spread: {current_spread:+.2f}%")
print(f"Current regime: {current_regime}")


# =============================================================================
# CELL 3: Calculate performance statistics
# =============================================================================

print("\n" + "="*65)
print("Calculating performance statistics by regime...")
print("="*65)

results = calculate_regime_stats(returns, regime_series)
table   = format_results_table(results)
print_results_table(table, results)

# Save table to CSV
table_path = os.path.join(TABLES_DIR, "04_regime_stats.csv")
table.to_csv(table_path)
print(f"\nTable saved to: {table_path}")


# =============================================================================
# CELL 4: CHART 1 — Annualized returns by asset and regime (bar chart)
# =============================================================================
#
# This is your second most important visualization.
# Three grouped bars per asset. Color = regime.
#
# What to look for:
#   TLT: tall bar in "Inverted" (rates fall, bond prices rise)
#   HYG: short or negative bar in "Inverted" (credit stress)
#   SPY: decent bar in "Steep" (expansion), weaker in "Inverted"

def plot_returns_bar_chart(results):
    assets  = list(results.keys())
    regimes = ["Steep", "Flat", "Inverted"]

    # Build arrays for plotting
    x = np.arange(len(assets))
    width = 0.25   # width of each bar
    offsets = [-width, 0, width]   # position offset for each regime group

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (regime, offset) in enumerate(zip(regimes, offsets)):
        values = [
            results[a].get(regime, {}).get("Ann. Return (%)", np.nan)
            for a in assets
        ]
        bars = ax.bar(
            x + offset, values,
            width=width * 0.9,
            label=regime,
            color=REGIME_COLORS[regime],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5
        )

        # Add value labels on each bar
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 1.5
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    ypos,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8.5, color="black"
                )

    # Zero line
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

    # Asset labels and descriptions
    asset_labels = [
        f"{a}\n({['US Equities','Long Bonds','High Yield','Invest. Grade'][i]})"
        for i, a in enumerate(assets)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(asset_labels, fontsize=10)
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title(
        "Annualized Asset Returns by Yield Curve Regime (2003–Present)\n"
        "Steep = 2s10s > 100bps  |  Flat = −25 to +100bps  |  Inverted = 2s10s < −25bps",
        fontsize=11
    )
    ax.legend(title="Regime", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_returns_by_regime.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


plot_returns_bar_chart(results)


# =============================================================================
# CELL 5: CHART 2 — Sharpe ratio and Max drawdown comparison
# =============================================================================
#
# Two-panel chart:
#   Left:  Sharpe ratio by asset and regime (return per unit of risk)
#   Right: Max drawdown by asset and regime (worst loss experienced)
#
# Sharpe and drawdown together tell the full risk story.
# High return with terrible drawdown = not actually a good investment.

def plot_risk_metrics(results):
    assets  = list(results.keys())
    regimes = ["Steep", "Flat", "Inverted"]
    x = np.arange(len(assets))
    width = 0.25
    offsets = [-width, 0, width]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for regime, offset in zip(regimes, offsets):
        sharpes = [results[a].get(regime, {}).get("Sharpe Ratio", np.nan) for a in assets]
        mdds    = [results[a].get(regime, {}).get("Max Drawdown (%)", np.nan) for a in assets]

        ax1.bar(x + offset, sharpes, width * 0.9, label=regime,
                color=REGIME_COLORS[regime], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax2.bar(x + offset, mdds,    width * 0.9, label=regime,
                color=REGIME_COLORS[regime], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.4)

    for ax, title, ylabel in [
        (ax1, "Sharpe Ratio by Regime", "Sharpe Ratio (higher = better)"),
        (ax2, "Maximum Drawdown by Regime", "Max Drawdown (%) — more negative = worse"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(assets, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.legend(title="Regime", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Risk Metrics by Asset and Yield Curve Regime (2003–Present)",
        fontsize=12
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_risk_metrics_by_regime.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


plot_risk_metrics(results)


# =============================================================================
# CELL 6: CHART 3 — Regime timeline
# =============================================================================
#
# Shows which regime we were in at every point in history.
# Color-coded by regime. Overlaid with recession shading.
# This is a good context chart — it shows the full history at a glance.

def get_recession_periods(df):
    periods = []
    in_recession = False
    start_date = None
    for date, row in df.iterrows():
        if row["RECESSION"] == 1 and not in_recession:
            in_recession = True
            start_date = date
        elif row["RECESSION"] == 0 and in_recession:
            in_recession = False
            periods.append((start_date, date))
    if in_recession:
        periods.append((start_date, df.index[-1]))
    return periods


def plot_regime_timeline(df, regime_series):
    """
    Horizontal bar chart showing yield curve regime over time.
    Each regime gets its own color band.
    """
    # Align regime to df dates
    common = df.index.intersection(regime_series.index)
    df_plot = df.loc[common].copy()
    df_plot["REGIME"] = regime_series.loc[common]

    recession_periods = get_recession_periods(df_plot)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: 2s10s spread colored by regime
    for regime, color in REGIME_COLORS.items():
        mask = df_plot["REGIME"] == regime
        ax1.fill_between(df_plot.index, df_plot["SPREAD_2S10S"],
                         where=mask, color=color, alpha=0.4, label=regime)

    ax1.plot(df_plot.index, df_plot["SPREAD_2S10S"],
             color="black", linewidth=0.6, alpha=0.6)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.axhline(1.0,  color=REGIME_COLORS["Steep"],    linewidth=0.8,
                linestyle=":", alpha=0.7, label="Steep threshold (+100bps)")
    ax1.axhline(-0.25, color=REGIME_COLORS["Inverted"], linewidth=0.8,
                linestyle=":", alpha=0.7, label="Inverted threshold (−25bps)")

    for start, end in recession_periods:
        ax1.axvspan(start, end, color="gray", alpha=0.2, zorder=0)

    ax1.set_ylabel("2s10s Spread (%)")
    ax1.set_title(
        "Yield Curve Regime Classification Over Time\n"
        "Blue = Steep  |  Orange = Flat  |  Red = Inverted  |  Gray = NBER Recession",
        fontsize=11
    )
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: regime as a solid color bar (easier to see)
    regime_num = df_plot["REGIME"].map({"Steep": 1, "Flat": 0.5, "Inverted": 0})
    regime_color = df_plot["REGIME"].map(REGIME_COLORS)

    # Plot each regime period as a colored fill
    for regime, color in REGIME_COLORS.items():
        mask = df_plot["REGIME"] == regime
        ax2.fill_between(df_plot.index, 0, 1, where=mask.values,
                         color=color, alpha=0.7, transform=ax2.get_xaxis_transform())

    for start, end in recession_periods:
        ax2.axvspan(start, end, color="gray", alpha=0.3, zorder=2)

    ax2.set_yticks([])
    ax2.set_ylabel("Regime", fontsize=9)

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_regime_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


plot_regime_timeline(df, regime_series)


# =============================================================================
# CELL 7: Interview talking points
# =============================================================================

print("\n" + "="*70)
print("INTERVIEW TALKING POINTS — cross-asset regime findings")
print("="*70)
print("""
When asked "what did you find?" — say this:

  "I classified 20+ years of trading days into three yield curve regimes —
  steep, flat, and inverted — and measured how SPY, TLT, HYG, and LQD
  performed in each."

Then read them the key numbers from the table above:

  TLT in inverted regimes vs steep regimes: [fill in your numbers]
  → Long bonds outperform in inversions because the Fed cuts rates
    eventually, which pushes bond prices up.

  HYG in inverted regimes vs steep regimes: [fill in your numbers]
  → High yield underperforms because riskier companies face default risk
    when credit conditions tighten.

  SPY drawdown in inverted vs steep: [fill in your numbers]
  → Equities suffer more in inverted regimes as recession risk rises.

  Current regime as of today: [say what regime_series.iloc[-1] shows]
  → And what that historically implies for each asset class going forward.

The punchline for Bridgewater:
  "The cross-asset behavior validates the yield curve mechanism —
  it's not just a statistical correlation, it's a cause-effect chain
  that shows up consistently in asset prices across three decades."
""")

print("="*70)
print("Part 4 complete. All four analysis parts are done.")
print()
print("Next steps:")
print("  1. Write the Bridgewater macro memo (memo/yield_curve_memo.pdf)")
print("     Use the numbers from all four parts.")
print("  2. Fill in resume bullets with actual ROC-AUC and regime return numbers.")
print("  3. Push to GitHub with the README.")
print("="*70)
