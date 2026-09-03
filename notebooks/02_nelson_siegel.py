# =============================================================================
# 02_nelson_siegel.py
# Nelson-Siegel yield curve decomposition — full analysis script.
#
# Run this AFTER 01_data_preparation.py.
# It will load your cached CSV, fit the NS model, and produce:
#   1. Beta factor time series chart (with recession shading)
#   2. Four-panel regime chart (steep / flat / inverted / current)
#   3. An updated CSV with beta columns added
#
# Run from terminal: python notebooks/02_nelson_siegel.py
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # suppress scipy convergence warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_data
from nelson_siegel import (
    fit_nelson_siegel_all,
    get_fitted_curve,
    interpret_factors,
    nelson_siegel_curve
)
from config import FIGURES_DIR, COMBINED_CSV, NS_MATURITIES

sns.set_theme(style="whitegrid", font_scale=1.1)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# CELL 1: Load data
# =============================================================================

print("Loading data...")
df = load_data()
print(f"Loaded {len(df):,} rows")


# =============================================================================
# CELL 2: Fit Nelson-Siegel to every day
# =============================================================================
# This is the slow step. It takes 2-5 minutes.
# If you've already run this and saved the output CSV, skip to CELL 3.

# Check if we've already fitted (beta columns present in CSV)
# We force a re-fit if beta columns exist but used the OLD sign convention
# (i.e. beta1 was mostly negative across history, which is wrong after the fix).
# The correct convention: beta1 should be mostly POSITIVE (normal curve = positive slope).
needs_refit = True
if "NS_BETA1" in df.columns:
    median_b1 = df["NS_BETA1"].median()
    if median_b1 > 0:
        print(f"NS factors already fitted with correct sign convention (median β1={median_b1:.2f}) — skipping.")
        needs_refit = False
    else:
        print(f"Detected old sign convention (median β1={median_b1:.2f} — mostly negative).")
        print("Dropping stale NS columns and re-fitting with corrected convention...")
        df = df.drop(columns=["NS_BETA0", "NS_BETA1", "NS_BETA2", "NS_LAMBDA"], errors="ignore")

if needs_refit:
    df = fit_nelson_siegel_all(df)

    # Save the updated dataset with beta columns
    df.to_csv(COMBINED_CSV)
    print(f"\nUpdated dataset saved to {COMBINED_CSV}")

# Print what the factors are saying about the current curve
interpret_factors(df)


# =============================================================================
# CELL 3: Helper — get recession periods (same as Part 1)
# =============================================================================

def get_recession_periods(df):
    """Extract (start, end) tuples for every NBER recession in the dataset."""
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


recession_periods = get_recession_periods(df)


# =============================================================================
# CELL 4: CHART 1 — Beta factor time series with recession shading
# =============================================================================
#
# This chart shows β0, β1, β2 over 30+ years.
# Key thing to look for: β1 (slope) dips below zero just before each recession.
# That's visual proof that the decomposition is capturing the same signal
# as the raw spread chart — but now in factor form.
#
# β0 (level) will track the broad interest rate environment:
#   high in the early 1990s, very low 2010-2021, rising 2022-present.
# β2 (curvature) is noisier and less intuitive — that's normal.

def plot_ns_factors(df, recession_periods):
    """
    Three-panel chart: one panel per NS factor, all with recession shading.
    """
    # Drop rows where fitting failed
    df_plot = df.dropna(subset=["NS_BETA0", "NS_BETA1", "NS_BETA2"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    factor_config = [
        ("NS_BETA0", "β₀ — Level", "#2166ac",
         "Long-run rate where curve converges at long maturities"),
        ("NS_BETA1", "β₁ — Slope", "#d6604d",
         "Positive = normal upward curve  |  Negative = INVERTED  (watch this one)"),
        ("NS_BETA2", "β₂ — Curvature", "#4dac26",
         "Hump/trough shape in the middle of the curve"),
    ]

    for ax, (col, label, color, subtitle) in zip(axes, factor_config):
        ax.plot(df_plot.index, df_plot[col],
                color=color, linewidth=0.8, label=label)

        # Zero line — important reference for β1 (slope)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

        # Recession shading
        for start, end in recession_periods:
            ax.axvspan(start, end, color="gray", alpha=0.2)

        # For beta1 only: shade below zero in red (inversion zones)
        # After the sign flip: NS_BETA1 < 0 means inverted, matching spread convention
        if col == "NS_BETA1":
            ax.fill_between(df_plot.index, df_plot[col], 0,
                            where=(df_plot[col] < 0),
                            color="red", alpha=0.25, label="Inverted (β₁ < 0)")

        ax.set_ylabel("Value (%)", fontsize=10)
        ax.set_title(f"{label}  —  {subtitle}", fontsize=11, loc="left")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Format x-axis dates
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[-1].set_xlabel("Date")

    fig.suptitle(
        "Nelson-Siegel Yield Curve Factors (1990–Present)\n"
        "Gray bands = NBER recessions  |  Red zones on β₁ = yield curve inversions",
        fontsize=11
    )
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "02_ns_factors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


plot_ns_factors(df, recession_periods)


# =============================================================================
# CELL 5: CHART 2 — Four-panel regime yield curve visualization
# =============================================================================
#
# This is your best interview chart.
# It shows one actual yield curve from each of four regimes:
#   1. STEEP    — early 1990s, economy in expansion after recession
#   2. FLAT     — mid-2000s, Fed tightening but not yet inverted
#   3. INVERTED — 2006-2007, just before the financial crisis
#   4. CURRENT  — today
#
# Each panel shows:
#   - Dots: the 5 actual observed yields (3M, 2Y, 5Y, 10Y, 30Y)
#   - Line: the smooth Nelson-Siegel fitted curve through those dots
#
# You should see the curve shape change dramatically across regimes:
#   Steep = strong upward slope
#   Flat  = nearly horizontal
#   Inverted = downward slope (short rates above long rates)
#   Current = whatever it is today

def get_representative_date(df, target_date_str, yield_cols):
    """
    Find the closest date in the dataset to a target date string.
    Returns the date and that day's NS parameters and observed yields.
    """
    target = pd.Timestamp(target_date_str)
    # Get dates where NS fitting succeeded
    valid = df.dropna(subset=["NS_BETA0"])
    # Find closest date
    idx = valid.index.get_indexer([target], method="nearest")[0]
    date = valid.index[idx]
    row = valid.iloc[idx]
    return date, row


def plot_four_regime_curves(df):
    """
    Four-panel chart showing the yield curve shape in each regime.
    Uses actual dates that represent each regime well.
    """
    yield_cols = ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"]
    maturities_observed = np.array(NS_MATURITIES)  # [0.25, 2, 5, 10, 30]

    # Representative dates for each regime
    # These were chosen to illustrate each regime clearly
    regime_dates = {
        "Steep\n(Jan 1994)":    "1994-01-03",   # Post-1990 recession, normal expansion
        "Flat\n(Dec 2005)":     "2005-12-01",   # Fed tightening cycle, curve flattening
        "Inverted\n(Jul 2007)": "2007-07-02",   # Pre-financial crisis, deep inversion
        "Current":              df.dropna(subset=["NS_BETA0"]).index[-1].strftime("%Y-%m-%d"),
    }

    colors = ["#2166ac", "#f4a582", "#d6604d", "#1a9641"]
    labels = list(regime_dates.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # Makes it easier to index: axes[0], axes[1], etc.

    for ax, (label, date_str), color in zip(axes, regime_dates.items(), colors):
        date, row = get_representative_date(df, date_str, yield_cols)

        # Get observed yields (the actual 5 data points)
        observed = row[yield_cols].values.astype(float)

        # Get the smooth fitted curve (100 points)
        b0 = row["NS_BETA0"]
        b1 = row["NS_BETA1"]
        b2 = row["NS_BETA2"]
        lam = row["NS_LAMBDA"]

        m_fine, y_fitted = get_fitted_curve(b0, b1, b2, lam)

        # Plot the smooth fitted line
        ax.plot(m_fine, y_fitted,
                color=color, linewidth=2.5, label="NS fitted curve", zorder=2)

        # Plot the actual observed data points as dots
        ax.scatter(maturities_observed, observed,
                   color=color, s=80, zorder=3,
                   label="Observed yields", edgecolors="white", linewidth=1)

        # Zero reference line (helps visualize inversion)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        # Add beta values as text in the corner so you can see the factors
        textstr = f"β₀={b0:.2f}%  β₁={b1:.2f}%  β₂={b2:.2f}%"
        ax.text(0.03, 0.05, textstr, transform=ax.transAxes,
                fontsize=9, color="gray",
                verticalalignment="bottom")

        # Label maturity axis with readable names instead of numbers
        ax.set_xticks(maturities_observed)
        ax.set_xticklabels(["3M", "2Y", "5Y", "10Y", "30Y"])

        ax.set_xlabel("Maturity")
        ax.set_ylabel("Yield (%)")
        ax.set_title(f"{label}\n{date.strftime('%B %d, %Y')}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Print the factors so you can see them in terminal too
        slope_desc = "INVERTED" if b1 < 0 else ("flat" if b1 < 1 else "steep")
        print(f"{label.replace(chr(10), ' '):25s} | "
              f"Date: {date.date()} | "
              f"β₀={b0:.2f} β₁={b1:.2f} β₂={b2:.2f} | "
              f"Curve: {slope_desc}")

    fig.suptitle(
        "Yield Curve Shape Across Four Regimes\n"
        "Dots = observed Treasury yields  |  Line = Nelson-Siegel fitted curve",
        fontsize=12
    )
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "02_four_regime_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.show()


print("\nGenerating four-regime yield curve chart...")
plot_four_regime_curves(df)


# =============================================================================
# CELL 6: Factor statistics by recession state
# =============================================================================
#
# This table answers: do the NS factors actually differ during recessions?
# If the model is working, β1 should be significantly more negative
# in the pre-recession period than during normal expansions.

print("\n" + "=" * 65)
print("NELSON-SIEGEL FACTOR STATISTICS BY ECONOMIC STATE")
print("=" * 65)

df_valid = df.dropna(subset=["NS_BETA0", "NS_BETA1", "NS_BETA2"])

for state, mask in [
    ("Expansion (RECESSION=0)", df_valid["RECESSION"] == 0),
    ("Recession  (RECESSION=1)", df_valid["RECESSION"] == 1),
]:
    subset = df_valid[mask]
    print(f"\n{state} — {len(subset):,} trading days")
    print(f"  β₀ (level):     mean={subset['NS_BETA0'].mean():.2f}%  "
          f"std={subset['NS_BETA0'].std():.2f}%")
    print(f"  β₁ (slope):     mean={subset['NS_BETA1'].mean():.2f}%  "
          f"std={subset['NS_BETA1'].std():.2f}%")
    print(f"  β₂ (curvature): mean={subset['NS_BETA2'].mean():.2f}%  "
          f"std={subset['NS_BETA2'].std():.2f}%")

print("\n")
print("What this should show:")
print("  → β₁ is meaningfully more negative during recessions")
print("  → β₀ varies less — rates can be high or low during recessions")
print("  → This validates that β₁ is the recession-predictive factor")

print("\n" + "=" * 65)
print("Part 2 complete.")
print("Next: run python notebooks/03_recession_model.py")
print("=" * 65)
