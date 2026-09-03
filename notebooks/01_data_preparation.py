# =============================================================================
# 01_data_preparation.py
# This file mirrors the Jupyter notebook 01_data_preparation.ipynb
# but as a plain .py script for people who prefer running scripts directly.
#
# If you're using Jupyter: open notebooks/01_data_preparation.ipynb instead.
# If you're running from terminal: python notebooks/01_data_preparation.py
# =============================================================================

# ── CELL 1: Imports ──────────────────────────────────────────────────────────
# We import everything we need at the top.
# Think of "import" like plugging in a tool before using it.

import sys
import os
import pandas as pd         # pandas: our spreadsheet engine
import matplotlib.pyplot as plt   # matplotlib: our charting library
import matplotlib.dates as mdates  # helps format dates on chart axes
import seaborn as sns       # seaborn: prettier chart styles built on matplotlib

# Add src/ to the path so Python can find our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_data, print_summary
from config import FIGURES_DIR

# Make the charts look clean and professional
# "whitegrid" adds light grid lines on a white background
sns.set_theme(style="whitegrid", font_scale=1.1)

print("All imports successful.")


# ── CELL 2: Load the data ────────────────────────────────────────────────────
# This single call does everything:
#   - Checks if data is already saved locally
#   - If yes: loads from CSV (fast)
#   - If no: downloads from FRED, cleans, saves
#
# The first time you run this, it will ask you to add your FRED API key
# to src/config.py if you haven't already.

df = load_data()

# Print a summary so we can verify the data looks right
print_summary(df)


# ── CELL 3: Inspect the data ─────────────────────────────────────────────────
# .head() shows the first 5 rows — always do this after loading data
# to make sure the structure looks right.

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names and data types:")
print(df.dtypes)

# .describe() gives you statistical summaries: mean, min, max, etc.
# This is a quick sanity check — e.g., yields should never be negative
# (though they were briefly in some countries, not in the US in this dataset)
print("\nStatistical summary:")
print(df[["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30",
          "SPREAD_2S10S", "SPREAD_3M10S"]].describe().round(2))


# ── CELL 4: SANITY CHECK CHART — The most important first chart ──────────────
#
# This is Step 5 from the project spec: plot both spreads with NBER shading.
# You need to see inversions precede the recessions.
# If they don't, something is wrong with your data alignment.
#
# What we're looking for:
#   - The spread lines dip below zero (inversion) BEFORE each gray band
#   - Gray bands = recession periods (from NBER data)
#   - Recessions in dataset: 1990, 2001, 2007-09, 2020

def get_recession_periods(df):
    """
    Extracts the start and end dates of each recession period in the dataset.

    We do this by finding every date where RECESSION goes from 0 to 1 (start)
    and from 1 to 0 (end). Returns a list of (start, end) tuples.
    """
    recession_periods = []

    # shift(1) moves the column down by 1 row.
    # So df["RECESSION"].shift(1) is "what was the value YESTERDAY?"
    # When today=1 and yesterday=0, that's a recession START.
    in_recession = False
    start_date = None

    for date, row in df.iterrows():
        if row["RECESSION"] == 1 and not in_recession:
            in_recession = True
            start_date = date
        elif row["RECESSION"] == 0 and in_recession:
            in_recession = False
            recession_periods.append((start_date, date))

    # Handle case where data ends during a recession
    if in_recession:
        recession_periods.append((start_date, df.index[-1]))

    return recession_periods


def plot_spreads_with_recession_shading(df):
    """
    Creates the headline sanity check chart:
    - Top panel: 2s10s spread (10-year minus 2-year)
    - Bottom panel: 3m10s spread (10-year minus 3-month)
    - Gray shading: NBER recession periods

    If inversions (below the zero line) appear BEFORE each gray band,
    your data is correct.
    """
    recession_periods = get_recession_periods(df)

    # Create a figure with 2 subplots stacked vertically
    # figsize=(14, 8): 14 inches wide, 8 inches tall
    # sharex=True: both panels share the same x-axis (date axis)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ── TOP PANEL: 2s10s spread ──────────────────────────────────────────────
    ax1.plot(df.index, df["SPREAD_2S10S"],
             color="#1f77b4",   # standard matplotlib blue
             linewidth=0.8,
             label="2s10s Spread (10Y − 2Y)")

    # Horizontal zero line — inversion = anything below this
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Shade the area below zero in light red (the inversion zones)
    ax1.fill_between(df.index, df["SPREAD_2S10S"], 0,
                     where=(df["SPREAD_2S10S"] < 0),
                     color="red", alpha=0.2, label="Inverted")

    # Add gray recession shading on both panels
    # axvspan draws a vertical shaded region between two x-values (dates here)
    for start, end in recession_periods:
        ax1.axvspan(start, end, color="gray", alpha=0.2)

    # Add one invisible gray bar for the legend label
    ax1.axvspan(df.index[0], df.index[0],
                color="gray", alpha=0.2, label="NBER Recession")

    ax1.set_ylabel("Spread (percentage points)")
    ax1.set_title("2s10s Yield Spread (10-Year minus 2-Year Treasury)", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── BOTTOM PANEL: 3m10s spread ───────────────────────────────────────────
    ax2.plot(df.index, df["SPREAD_3M10S"],
             color="#ff7f0e",   # orange
             linewidth=0.8,
             label="3m10s Spread (10Y − 3M)")

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax2.fill_between(df.index, df["SPREAD_3M10S"], 0,
                     where=(df["SPREAD_3M10S"] < 0),
                     color="red", alpha=0.2, label="Inverted")

    for start, end in recession_periods:
        ax2.axvspan(start, end, color="gray", alpha=0.2)

    ax2.axvspan(df.index[0], df.index[0],
                color="gray", alpha=0.2, label="NBER Recession")

    ax2.set_ylabel("Spread (percentage points)")
    ax2.set_xlabel("Date")
    ax2.set_title("3m10s Yield Spread (10-Year minus 3-Month Treasury)", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Format the x-axis to show years nicely
    # AutoDateLocator picks appropriate tick spacing
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add a big title for the whole figure
    fig.suptitle(
        "US Treasury Yield Curve Spreads with NBER Recession Shading (1990–Present)\n"
        "Key check: spread inversions (red zones) should precede gray recession bands",
        fontsize=11,
        y=1.01
    )

    plt.tight_layout()

    # Save the figure
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "01_spread_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {save_path}")

    plt.show()


# ── CELL 5: Run the chart ────────────────────────────────────────────────────
plot_spreads_with_recession_shading(df)


# ── CELL 6: WHAT TO LOOK FOR ─────────────────────────────────────────────────
#
# This cell just prints guidance — no code to run.
# Read this while looking at your chart.

print("""
WHAT TO LOOK FOR IN YOUR CHART:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each gray band is a recession. Each red zone is an inversion.
The red zones should appear BEFORE the gray bands — not during, not after.

Expected pattern:
  ✓ 1989–1990: Inversion → 1990 recession (gray band)
  ✓ 2000–2001: Inversion → 2001 recession (gray band)
  ✓ 2006–2007: Inversion → 2008–09 recession (gray band)
  ~ 2020:      Very brief/mild inversion → COVID recession (special case)
               The COVID recession was caused by an external shock, not
               credit contraction, so the signal was weak. Note this.
  ? 2022–2023: Deep inversion → recession? (still playing out)

If your chart shows inversions AFTER recessions, your data alignment is wrong.
If there are no inversions at all, your spread calculation is wrong (maybe subtracted backwards).
""")


# ── CELL 7: Quick inversion statistics ───────────────────────────────────────
# Just some numbers to have ready when asked "what did you find?"

print("INVERSION STATISTICS")
print("=" * 50)

for spread_col, invert_col, name in [
    ("SPREAD_2S10S", "INVERTED_2S10S", "2s10s"),
    ("SPREAD_3M10S", "INVERTED_3M10S", "3m10s"),
]:
    total_days     = len(df)
    inverted_days  = df[invert_col].sum()
    pct_inverted   = inverted_days / total_days * 100
    min_spread     = df[spread_col].min()
    max_spread     = df[spread_col].max()
    current_spread = df[spread_col].iloc[-1]

    print(f"\n{name} spread:")
    print(f"  Current value:    {current_spread:+.2f}%")
    print(f"  Days inverted:    {inverted_days:,} of {total_days:,} ({pct_inverted:.1f}%)")
    print(f"  Most inverted:    {min_spread:.2f}%")
    print(f"  Most steep:       {max_spread:.2f}%")

print("\nPart 1 complete. Your data is loaded and verified.")
print("Next: Run 02_nelson_siegel.py (or open the notebook)")
