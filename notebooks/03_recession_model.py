# =============================================================================
# 03_recession_model.py
# Recession prediction model — full analysis script.
#
# Run this AFTER 02_nelson_siegel.py.
# Produces:
#   1. Logistic regression summary table (printed to terminal)
#   2. ROC curve chart
#   3. Predicted recession probability over full history (headline chart)
#   4. Your resume headline finding sentence
#
# Run from terminal: python notebooks/03_recession_model.py
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_data
from recession_model import (
    build_recession_target,
    build_features,
    split_train_test,
    fit_logistic_model,
    evaluate_model,
    build_full_probability_series,
    build_headline_finding,
)
from config import FIGURES_DIR, TABLES_DIR, TRAIN_TEST_SPLIT

sns.set_theme(style="whitegrid", font_scale=1.1)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# =============================================================================
# CELL 1: Load data
# =============================================================================

print("Loading data (with NS factors from Part 2)...")
df = load_data()

# Verify NS factors are present — if not, tell the user to run Part 2 first
if "NS_BETA1" not in df.columns:
    raise RuntimeError(
        "NS factors not found in dataset.\n"
        "Run python notebooks/02_nelson_siegel.py first."
    )

print(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")


# =============================================================================
# CELL 2: Build target variable and features
# =============================================================================

print("\n" + "="*65)
print("STEP 1: Building target variable (12-month forward recession label)")
print("="*65)
target = build_recession_target(df)

print("\n" + "="*65)
print("STEP 2: Assembling feature matrix")
print("="*65)
features = build_features(df)

# Align target to features index (some rows may have been dropped due to NaN)
target = target[target.index.isin(features.index)]

print(f"\nAligned dataset: {len(features):,} rows")


# =============================================================================
# CELL 3: Train/test split
# =============================================================================

print("\n" + "="*65)
print("STEP 3: Chronological train/test split")
print("="*65)
X_train, X_test, y_train, y_test = split_train_test(features, target)


# =============================================================================
# CELL 4: Fit the logistic regression
# =============================================================================

print("\n" + "="*65)
print("STEP 4: Fitting logistic regression on training data (pre-2010)")
print("="*65)
model_result = fit_logistic_model(X_train, y_train)


# =============================================================================
# CELL 5: Evaluate model on test set
# =============================================================================

print("\n" + "="*65)
print("STEP 5: Evaluating model on test set (post-2010, out-of-sample)")
print("="*65)
eval_results = evaluate_model(model_result, X_train, X_test, y_train, y_test)


# =============================================================================
# CELL 6: Build full-history probability series
# =============================================================================

prob_full = build_full_probability_series(model_result, features)

# Add to main dataframe
df = df.join(prob_full, how="left")


# =============================================================================
# CELL 7: Headline finding
# =============================================================================

# Attach the prob_test series properly for hit-counting
eval_results["prob_test"] = pd.Series(
    eval_results["prob_test"].values,
    index=X_test.index
)

hits, total, auc = build_headline_finding(eval_results, df, features)


# =============================================================================
# CELL 8: CHART 1 — ROC Curve
# =============================================================================
#
# The ROC curve plots the trade-off between:
#   True Positive Rate  (recall — what fraction of recessions did we catch?)
#   False Positive Rate (what fraction of non-recession days did we falsely flag?)
#
# A perfect model hugs the top-left corner (catch everything, no false alarms).
# A random model is a diagonal line (no better than a coin flip).
# The AUC is the area under this curve — higher is better, 0.70+ is our target.

def plot_roc_curve(y_test, prob_test, auc_score):
    fpr, tpr, _ = roc_curve(y_test, prob_test)

    fig, ax = plt.subplots(figsize=(7, 7))

    # The actual ROC curve
    ax.plot(fpr, tpr, color="#d6604d", linewidth=2.5,
            label=f"Recession model (AUC = {auc_score:.3f})")

    # The diagonal = random model baseline
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1,
            linestyle="--", label="Random baseline (AUC = 0.50)")

    # Mark the optimal threshold point
    threshold = eval_results["threshold"]
    from sklearn.metrics import roc_curve as rc
    fprs, tprs, threshs = rc(y_test, prob_test)
    # Find the point closest to our optimal threshold
    idx = np.argmin(np.abs(threshs - threshold))
    ax.scatter(fprs[idx], tprs[idx], color="#d6604d", s=100, zorder=5,
               label=f"Optimal threshold ({threshold:.2f})")

    ax.set_xlabel("False Positive Rate\n(fraction of non-recession days falsely flagged)")
    ax.set_ylabel("True Positive Rate\n(fraction of pre-recession days correctly flagged)")
    ax.set_title("ROC Curve — Recession Prediction Model\nOut-of-sample evaluation (post-2010)",
                 fontsize=12)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Add AUC annotation
    ax.text(0.6, 0.15,
            f"AUC = {auc_score:.3f}\n{'✓ Exceeds 0.70 target' if auc_score >= 0.70 else '✗ Below 0.70 target'}",
            transform=ax.transAxes,
            fontsize=12, color="#d6604d",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d6604d", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "03_roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


plot_roc_curve(y_test, eval_results["prob_test"], eval_results["auc"])


# =============================================================================
# CELL 9: CHART 2 — Predicted recession probability over full history
# =============================================================================
#
# This is your HEADLINE CHART — the most important visualization in the project.
#
# What it shows:
#   - Blue line: model's predicted recession probability on each day (0 to 100%)
#   - Gray shading: actual NBER recession periods
#   - Red dashed line: the optimal threshold (above = model says "recession imminent")
#   - Train/test split line: separates in-sample from out-of-sample
#
# What you want to see:
#   - Probability spikes BEFORE each recession (leading indicator, not lagging)
#   - Probability drops between recessions (not stuck high forever)
#   - The 2020 COVID spike may be absent or weak (expected — no inversion preceded it)
#   - The 2022-2024 inversion period shows elevated probability

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


def plot_recession_probability(df, eval_results):
    """
    Headline chart: predicted recession probability over 30+ years.
    """
    prob_col = "RECESSION_PROB"
    if prob_col not in df.columns or df[prob_col].isna().all():
        print("No probability column found — skipping headline chart.")
        return

    recession_periods = get_recession_periods(df)
    split = pd.Timestamp(TRAIN_TEST_SPLIT)
    threshold = eval_results["threshold"]

    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot predicted probability as percentage (multiply by 100 for readability)
    df_plot = df.dropna(subset=[prob_col])
    ax.plot(df_plot.index, df_plot[prob_col] * 100,
            color="#2166ac", linewidth=1.0,
            label="Predicted recession probability (%)")

    # Fill above threshold in orange (model is flagging elevated risk)
    ax.fill_between(df_plot.index, df_plot[prob_col] * 100, threshold * 100,
                    where=(df_plot[prob_col] >= threshold),
                    color="#f4a582", alpha=0.5, label=f"Above threshold ({threshold*100:.0f}%)")

    # Recession shading
    for start, end in recession_periods:
        ax.axvspan(start, end, color="gray", alpha=0.25, zorder=0)
    ax.axvspan(df.index[0], df.index[0], color="gray", alpha=0.25,
               label="NBER Recession")

    # Threshold line
    ax.axhline(threshold * 100, color="#d6604d", linewidth=1.2,
               linestyle="--", alpha=0.8,
               label=f"Decision threshold ({threshold*100:.0f}%)")

    # Train/test split line
    ax.axvline(split, color="black", linewidth=1.2, linestyle=":",
               label=f"Train/test split ({TRAIN_TEST_SPLIT})")

    # Labels in the two regions
    ax.text(split - pd.Timedelta(days=1000), 92, "IN-SAMPLE\n(training)",
            fontsize=9, color="gray", ha="right")
    ax.text(split + pd.Timedelta(days=200), 92, "OUT-OF-SAMPLE\n(evaluation)",
            fontsize=9, color="gray", ha="left")

    ax.set_ylabel("Predicted Recession Probability (%)")
    ax.set_xlabel("Date")
    ax.set_ylim([0, 100])
    ax.set_title(
        f"Predicted 12-Month Recession Probability — Logistic Regression Model\n"
        f"Out-of-sample ROC-AUC: {eval_results['auc']:.3f} | "
        f"Gray bands = NBER recessions | "
        f"Orange = model flagging elevated risk",
        fontsize=11
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "03_recession_probability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


plot_recession_probability(df, eval_results)


# =============================================================================
# CELL 10: Save results table
# =============================================================================

results_table = pd.DataFrame({
    "Metric": ["ROC-AUC", "Precision", "Recall", "F1 Score",
               "Optimal Threshold", "Recessions Predicted", "Total Recessions"],
    "Value":  [
        f"{eval_results['auc']:.3f}",
        f"{eval_results['precision']:.3f}",
        f"{eval_results['recall']:.3f}",
        f"{eval_results['f1']:.3f}",
        f"{eval_results['threshold']:.2f}",
        f"{hits} of {total}",
        f"{total}",
    ]
})

table_path = os.path.join(TABLES_DIR, "03_model_metrics.csv")
results_table.to_csv(table_path, index=False)
print(f"\nMetrics table saved to: {table_path}")
print(results_table.to_string(index=False))


# =============================================================================
# CELL 11: Current signal — what is the model saying TODAY?
# =============================================================================
# This is the number you need memorized for every interview.

print("\n" + "="*65)
print("CURRENT SIGNAL — what the model says TODAY")
print("="*65)

if "RECESSION_PROB" in df.columns:
    latest = df.dropna(subset=["RECESSION_PROB"]).iloc[-1]
    prob_today = latest["RECESSION_PROB"] * 100
    spread_today = latest["SPREAD_2S10S"]
    date_today = latest.name.date()

    print(f"\n  As of {date_today}:")
    print(f"  2s10s spread:              {spread_today:+.2f}%")
    print(f"  Predicted recession prob:  {prob_today:.1f}%")
    print(f"  Decision threshold:        {eval_results['threshold']*100:.0f}%")
    print(f"  Model signal:              {'⚠ ELEVATED RISK' if prob_today >= eval_results['threshold']*100 else '✓ Below threshold'}")
    print()
    print("  Memorize this for interviews. Check FRED the morning of any interview")
    print("  and update the spread number. The probability will follow.")

print("\n" + "="*65)
print("Part 3 complete.")
print("Next: python notebooks/04_cross_asset_regimes.py")
print("="*65)
