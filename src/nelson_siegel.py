# =============================================================================
# nelson_siegel.py
# Nelson-Siegel yield curve decomposition.
#
# What this file does:
#   1. Defines the Nelson-Siegel formula as a Python function
#   2. Fits it to observed yields on every day in the dataset
#   3. Stores beta0 (level), beta1 (slope), beta2 (curvature) as new columns
#
# After running this, your DataFrame will have three new columns:
#   NS_BETA0, NS_BETA1, NS_BETA2
# These are the "factors" that describe the yield curve shape each day.
# =============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import NS_MATURITIES, NS_INITIAL_PARAMS


# =============================================================================
# THE NELSON-SIEGEL FORMULA
# =============================================================================

def nelson_siegel_curve(maturity, beta0, beta1, beta2, lam):
    """
    The Nelson-Siegel yield curve model.

    Given a maturity (in years) and four parameters, returns the predicted yield.

    Parameters
    ----------
    maturity : float or array
        Time to maturity in years. E.g., 0.25 = 3 months, 10 = 10 years.
    beta0 : float
        LEVEL factor. This is where the curve converges at very long maturities.
        Think of it as the "overall height" of the curve.
        Economically: the long-run expected nominal interest rate.
    beta1 : float
        SLOPE factor. Controls how steeply the curve rises from short to long.
        Positive = normal upward slope. Negative = inverted curve.
        Economically: the difference between long and short rates.
    beta2 : float
        CURVATURE factor. Controls whether the curve has a hump in the middle.
        Positive = hump (medium rates above both extremes).
        Negative = trough (medium rates below both extremes).
    lam : float
        DECAY (lambda). Controls how fast the slope and curvature effects
        fade as maturity increases. Higher lambda = effects decay faster.
        This is a technical parameter — you interpret it less than the betas.

    Returns
    -------
    float or array
        Predicted yield at the given maturity, in percentage points.

    The Formula (broken down):
    --------------------------
    There are three terms:

    Term 1: beta0
        Flat line. Every maturity gets exactly beta0.
        At very long maturities, this is all that's left.

    Term 2: beta1 * [(1 - exp(-m/λ)) / (m/λ)]
        Starts at beta1 when maturity=0, decays to 0 as maturity→∞.
        This is the "slope loading" — how much the slope factor affects
        each maturity. Short maturities feel it most, long maturities less.

    Term 3: beta2 * [(1 - exp(-m/λ)) / (m/λ) - exp(-m/λ)]
        Starts at 0, peaks at an intermediate maturity, then decays to 0.
        This is the "curvature loading" — creates the hump shape.
        Medium maturities feel this most.
    """
    # Avoid division by zero if maturity is 0 (shouldn't happen in practice)
    maturity = np.maximum(maturity, 1e-10)

    # The "loading" term — appears in both slope and curvature
    # exp(-maturity / lam) is an exponential decay function
    # As maturity increases, this term shrinks toward 0
    loading = (1 - np.exp(-maturity / lam)) / (maturity / lam)

    # Assemble the three terms
    term1 = beta0                                           # level
    term2 = beta1 * loading                                 # slope
    term3 = beta2 * (loading - np.exp(-maturity / lam))    # curvature

    return term1 + term2 + term3


# =============================================================================
# FIT NELSON-SIEGEL TO A SINGLE DAY'S YIELDS
# =============================================================================

def fit_one_day(observed_yields, maturities=NS_MATURITIES, initial_params=NS_INITIAL_PARAMS):
    """
    Fits the Nelson-Siegel model to one day's observed yields.

    This uses scipy's curve_fit function, which is an optimization algorithm.
    It tries different values of beta0/beta1/beta2/lambda, computes what the
    formula predicts, compares to actual yields, and adjusts until the
    predictions match as closely as possible. This process is called
    "least squares optimization."

    Parameters
    ----------
    observed_yields : array-like
        The actual Treasury yields on this date.
        Must be in the same order as maturities: [3M, 2Y, 5Y, 10Y, 30Y]
    maturities : list
        The corresponding maturities in years: [0.25, 2, 5, 10, 30]
    initial_params : list
        Starting guess for [beta0, beta1, beta2, lambda].
        The optimizer starts here and searches for better values.

    Returns
    -------
    tuple (beta0, beta1, beta2, lam) or (nan, nan, nan, nan)
        The fitted parameters. Returns NaN if fitting fails.
    """
    maturities = np.array(maturities)
    observed_yields = np.array(observed_yields)

    # If any of the observed yields are NaN (missing), we can't fit
    if np.any(np.isnan(observed_yields)):
        return np.nan, np.nan, np.nan, np.nan

    try:
        # curve_fit finds the parameter values that minimize the difference
        # between nelson_siegel_curve(maturities, *params) and observed_yields.
        # p0 = initial guess. bounds = constraints on parameter values.
        # lambda must be positive (can't have negative decay).
        # beta0 must be positive (we don't expect negative long-run rates for US).
        params, _ = curve_fit(
            nelson_siegel_curve,
            maturities,
            observed_yields,
            p0=initial_params,
            bounds=(
                [0,    -8, -8, 0.1],   # lower bounds: [β0, β1, β2, λ]
                [20,    8,  8, 5.0],   # upper bounds — tighter to prevent runaway fits
            ),
            maxfev=5000   # maximum number of optimizer iterations
        )
        beta0, beta1, beta2, lam = params
        return beta0, beta1, beta2, lam

    except (RuntimeError, ValueError):
        # RuntimeError: optimizer couldn't converge within maxfev iterations
        # ValueError: something unexpected about the data shape
        # In either case, return NaN for this date — we'll handle it later
        return np.nan, np.nan, np.nan, np.nan


# =============================================================================
# FIT NELSON-SIEGEL TO THE ENTIRE DATASET
# =============================================================================

def fit_nelson_siegel_all(df, yield_cols=None):
    """
    Applies Nelson-Siegel fitting to every row in the dataset.

    This is the main function you call. It loops over every trading day,
    fits the curve, and stores beta0/beta1/beta2/lambda as new columns.

    This takes a few minutes to run (fitting ~8,500 days).
    A progress bar (tqdm) shows you how far along it is.

    Parameters
    ----------
    df : pd.DataFrame
        Your full dataset from data_loader.py
    yield_cols : list, optional
        Column names for the five yield series.
        Defaults to ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"]

    Returns
    -------
    pd.DataFrame
        The same DataFrame with four new columns:
        NS_BETA0, NS_BETA1, NS_BETA2, NS_LAMBDA
    """
    if yield_cols is None:
        yield_cols = ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"]

    print("Fitting Nelson-Siegel model to every trading day...")
    print(f"Processing {len(df):,} rows. This takes 2-5 minutes.")
    print("Each row = one day. Each fit = find β0, β1, β2, λ for that day's curve.")

    # Try to import tqdm for the progress bar
    # If not installed, fall back to no progress bar
    try:
        from tqdm import tqdm
        rows = tqdm(df.iterrows(), total=len(df), desc="Fitting NS model")
    except ImportError:
        print("(Install tqdm for a progress bar: pip install tqdm)")
        rows = df.iterrows()

    # Store results in lists — faster than modifying DataFrame row by row
    results = []

    for date, row in rows:
        # Extract just the five yield values for this day
        yields = row[yield_cols].values.astype(float)

        # Fit the curve and get parameters
        beta0, beta1, beta2, lam = fit_one_day(yields)

        results.append({
            "NS_BETA0":  beta0,
            "NS_BETA1":  -beta1,  # SIGN FLIP: raw NS beta1 is negative when curve is
                                   # normal, positive when inverted — opposite of intuition.
                                   # We negate it so convention matches the raw spread:
                                   #   NS_BETA1 > 0 = normal upward-sloping curve
                                   #   NS_BETA1 < 0 = inverted curve
            "NS_BETA2":  beta2,
            "NS_LAMBDA": lam,
        })

    # Convert results list to DataFrame, then join back to original
    results_df = pd.DataFrame(results, index=df.index)
    df = df.join(results_df)

    # Winsorize extreme outlier values.
    # On some days the optimizer converges to a mathematically valid but
    # economically absurd solution (e.g. beta1 = -10 when rates are only 5%).
    # These are numerical artifacts. We clip at the 1st/99th percentile so
    # they don't distort charts or the regression model downstream.
    for col in ["NS_BETA0", "NS_BETA1", "NS_BETA2"]:
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=p01, upper=p99)

    # Report results
    failed = df["NS_BETA0"].isna().sum()
    success = len(df) - failed
    print(f"\nFitting complete.")
    print(f"  Successfully fit: {success:,} days ({success/len(df)*100:.1f}%)")
    if failed > 0:
        print(f"  Failed to fit:    {failed:,} days (will be NaN — acceptable if small)")
    print(f"  Beta1 sign: positive = normal curve, negative = inverted")

    return df


# =============================================================================
# COMPUTE FITTED CURVE FOR A SINGLE DATE (for plotting)
# =============================================================================

def get_fitted_curve(beta0, beta1, beta2, lam, num_points=100):
    """
    Given fitted NS parameters, returns a smooth fitted curve
    for plotting — not just the 5 observed maturities, but 100 points
    spread across the full maturity range.

    This is what makes the Nelson-Siegel curve look smooth in charts.

    Parameters
    ----------
    beta0, beta1, beta2, lam : float
        The fitted parameters for a specific date.
    num_points : int
        How many points to compute along the curve.

    Returns
    -------
    maturities_fine : np.array  (100 maturity values from 0 to 30 years)
    yields_fitted   : np.array  (100 corresponding yield values)
    """
    # linspace creates evenly spaced values from 0.1 to 30
    maturities_fine = np.linspace(0.1, 30, num_points)
    yields_fitted = nelson_siegel_curve(maturities_fine, beta0, beta1, beta2, lam)
    return maturities_fine, yields_fitted


# =============================================================================
# INTERPRET THE FACTORS
# =============================================================================

def interpret_factors(df):
    """
    Prints a plain-English summary of what the NS factors are saying
    for the most recent day in the dataset.

    Call this after fitting to sanity-check that the factors make sense.
    """
    latest = df.dropna(subset=["NS_BETA0", "NS_BETA1", "NS_BETA2"]).iloc[-1]
    date = latest.name

    print(f"\nNELSON-SIEGEL FACTOR INTERPRETATION — {date.date()}")
    print("=" * 55)

    b0, b1, b2 = latest["NS_BETA0"], latest["NS_BETA1"], latest["NS_BETA2"]

    # Beta 0 — Level
    print(f"\nβ₀ (Level):     {b0:.2f}%")
    print(f"  → The long-run rate the curve converges to.")
    if b0 > 5:
        print(f"  → High rate environment (above 5%)")
    elif b0 < 3:
        print(f"  → Low rate environment (below 3%)")
    else:
        print(f"  → Moderate rate environment")

    # Beta 1 — Slope (after sign flip: positive = normal, negative = inverted)
    print(f"\nβ₁ (Slope):     {b1:.2f}%")
    print(f"  → Sign of the slope factor (corrected: positive = normal, negative = inverted).")
    if b1 < -1:
        print(f"  → STRONGLY INVERTED. Short rates well above long rates.")
        print(f"  → Historically, this level of inversion has consistently")
        print(f"     preceded recessions within 12-18 months.")
    elif b1 < 0:
        print(f"  → MILDLY INVERTED. Short rates slightly above long rates.")
    elif b1 < 1:
        print(f"  → FLAT to mildly positive slope. Transitional zone.")
    else:
        print(f"  → NORMAL upward slope. Economy in expansion phase.")

    # Beta 2 — Curvature
    print(f"\nβ₂ (Curvature): {b2:.2f}%")
    if b2 > 0:
        print(f"  → Positive hump: medium maturities yield MORE than both ends.")
    else:
        print(f"  → Negative trough: medium maturities yield LESS than both ends.")

    print()
