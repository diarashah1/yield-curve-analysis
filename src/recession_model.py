# =============================================================================
# recession_model.py
# Logistic regression recession prediction model.
#
# What this file does:
#   1. Builds the forward-looking recession label (target variable)
#   2. Assembles the feature set from yield spreads and NS factors
#   3. Fits a logistic regression on pre-2010 data
#   4. Evaluates on post-2010 out-of-sample data
#   5. Returns predicted probabilities for the full history
#
# Key concept — logistic regression:
#   Regular regression predicts a number (e.g. "next year's GDP = 2.3%").
#   Logistic regression predicts a PROBABILITY between 0 and 1
#   (e.g. "62% chance of recession in the next 12 months").
#   It uses the logistic/sigmoid function to squish any input into [0,1].
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)
import statsmodels.api as sm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RECESSION_HORIZON_DAYS, TRAIN_TEST_SPLIT


# =============================================================================
# STEP 1: BUILD THE TARGET VARIABLE
# =============================================================================

def build_recession_target(df, horizon_days=RECESSION_HORIZON_DAYS):
    """
    Creates a forward-looking binary recession label for each date.

    For each day t, we look forward `horizon_days` trading days (~12 months).
    If ANY day in that window has RECESSION=1, we label day t as 1.
    Otherwise, day t gets label 0.

    Why forward-looking?
    We want to know: "does today's yield curve predict a recession AHEAD?"
    Not "are we already in a recession?" The yield curve is a leading indicator
    — it signals recession risk 12-18 months in advance, not in real time.

    Important caveat (note this in your README and interviews):
    This creates "lookahead bias" by construction — we're using future data
    to create the label. That's intentional for studying historical predictive
    power, NOT for a live trading signal. In a live system you'd never have
    access to future recession dates.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a RECESSION column (0/1)
    horizon_days : int
        How many trading days forward to look. Default 252 ≈ 12 months.

    Returns
    -------
    pd.Series
        Binary series: 1 = recession starts within horizon, 0 = no recession
    """
    recession = df["RECESSION"].copy()

    # rolling(horizon_days).max() looks at each window of horizon_days
    # and returns 1 if ANY day in that window was a recession, 0 otherwise.
    # shift(-horizon_days) shifts the result backward so it aligns with
    # the START of each window rather than the end.
    # In other words: "did a recession happen in the next 252 days FROM today?"
    forward_recession = (
        recession
        .rolling(window=horizon_days, min_periods=1)
        .max()
        .shift(-horizon_days)
    )

    # Convert to integer (0 or 1) — rolling().max() returns float
    forward_recession = forward_recession.fillna(0).astype(int)
    forward_recession.name = "RECESSION_12M"

    n_positive = forward_recession.sum()
    n_total = len(forward_recession)
    print(f"Target variable built:")
    print(f"  Label=1 (recession within 12M): {n_positive:,} days ({n_positive/n_total*100:.1f}%)")
    print(f"  Label=0 (no recession):          {n_total-n_positive:,} days ({(n_total-n_positive)/n_total*100:.1f}%)")

    return forward_recession


# =============================================================================
# STEP 2: ASSEMBLE FEATURES
# =============================================================================

def build_features(df):
    """
    Assembles the feature matrix for the logistic regression.

    Features:
      SPREAD_2S10S         — primary recession predictor (10Y - 2Y spread)
      SPREAD_3M10S         — secondary predictor (10Y - 3M spread)
      NS_BETA1             — slope factor from Nelson-Siegel (should track spread)
      NS_BETA0             — level factor (high rates = tighter conditions)
      SPREAD_2S10S_3M_CHANGE — 3-month momentum: is the curve flattening fast?

    Why these five?
    Each captures a different dimension of yield curve information:
    - The two spreads: direct inversion signal
    - Beta1: the smooth decomposed slope (less noisy than raw spread on some days)
    - Beta0: the rate level environment (recessions happen differently at 8% vs 2%)
    - Momentum: a curve that's flattening FAST is more dangerous than one
      that's been flat for years — this captures that dynamic

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per trading day
    """
    features = pd.DataFrame(index=df.index)

    features["SPREAD_2S10S"]          = df["SPREAD_2S10S"]
    features["SPREAD_3M10S"]          = df["SPREAD_3M10S"]
    features["NS_BETA1"]              = df["NS_BETA1"]
    features["NS_BETA0"]              = df["NS_BETA0"]
    features["SPREAD_2S10S_3M_CHANGE"] = df["SPREAD_2S10S_3M_CHANGE"]

    # Drop rows where any feature is NaN
    # (happens at the start of the series before rolling calculations fill in)
    features = features.dropna()

    print(f"Feature matrix: {len(features):,} rows x {len(features.columns)} columns")
    print(f"Features: {list(features.columns)}")

    return features


# =============================================================================
# STEP 3: CHRONOLOGICAL TRAIN/TEST SPLIT
# =============================================================================

def split_train_test(features, target, split_date=TRAIN_TEST_SPLIT):
    """
    Splits data chronologically into training and test sets.

    WHY chronological (not random)?
    In time series, the future depends on the past. If you randomly assign
    days to train/test, future data leaks into training — the model "cheats"
    by learning from data it shouldn't have access to. This makes evaluation
    metrics look better than they really are.

    The correct approach: train on all data before the split date,
    test on all data after. The model never sees the test period during training.

    Pre-2010 training gives us:
      - 1990-2001 expansion and dot-com recession
      - 2001-2007 expansion
      - 2007-2009 financial crisis
    Post-2010 test gives us:
      - 2010-2019 expansion (true out-of-sample)
      - 2020 COVID recession (true out-of-sample)
      - 2022-2024 inversion (true out-of-sample)

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrames/Series
    """
    split = pd.Timestamp(split_date)

    X_train = features[features.index < split]
    X_test  = features[features.index >= split]
    y_train = target[target.index.isin(X_train.index)]
    y_test  = target[target.index.isin(X_test.index)]

    print(f"\nTrain/test split at {split_date}:")
    print(f"  Train: {len(X_train):,} days ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"  Test:  {len(X_test):,} days  ({X_test.index[0].date()} → {X_test.index[-1].date()})")
    print(f"  Train recession rate: {y_train.mean()*100:.1f}%")
    print(f"  Test  recession rate: {y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 4: FIT THE LOGISTIC REGRESSION
# =============================================================================

def fit_logistic_model(X_train, y_train):
    """
    Fits a logistic regression model using statsmodels.

    Why statsmodels instead of sklearn?
    Statsmodels gives you a full statistical summary table — p-values,
    confidence intervals, standard errors. This is what a quant researcher
    or rates desk analyst would actually use to validate that the coefficients
    are statistically meaningful, not just numerically non-zero.

    The key thing to check in the output:
      - SPREAD_2S10S coefficient should be NEGATIVE and significant
        (more negative spread = higher recession probability)
      - NS_BETA1 coefficient should also be negative (lower slope = more risk)
      - p-values < 0.05 indicate statistical significance

    What is a p-value?
    The probability of seeing this coefficient by random chance if the true
    coefficient were zero. Small p-value = the relationship is real, not noise.
    Convention: p < 0.05 is considered statistically significant.

    Parameters
    ----------
    X_train : pd.DataFrame  Feature matrix for training period
    y_train : pd.Series     Binary recession labels for training period

    Returns
    -------
    model : fitted statsmodels Logit model object
    """
    # add_constant adds an intercept term (a baseline probability level).
    # Without it, the model is forced through the origin, which is too restrictive.
    X_train_const = sm.add_constant(X_train)

    print("Fitting logistic regression on training data...")
    model = sm.Logit(y_train, X_train_const)
    result = model.fit(disp=False)  # disp=False suppresses optimizer output

    print("\n" + "="*65)
    print("LOGISTIC REGRESSION SUMMARY")
    print("="*65)
    print(result.summary2())

    # Print plain-English interpretation of key coefficients
    print("\nKEY COEFFICIENT INTERPRETATIONS:")
    print("-"*65)
    coefs = result.params
    pvals = result.pvalues

    for feat, direction, meaning in [
        ("SPREAD_2S10S",
         "negative", "more inverted 2s10s → higher recession risk"),
        ("SPREAD_3M10S",
         "negative", "more inverted 3m10s → higher recession risk"),
        ("NS_BETA1",
         "negative", "lower slope factor → higher recession risk"),
        ("NS_BETA0",
         "positive", "higher rate level → tighter conditions"),
        ("SPREAD_2S10S_3M_CHANGE",
         "negative", "faster flattening → higher recession risk"),
    ]:
        if feat in coefs:
            actual_sign = "negative" if coefs[feat] < 0 else "positive"
            sig = "✓ significant" if pvals[feat] < 0.05 else "✗ not significant"
            match = "✓" if actual_sign == direction else "✗ UNEXPECTED SIGN"
            print(f"  {feat:30s}: coef={coefs[feat]:+.3f}  p={pvals[feat]:.3f}  {sig}  {match}")

    return result


# =============================================================================
# STEP 5: PREDICT AND EVALUATE
# =============================================================================

def evaluate_model(result, X_train, X_test, y_train, y_test):
    """
    Generates predicted recession probabilities and evaluates model performance.

    Metrics explained:

    ROC-AUC (Area Under the ROC Curve):
      Measures how well the model ranks days — are high-probability days
      actually followed by recessions? Ranges from 0.5 (random) to 1.0 (perfect).
      Target: ≥ 0.70.

    Precision:
      Of all the days the model flagged as "recession incoming," what fraction
      actually were followed by a recession?
      High precision = few false alarms.

    Recall:
      Of all the actual pre-recession days, what fraction did the model catch?
      High recall = few missed recessions.

    F1 Score:
      Harmonic mean of precision and recall. Balances the two.
      Used to find the optimal probability threshold.

    Confusion Matrix:
      A 2x2 table showing:
        True Positives  (model said recession, recession happened)
        False Positives (model said recession, no recession — false alarm)
        True Negatives  (model said no recession, no recession)
        False Negatives (model said no recession, recession happened — miss)

    Returns
    -------
    dict with predicted probabilities, threshold, and all metrics
    """
    X_train_const = sm.add_constant(X_train)
    X_test_const  = sm.add_constant(X_test, has_constant="add")

    # Predict probabilities on both train and test sets
    prob_train = result.predict(X_train_const)
    prob_test  = result.predict(X_test_const)

    # Find optimal threshold on training set (maximizes F1)
    # Threshold = the probability cutoff above which we call it "recession imminent"
    # We search over all possible thresholds and pick the one with the best F1 score
    thresholds = np.linspace(0.05, 0.95, 200)
    f1_scores  = [f1_score(y_train, (prob_train >= t).astype(int),
                           zero_division=0) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    # Apply threshold to test set
    pred_test = (prob_test >= optimal_threshold).astype(int)

    # Calculate all metrics on the TEST set (out-of-sample)
    auc       = roc_auc_score(y_test, prob_test)
    precision = precision_score(y_test, pred_test, zero_division=0)
    recall    = recall_score(y_test, pred_test, zero_division=0)
    f1        = f1_score(y_test, pred_test, zero_division=0)
    cm        = confusion_matrix(y_test, pred_test)

    print("\n" + "="*65)
    print("MODEL EVALUATION — OUT-OF-SAMPLE (POST-2010)")
    print("="*65)
    print(f"  Optimal threshold (from training F1): {optimal_threshold:.2f}")
    print(f"  ROC-AUC:   {auc:.3f}  (target ≥ 0.70)")
    print(f"  Precision: {precision:.3f}  (of flagged days, this fraction were real)")
    print(f"  Recall:    {recall:.3f}  (of real pre-recession days, this fraction caught)")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"\n  Confusion Matrix (test set):")
    print(f"                   Predicted 0    Predicted 1")
    print(f"  Actual 0 (exp):  {cm[0,0]:>10,}     {cm[0,1]:>10,}  (false alarms)")
    print(f"  Actual 1 (rec):  {cm[1,0]:>10,}     {cm[1,1]:>10,}  (hits)")

    if auc >= 0.70:
        print(f"\n  ✓ ROC-AUC of {auc:.3f} exceeds target threshold of 0.70")
    else:
        print(f"\n  ✗ ROC-AUC of {auc:.3f} is below target — model has limited predictive power")
        print(f"    This can happen if the post-2010 period had unusual dynamics.")
        print(f"    Note the 2020 COVID recession explicitly broke the pattern.")

    return {
        "prob_train":          prob_train,
        "prob_test":           prob_test,
        "threshold":           optimal_threshold,
        "auc":                 auc,
        "precision":           precision,
        "recall":              recall,
        "f1":                  f1,
        "confusion_matrix":    cm,
        "pred_test":           pred_test,
    }


# =============================================================================
# STEP 6: COMBINE PREDICTED PROBABILITIES INTO FULL HISTORY SERIES
# =============================================================================

def build_full_probability_series(result, features):
    """
    Runs the fitted model on the ENTIRE dataset (train + test combined)
    to get a predicted recession probability for every day in history.

    This is what you plot in the headline chart — 30 years of model output
    overlaid with recession shading.

    Note: using train-period predictions is in-sample (the model has seen
    those dates). We show the full history for visualization purposes only.
    The out-of-sample evaluation (post-2010) is what matters for the resume.
    """
    X_all_const = sm.add_constant(features, has_constant="add")
    prob_all = result.predict(X_all_const)
    prob_series = pd.Series(prob_all, index=features.index, name="RECESSION_PROB")
    return prob_series


# =============================================================================
# HEADLINE FINDING BUILDER
# =============================================================================

def build_headline_finding(result_dict, df, features):
    """
    Generates the headline sentence for your resume bullet and Bridgewater memo.

    Counts how many NBER recessions occurred during the test period and
    how many the model successfully flagged (prob exceeded threshold before onset).

    Template: "The 2s10s spread inversion predicted X of the last Y NBER
    recessions within 12 months, with the logistic model achieving ROC-AUC
    of Z on out-of-sample data from 2010 to present."
    """
    split = pd.Timestamp(TRAIN_TEST_SPLIT)
    df_test = df[df.index >= split].copy()

    # Find recession start dates in the test period
    recession_starts = []
    in_rec = False
    for date, row in df_test.iterrows():
        if row["RECESSION"] == 1 and not in_rec:
            in_rec = True
            recession_starts.append(date)
        elif row["RECESSION"] == 0 and in_rec:
            in_rec = False

    # For each recession start, check if model probability exceeded threshold
    # at any point in the 252 days BEFORE it started
    prob_series = result_dict["prob_train"]  # use full series if available
    threshold   = result_dict["threshold"]

    hits = 0
    for start in recession_starts:
        window_start = start - pd.Timedelta(days=365)
        window_end   = start

        # Check if prob exceeded threshold in the pre-recession window
        if start in result_dict.get("prob_test", pd.Series()).index:
            prob = result_dict["prob_test"]
        else:
            continue

        window = prob[(prob.index >= window_start) & (prob.index < window_end)]
        if len(window) > 0 and (window >= threshold).any():
            hits += 1

    total = len(recession_starts)
    auc   = result_dict["auc"]

    print("\n" + "="*65)
    print("HEADLINE FINDING (for resume and Bridgewater memo)")
    print("="*65)
    print(f"\n  Recessions in test period (post-2010): {total}")
    print(f"  Successfully predicted (prob > {threshold:.2f} before onset): {hits}")
    print(f"  Out-of-sample ROC-AUC: {auc:.3f}")
    print()
    print("  RESUME BULLET:")
    print(f'  "The 2s10s spread inversion predicted {hits} of the last {total} NBER')
    print(f'  recessions within 12 months, with the logistic model achieving')
    print(f'  ROC-AUC of {auc:.2f} on out-of-sample data from 2010 to present."')
    print("="*65)

    return hits, total, auc
