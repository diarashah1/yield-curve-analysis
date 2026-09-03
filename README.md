# Yield Curve Dynamics & Recession Prediction Model

A Python-based analysis of 30+ years of US Treasury yield data, decomposing
yield curve shape into interpretable factors and building a recession prediction
model from spread inversions.

**Built by Diara Shah**

---

## What this project does

1. **Pulls** daily US Treasury yield data (1990–present) from FRED (Federal Reserve Economic Data)
2. **Decomposes** the yield curve using Nelson-Siegel factor modeling into level, slope, and curvature
3. **Predicts** recession probability using logistic regression on yield spread inversions
4. **Backtests** how equities, long bonds, high yield, and investment grade credit behave across yield curve regimes
5. **Synthesizes** findings into a macro memo applying the framework to current market conditions

---

## Key findings

*(Fill these in after running the model — these are placeholders)*

- The 2s10s spread inverted before **X of Y** NBER recessions in the dataset
- Logistic regression achieved **ROC-AUC of [Z]** on out-of-sample data (post-2010)
- TLT (long duration bonds) returned **[X]%** annualized in inverted curve regimes vs **[Y]%** in steep regimes
- HYG (high yield) showed the largest drawdown difference across regimes: **[X]%** inverted vs **[Y]%** steep

---

## How to reproduce this

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/yield-curve-analysis.git
cd yield-curve-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free FRED API key
- Go to https://fred.stlouisfed.org/docs/api/api_key.html
- Register for a free account (30 seconds)
- Copy your API key
- Open `src/config.py` and paste it into `FRED_API_KEY = "..."`

### 4. Run the analysis in order
```bash
# Part 1: Data preparation and sanity check chart
python notebooks/01_data_preparation.py

# Part 2: Nelson-Siegel fitting
python notebooks/02_nelson_siegel.py

# Part 3: Recession prediction model
python notebooks/03_recession_model.py

# Part 4: Cross-asset regime analysis
python notebooks/04_cross_asset_regimes.py
```

Data is cached to `data/` on first run — subsequent runs are instant.

---

## Data sources

| Series | Source | Description |
|--------|--------|-------------|
| DGS3MO, DGS2, DGS5, DGS10, DGS30 | FRED | Daily US Treasury constant maturity yields |
| USREC | FRED / NBER | Monthly recession indicator (1 = recession) |
| SPY, TLT, HYG, LQD | Yahoo Finance (yfinance) | ETF daily returns for regime analysis |

All free. No Bloomberg. Fully reproducible on any machine.

---

## Repository structure

```
yield-curve-analysis/
├── README.md
├── requirements.txt
├── data/                    # CSV files (auto-generated, not tracked in git)
│   ├── treasury_yields.csv
│   ├── nber_recessions.csv
│   ├── asset_returns.csv
│   └── combined_data.csv
├── notebooks/               # Analysis scripts (run these in order)
│   ├── 01_data_preparation.py
│   ├── 02_nelson_siegel.py
│   ├── 03_recession_model.py
│   └── 04_cross_asset_regimes.py
├── src/                     # Core modules (imported by notebooks)
│   ├── config.py            # All settings and constants
│   ├── data_loader.py       # FRED data ingestion and cleaning
│   ├── nelson_siegel.py     # NS curve fitting (Part 2)
│   ├── recession_model.py   # Logistic regression (Part 3)
│   └── regime_analysis.py   # Cross-asset analysis (Part 4)
├── outputs/
│   ├── figures/             # All saved charts
│   └── tables/              # Summary tables
└── memo/
    └── yield_curve_memo.pdf # Bridgewater writing sample
```

---

## Known limitations

- **Lookahead bias in target construction**: The 12-month forward recession label uses future data by definition. This is intentional for historical study of predictive power, not a live trading signal.
- **NBER recession labels are lagging**: NBER declares recessions months after they begin. The label dates are approximate.
- **2020 COVID recession**: This recession was caused by an external shock (pandemic), not the typical credit contraction that inversion predicts. The model's miss here is expected and documented.
- **Stationarity assumption**: The logistic model assumes relationships between spreads and recession risk are stable over time. The 1980s high-rate environment differs structurally from the 2010s low-rate environment.
