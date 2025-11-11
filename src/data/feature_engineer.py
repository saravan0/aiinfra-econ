# src/data/feature_engineer.py
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

MASTER_IN = INTERIM / "wgi_econ_master.csv"
OUT = PROCESSED / "features.csv"

df = pd.read_csv(MASTER_IN, low_memory=False)

# canonical columns expected
# compute trade_exposure if not present and exports/imports/gdp available
if 'trade_exposure' not in df.columns and {'exports_usd','imports_usd','gdp_usd'}.issubset(df.columns):
    df['trade_exposure'] = (df['exports_usd'].replace({np.nan:0}) + df['imports_usd'].replace({np.nan:0})) / df['gdp_usd']

# per-capita conversions (if gdp_per_capita_usd missing but gdp_usd and population exist, not implemented here)
# log transforms (use safe log1p on positive economics variables)
to_log = ['gdp_usd','gdp_per_capita_usd','fdi_inflow_usd','total_reserves_usd','exports_usd','imports_usd','external_debt_usd','current_account_balance_usd']
for c in to_log:
    if c in df.columns:
        df[c + '_ln'] = np.where(df[c].notna() & (df[c]>0), np.log1p(df[c]), np.nan)

# gov composite (simple mean of available WGI pillars, zscore each pillar first)
wgi_cols = [c for c in ['voice_accountability','political_stability','gov_effectiveness','reg_quality','rule_of_law','control_corruption'] if c in df.columns]
if wgi_cols:
    # zscore per pillar across full dataset (leave NaNs)
    from scipy import stats
    for c in wgi_cols:
        col_z = c + '_z'
        df[col_z] = (df[c] - df[c].mean()) / (df[c].std(ddof=0))
    zcols = [c + '_z' for c in wgi_cols]
    df['gov_index_zmean'] = df[zcols].mean(axis=1)

# rolling temperature anomaly: if a temp_anom_roll exists keep it; else if temp_anom exists compute 3-year rolling mean per country
if 'temp_anom_roll' not in df.columns and 'temp_anom_degC' in df.columns:
    df = df.sort_values(['iso3','year'])
    df['temp_anom_roll'] = df.groupby('iso3')['temp_anom_degC'].transform(lambda s: s.rolling(window=3, min_periods=1).mean())

# create lag features for gdp and gov_index (1-year lag)
df = df.sort_values(['iso3','year'])
for c in ['gdp_usd','gov_index_zmean','gdp_per_capita_usd']:
    if c in df.columns:
        df[c + '_lag1'] = df.groupby('iso3')[c].shift(1)

# simple imputations (small, conservative):
# - fill missing trade_exposure with 0 if exports+imports both 0 otherwise leave NaN
if 'trade_exposure' in df.columns:
    df['trade_exposure'] = df['trade_exposure'].where(df['trade_exposure'].notna(), 
                                                     np.where((df['exports_usd'].fillna(0)==0) & (df['imports_usd'].fillna(0)==0), 0, np.nan))

# - fill small-missingness WGI pillars by country mean (if missing_fraction < 0.4)
pillars = wgi_cols
for c in pillars:
    miss_frac = df[c].isna().mean()
    if miss_frac < 0.4:
        df[c + '_imputed'] = df.groupby('iso3')[c].transform(lambda s: s.fillna(s.mean()))

# save features
df.to_csv(OUT, index=False)
print("Saved engineered features ->", OUT)
print("Rows:", len(df), "Cols:", len(df.columns))
