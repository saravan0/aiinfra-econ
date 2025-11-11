# src/data/safe_transforms_and_vif.py
"""
Safe log transforms, monetary scale diagnostics, correlation export, VIF export.
Outputs (in data/interim):
 - monetary_scale_check.csv
 - top_correlations.csv
 - vif.csv
Requires: pandas, numpy, statsmodels
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
INTERIM.mkdir(parents=True, exist_ok=True)

FEATURES = PROCESSED / "features.csv"
MASTER = INTERIM / "wgi_econ_master.csv"

# load best available (prefer processed features)
if FEATURES.exists():
    df = pd.read_csv(FEATURES, low_memory=False)
    print("Loaded processed features:", FEATURES)
elif MASTER.exists():
    df = pd.read_csv(MASTER, low_memory=False)
    print("Loaded master:", MASTER)
else:
    raise SystemExit("Missing features.csv or wgi_econ_master.csv")

# safe log helper
def safe_log(series):
    # return log1p for strictly positive values; NaN otherwise
    s = pd.to_numeric(series, errors='coerce')
    out = pd.Series(np.nan, index=s.index)
    mask = s > 0
    out.loc[mask] = np.log1p(s.loc[mask])
    return out

# monetary columns to inspect
mon_cols = [c for c in ['gdp_usd','exports_usd','imports_usd','fdi_inflow_usd','total_reserves_usd','current_account_balance_usd','external_debt_usd'] if c in df.columns]

# scale diagnostics
rows = []
for c in mon_cols:
    s = pd.to_numeric(df[c], errors='coerce').dropna()
    if len(s)==0:
        rows.append((c,0, np.nan, np.nan, np.nan))
        continue
    med = s.median()
    mx = s.max()
    mn = s.min()
    ratio = mx/(med if med!=0 else 1)
    rows.append((c, len(s), med, mx, ratio))
scale_df = pd.DataFrame(rows, columns=['column','n_nonnull','median','max','max_over_median'])
scale_df.to_csv(INTERIM / "monetary_scale_check.csv", index=False)
print("monetary scale check -> data/interim/monetary_scale_check.csv")
print(scale_df.to_string(index=False))

# create safe log columns and save a small preview (do not overwrite features)
for c in mon_cols:
    df[c + "_ln_safe"] = safe_log(df[c])

# correlation: pick numeric columns (drop identifiers)
num = df.select_dtypes(include=[np.number]).copy()
if 'year' in num.columns: num = num.drop(columns=['year'])
# compute correlation matrix and export top absolute correlations
corr = num.corr().abs()
pairs = []
cols = corr.columns.tolist()
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        pairs.append((cols[i], cols[j], corr.iloc[i,j]))
pairs_df = pd.DataFrame(pairs, columns=['x','y','abs_corr']).sort_values('abs_corr', ascending=False)
pairs_df.to_csv(INTERIM / "top_correlations.csv", index=False)
print("Saved top correlations -> data/interim/top_correlations.csv")
print("Top 10 correlations:")
print(pairs_df.head(10).to_string(index=False))

# VIF calculation (requires statsmodels)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # choose features for VIF: numeric columns with reasonable non-NA
    vif_cols = [c for c in num.columns if num[c].notna().mean() > 0.5]  # at least 50% non-missing
    X = num[vif_cols].dropna()
    if X.shape[0] == 0:
        print("No rows with full data for VIF calculation. Consider imputing first.")
    else:
        vif_list = []
        for i, col in enumerate(X.columns):
            try:
                v = variance_inflation_factor(X.values, i)
            except Exception as e:
                v = np.nan
            vif_list.append((col, v))
        vif_df = pd.DataFrame(vif_list, columns=['feature','vif']).sort_values('vif', ascending=False)
        vif_df.to_csv(INTERIM / "vif.csv", index=False)
        print("Saved VIF -> data/interim/vif.csv")
        print(vif_df.head(20).to_string(index=False))
except ImportError:
    print("statsmodels not installed. To compute VIF run: pip install statsmodels")
    # still save correlations and scale results
