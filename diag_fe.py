# diag_fe.py — quick FE diagnostics
import pandas as pd
from pathlib import Path
from src.model import model_defs as mdefs

p = Path("data/processed/features_lean_imputed.csv")
print("Using features file:", p)
df = pd.read_csv(p, low_memory=False)
req = ["iso3", "gdp_growth_pct", "gov_index_zmean", "trade_exposure", "inflation_consumer_prices_pct", "fdi_inflow_usd_ln_safe", "imports_usd_ln_safe", "exports_usd_ln_safe"]
present = [c for c in req if c in df.columns]
print("Requested cols present:", len(present), "/", len(req))
sdf = df[["iso3", "gdp_growth_pct"] + [c for c in req if c not in ("iso3","gdp_growth_pct") and c in df.columns]].dropna()
print("Rows in strict FE subset (dropna on entity+target+predictors):", len(sdf))
sdf_fe = mdefs.country_fixed_effects(sdf, country_col="iso3", drop_first=True, prefix="FE")
fe_cols = [c for c in sdf_fe.columns if c.startswith("FE_")]
print("FE dummies detected:", len(fe_cols))
# sample dtypes
print("\nNon-int FE dtypes (sample):")
print(sdf_fe[fe_cols].dtypes.head(40))
