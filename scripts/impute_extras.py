# scripts/impute_extras.py
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()
IN = ROOT / "data" / "processed" / "features_lean.csv"
OUT = ROOT / "data" / "processed" / "features_imputed_country_mean.csv"

df = pd.read_csv(IN, low_memory=False)

extras = ["fdi_inflow_usd_ln_safe","imports_usd_ln_safe","exports_usd_ln_safe"]
# For each extra, impute by country mean (only where that country's mean exists)
for c in extras:
    if c in df.columns:
        df[c + "_imputed"] = df.groupby("iso3")[c].transform(lambda s: s.fillna(s.mean()))
        # fallback to original name if you prefer overwriting:
        # df[c] = df[c].fillna(df[c + "_imputed"])

# write
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print("Wrote imputed file:", OUT)
print("Preview non-null counts:")
print(df[[c for c in extras if c in df.columns]].notna().sum())
