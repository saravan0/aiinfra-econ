# src/data/lock_schema.py
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
PROCESSED = ROOT / "data" / "processed"
IN = PROCESSED / "features.csv"
OUT = PROCESSED / "features_lean.csv"

if not IN.exists():
    raise SystemExit("Missing processed/features.csv; run feature_engineer first")

df = pd.read_csv(IN, low_memory=False)

# locked lean schema (Option A)
cols = [
 "country","iso3","year",
 "gdp_usd_ln_safe","gdp_per_capita_usd_ln_safe","gdp_growth_pct",
 "exports_usd_ln_safe","imports_usd_ln_safe","fdi_inflow_usd_ln_safe",
 "current_account_balance_usd_ln_safe","total_reserves_usd_ln_safe",
 "inflation_consumer_prices_pct","trade_exposure",
 "voice_accountability_imputed","political_stability_imputed",
 "gov_effectiveness_imputed","reg_quality_imputed","rule_of_law_imputed","control_corruption_imputed",
 "gov_index_zmean","temp_anom_roll"
]

present = [c for c in cols if c in df.columns]
missing = [c for c in cols if c not in df.columns]
print("Present cols:", len(present), "Missing cols:", missing)
out = df[present].copy()
out.to_csv(OUT, index=False)
print("Saved locked lean features ->", OUT)
