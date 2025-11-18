# src/data/impute_predictors.py
"""
Controlled imputation layer for predictors.

Rules:
 - coverage > 0.70  -> country-mean imputation for missing values
 - 0.50 <= coverage <= 0.70 -> global-mean imputation
 - coverage < 0.50 -> mark as 'drop' (do not impute automatically)

Outputs:
 - data/processed/features_lean_imputed.csv  (imputed file)
 - reports/imputation_manifest.json
 - reports/imputation_summary.csv
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict

ROOT = Path.cwd()
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

IN = PROCESSED / "features_lean.csv"
OUT = PROCESSED / "features_lean_imputed.csv"
MANIFEST = REPORTS / "imputation_manifest.json"
SUMMARY_CSV = REPORTS / "imputation_summary.csv"

# Configure the predictors we consider for imputation (typically extras)
# You can edit this list or load from config/predictors_core.yml
TARGET_PREDICTORS = [
    "fdi_inflow_usd_ln_safe",
    "imports_usd_ln_safe",
    "exports_usd_ln_safe",
    "current_account_balance_usd_ln_safe",
    "total_reserves_usd_ln_safe",
    "gdp_usd_ln_safe",
    "gdp_per_capita_usd_ln_safe"
]

# thresholds
COUNTRY_MEAN_THRESH = 0.70
GLOBAL_MEAN_THRESH = 0.50

def load():
    if not IN.exists():
        raise SystemExit(f"Missing input: {IN}")
    return pd.read_csv(IN, low_memory=False)

def coverage_stats(df: pd.DataFrame, predictors):
    stats = []
    n_total = len(df)
    for p in predictors:
        present = p in df.columns
        n_nonnull = int(df[p].notna().sum()) if present else 0
        pct = float(n_nonnull / n_total) if n_total > 0 else 0.0
        stats.append({"predictor": p, "present": present, "n_nonnull": n_nonnull, "pct": round(pct, 4)})
    return stats

def impute(df: pd.DataFrame, predictors) -> (pd.DataFrame, Dict):
    manifest = {"decisions": {}, "n_rows": len(df)}
    df_out = df.copy()
    for p in predictors:
        if p not in df_out.columns:
            manifest["decisions"][p] = {"action": "missing_column"}
            continue
        n_nonnull = int(df_out[p].notna().sum())
        pct = n_nonnull / len(df_out) if len(df_out) > 0 else 0.0
        if pct >= COUNTRY_MEAN_THRESH:
            # country mean (group by iso3)
            df_out[p] = df_out.groupby("iso3")[p].transform(lambda s: s.fillna(s.mean()))
            action = "country_mean_impute"
        elif pct >= GLOBAL_MEAN_THRESH:
            # global mean
            gm = df_out[p].mean(skipna=True)
            df_out[p] = df_out[p].fillna(gm)
            action = "global_mean_impute"
        else:
            # do not impute automatically
            action = "drop_candidate"
        # record stats
        manifest["decisions"][p] = {"n_nonnull": n_nonnull, "pct": round(pct, 4), "action": action}
    return df_out, manifest

def write_outputs(df_imputed: pd.DataFrame, manifest: Dict, stats):
    df_imputed.to_csv(OUT, index=False)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(stats).to_csv(SUMMARY_CSV, index=False)
    print("Wrote imputed features ->", OUT)
    print("Wrote imputation manifest ->", MANIFEST)
    print("Wrote imputation summary ->", SUMMARY_CSV)

def main():
    df = load()
    stats = coverage_stats(df, TARGET_PREDICTORS)
    df_imputed, manifest = impute(df, [s["predictor"] for s in stats])
    # include a timestamp
    from datetime import datetime
    manifest["generated_at"] = datetime.utcnow().isoformat() + "Z"
    write_outputs(df_imputed, manifest, stats)

if __name__ == "__main__":
    main()
