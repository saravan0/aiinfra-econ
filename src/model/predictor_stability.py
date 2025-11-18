# src/model/predictor_stability.py
"""
Predictor stability & sample-size diagnostics.

Consumes:
 - data/processed/features_lean.csv
 - data/processed/features_lean_imputed.csv  (optional)

Produces:
 - reports/predictor_stability.csv
 - reports/predictor_stability.md
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path.cwd()
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

LEAN = PROCESSED / "features_lean.csv"
IMPUTED = PROCESSED / "features_lean_imputed.csv"

OUT_CSV = REPORTS / "predictor_stability.csv"
OUT_MD = REPORTS / "predictor_stability.md"

# baseline predictor sets (adjust to your config)
CORE = ["gov_index_zmean", "trade_exposure", "inflation_consumer_prices_pct"]
EXTRAS = ["fdi_inflow_usd_ln_safe", "imports_usd_ln_safe", "exports_usd_ln_safe"]

def load(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)

def coverage_table(df: pd.DataFrame, predictors):
    n = len(df)
    rows = []
    for p in predictors:
        present = p in df.columns
        n_nonnull = int(df[p].notna().sum()) if present else 0
        pct = round(100 * (n_nonnull / n), 2) if n > 0 else 0.0
        rows.append({"predictor": p, "present": present, "n_nonnull": n_nonnull, "pct": pct})
    return pd.DataFrame(rows)

def sample_counts(df: pd.DataFrame):
    out = {}
    out["total_rows"] = len(df)
    out["n_target_only"] = int(df[df["gdp_growth_pct"].notna()].shape[0]) if "gdp_growth_pct" in df.columns else 0
    # be defensive: ensure CORE exists in df before passing to dropna
    core_present = [c for c in CORE if c in df.columns]
    out["n_core"] = int(df[core_present].dropna().shape[0]) if core_present else 0
    extras_present = [c for c in EXTRAS if c in df.columns]
    out["n_core_plus_extras"] = int(df[core_present + extras_present].dropna().shape[0]) if core_present and extras_present else 0
    # core + each extra
    for e in EXTRAS:
        if e in df.columns and core_present:
            out[f"core_plus_{e}"] = int(df[core_present + [e]].dropna().shape[0])
        else:
            out[f"core_plus_{e}"] = 0
    return out

def md_report(baseline_df, imputed_df, cov_table_raw, sample_before, sample_after):
    lines = []
    lines.append(f"# Predictor stability report\n\nGenerated: {datetime.utcnow().isoformat()}Z\n")
    lines.append(f"Input lean file: {LEAN}\n")
    lines.append(f"Total rows (original): {sample_before['total_rows']}\n\n")
    lines.append("## Coverage (original)\n")
    lines.append(cov_table_raw.to_markdown(index=False))
    lines.append("\n\n## Sample counts (original)\n")
    for k, v in sample_before.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n\n## After imputation (if applied)\n")
    for k, v in sample_after.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n\n### Quick recommendation\n")
    lines.append("- Use the imputed file as model input if `n_core_plus_extras` increases substantially (>=10% increase).\n")
    return "\n".join(lines)

def main():
    base = load(LEAN)
    if base is None:
        raise SystemExit(f"Missing {LEAN}")
    # CORRECTED: avoid ambiguous truth-testing of a DataFrame
    imputed = load(IMPUTED)  # returns DataFrame or None
    predictors = CORE + EXTRAS
    cov_base = coverage_table(base, predictors)
    sample_before = sample_counts(base)
    sample_after = sample_before
    if imputed is not None:
        cov_imputed = coverage_table(imputed, predictors)
        sample_after = sample_counts(imputed)
    else:
        cov_imputed = None

    # write CSV summary (base + optional imputed side-by-side)
    df_out = cov_base.copy()
    if cov_imputed is not None:
        df_out = df_out.merge(
            cov_imputed[["predictor", "n_nonnull", "pct"]]
            .rename(columns={"n_nonnull": "n_nonnull_imputed", "pct": "pct_imputed"}),
            on="predictor", how="left"
        )
    df_out.to_csv(OUT_CSV, index=False)

    # write markdown report
    report_md = md_report(base, imputed, cov_base, sample_before, sample_after)
    OUT_MD.write_text(report_md, encoding="utf8")
    print("Wrote predictor stability csv ->", OUT_CSV)
    print("Wrote predictor stability md  ->", OUT_MD)

if __name__ == "__main__":
    main()
