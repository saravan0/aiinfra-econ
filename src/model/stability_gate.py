# src/model/stability_gate.py
"""
Stability Gate — Sample Integrity Check for the Modeling Pipeline

This module enforces minimum sample quality before any downstream
econometric or ML modeling steps are executed. It acts as a safeguard
against silent dataset degradation, ensuring that the modeling stage
is only reached when the predictor set remains sufficiently complete
and statistically usable.

Inputs:
  • reports/predictor_stability.csv

Enforced Conditions:
  • Minimum required sample size for the combined baseline + extras
    predictor set (n_core_plus_extras ≥ MIN_REQUIRED).
  • Minimum per-predictor coverage threshold (pct ≥ MIN_COVERAGE_PCT).

If any condition fails, the pipeline terminates with a clear diagnostic
message. This prevents weak or unstable models from being fitted due to
insufficient coverage or collapsed sample size — a critical requirement
for reproducibility, robustness, and research-grade reliability.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
import sys

# thresholds — tuned for *your dataset*
MIN_REQUIRED = 2500       # hard minimum for full baseline+extras
MIN_COVERAGE_PCT = 60.0   # minimum allowed per-predictor coverage %

REPORTS = Path("reports")
STAB_CSV = REPORTS / "predictor_stability.csv"

def fail(msg: str):
    print(f"\n❌ Stability Gate FAILED:\n{msg}\n")
    raise SystemExit(1)

def main():
    if not STAB_CSV.exists():
        fail(f"Missing {STAB_CSV}. Run predictor_stability first.")

    df = pd.read_csv(STAB_CSV)

    # --- check predictor coverage ---
    low_cov = df[df["pct"] < MIN_COVERAGE_PCT]
    if not low_cov.empty:
        fail(
            "Some predictors have coverage below threshold "
            f"({MIN_COVERAGE_PCT}%):\n" +
            low_cov.to_string(index=False)
        )

    # --- check usable sample size ---
    try:
        df2 = df.copy()
        if "n_nonnull_imputed" in df.columns:
            approx = df[
                df["predictor"].isin([
                    "fdi_inflow_usd_ln_safe",
                    "imports_usd_ln_safe",
                    "exports_usd_ln_safe"
                ])
            ]["n_nonnull_imputed"].min()
            n_core_plus_extras = int(approx)
        else:
            fail("predictor_stability.csv missing imputed columns.")
    except Exception:
        fail("Could not extract n_core_plus_extras from stability CSV.")

    if n_core_plus_extras < MIN_REQUIRED:
        fail(
            f"Effective sample collapsed: got {n_core_plus_extras}, "
            f"required >= {MIN_REQUIRED}."
        )

    print("\n✅ Stability Gate PASSED — sample size & coverage OK.\n")

if __name__ == "__main__":
    main()
