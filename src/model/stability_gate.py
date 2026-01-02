"""
Enforce minimum sample integrity conditions before downstream modeling.

Inputs:
 - reports/predictor_stability.csv

Enforced conditions:
 - Minimum required sample size for the active predictor set.
 - Minimum per-predictor coverage threshold.

Behavior:
 - Terminates the pipeline with a diagnostic message if conditions are not met.
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
