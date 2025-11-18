# src/model/stability_gate.py
"""
Stability Gate — prevents the modelling pipeline from running
if the usable sample collapses below guaranteed thresholds.

Reads:
  reports/predictor_stability.csv

Checks:
  - n_core_plus_extras >= MIN_REQUIRED
  - pct of each predictor >= MIN_COVERAGE_PCT

Fails with SystemExit if violated.

This protects your One Piece pipeline from silent degradation.
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
    # These come from predictor_stability.md numbers – we infer them from CSV
    try:
        # Infer core+extras count from CSV by combining counts
        # (alternative: read from predictor_stability.md)
        df2 = df.copy()
        # sample size is in auxiliary file; fallback: require CSV to have imputed columns
        if "n_nonnull_imputed" in df.columns:
            # approximate by minimum across extras + core
            approx = df[df["predictor"].isin([
                "fdi_inflow_usd_ln_safe",
                "imports_usd_ln_safe",
                "exports_usd_ln_safe"
            ])]["n_nonnull_imputed"].min()
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
