# src/data/lock_schema.py
"""
Lock a stable, minimal feature schema for downstream modeling.

Produces:
  - data/processed/features_lean.csv
  - data/processed/features_lean_schema.json

Design Notes (admissions-grade):
  - Uses project-root resolution (not current working directory).
  - Separates critical vs optional variables.
  - Emits schema-level md5 digest for drift detection.
  - Logs missing variables with severity tiers.
  - Ensures deterministic column order.
"""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
IN = PROCESSED / "features.csv"
OUT = PROCESSED / "features_lean.csv"
SCHEMA_OUT = PROCESSED / "features_lean_schema.json"

# Expected schema for modeling
CRITICAL = [
    "country", "iso3", "year",
    "gdp_growth_pct",
    "gov_index_zmean",
]

# Monetary/log-transformed predictors (optional but preferred)
ECONOMIC = [
    "gdp_usd_ln_safe",
    "gdp_per_capita_usd_ln_safe",
    "exports_usd_ln_safe",
    "imports_usd_ln_safe",
    "fdi_inflow_usd_ln_safe",
    "current_account_balance_usd_ln_safe",
    "total_reserves_usd_ln_safe",
]

# Additional preferred predictors
OTHER = [
    "inflation_consumer_prices_pct",
    "trade_exposure",
    "temp_anom_roll",
    "voice_accountability_imputed",
    "political_stability_imputed",
    "gov_effectiveness_imputed",
    "reg_quality_imputed",
    "rule_of_law_imputed",
    "control_corruption_imputed",
]

TARGET_SCHEMA = CRITICAL + ECONOMIC + OTHER

# Load source
if not IN.exists():
    raise SystemExit("Missing features.csv — run feature_engineer first.")

df = pd.read_csv(IN, low_memory=False)

# Validate columns
present = [c for c in TARGET_SCHEMA if c in df.columns]
missing = [c for c in TARGET_SCHEMA if c not in df.columns]

log.info("Total expected columns: %d", len(TARGET_SCHEMA))
log.info("Columns present:        %d", len(present))
log.info("Columns missing:        %d", len(missing))

# HARD FAIL on critical columns missing
missing_critical = [c for c in CRITICAL if c not in df.columns]
if missing_critical:
    log.error("Critical missing columns: %s", missing_critical)
    raise SystemExit(
        f"Cannot lock schema: missing critical columns → {missing_critical}. "
        f"Your pipeline produced insufficient features."
    )

# Soft warning on optional economic or macro variables
missing_econ = [c for c in ECONOMIC if c not in df.columns]
if missing_econ:
    log.warning("Missing economic predictors (non-critical): %s", missing_econ)

missing_other = [c for c in OTHER if c not in df.columns]
if missing_other:
    log.warning("Missing auxiliary predictors (safe to skip): %s", missing_other)

# Extract deterministic subset
lean = df[present].copy()
lean = lean[TARGET_SCHEMA[:len(present)]] if len(present) == len(TARGET_SCHEMA) else lean  # deterministic ordering

OUT.parent.mkdir(parents=True, exist_ok=True)
lean.to_csv(OUT, index=False)
log.info("Saved locked lean features → %s (cols=%d, rows=%d)", OUT, len(lean.columns), len(lean))

# Schema digest (for drift tracking)
schema_text = json.dumps({"columns": present}, sort_keys=True).encode("utf-8")
schema_md5 = hashlib.md5(schema_text).hexdigest()

# Write schema manifest
schema_manifest = {
    "source": str(IN),
    "output": str(OUT),
    "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    "columns_present": present,
    "columns_missing": missing,
    "missing_critical": missing_critical,
    "schema_md5": schema_md5,
    "n_columns": len(present),
    "n_rows": len(lean),
}

SCHEMA_OUT.write_text(json.dumps(schema_manifest, indent=2))
log.info("Saved schema manifest → %s", SCHEMA_OUT)
log.info("Schema MD5: %s", schema_md5)
