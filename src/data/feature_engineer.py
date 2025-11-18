# src/data/feature_engineer.py
"""
Feature engineering for the wgi_econ_master dataset.

Produces:
 - data/processed/features.csv

Design choices (admissions-relevant):
 - Conservative transforms only (log1p for positive monetary variables).
 - Explicit handling of per-country rolling/statistics (temp anomaly rolling mean).
 - Conservative imputations: country-mean imputation for WGI pillars only when missingness is moderate (<40%).
 - Adds lags for key predictors to support modeling and interpretation.
 - Records provenance (md5 + sources.yaml) for the output artifact (if registry available).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Project-root-aware paths
ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

MASTER_IN = INTERIM / "wgi_econ_master.csv"
DEFAULT_OUT = PROCESSED / "features.csv"

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Try to import record_artifact if available; degrade gracefully.
try:
    from src.utils.data_registry import record_artifact  # type: ignore
except Exception:
    record_artifact = None  # type: ignore


def _safe_log1p_series(s: pd.Series) -> pd.Series:
    """Apply log1p for strictly positive values; NaN otherwise."""
    snum = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index)
    mask = snum > 0
    out.loc[mask] = np.log1p(snum.loc[mask])
    return out


def _zscore_series(s: pd.Series) -> pd.Series:
    """Compute z-score with population std (ddof=0). Preserve NaNs."""
    s_num = pd.to_numeric(s, errors="coerce")
    mu = s_num.mean(skipna=True)
    sigma = s_num.std(ddof=0, skipna=True)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series([np.nan] * len(s_num), index=s_num.index)
    return (s_num - mu) / sigma


def _ensure_index_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by iso3, year if present to make rolling/lag deterministic."""
    if {"iso3", "year"}.issubset(df.columns):
        return df.sort_values(["iso3", "year"]).reset_index(drop=True)
    return df.reset_index(drop=True)


def build_features(master_path: Optional[Path] = None, out_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load master dataset, compute engineered features, write CSV, and record provenance (if available).

    Returns the engineered DataFrame.
    """
    master_path = Path(master_path) if master_path else MASTER_IN
    out_path = Path(out_path) if out_path else DEFAULT_OUT

    if not master_path.exists():
        LOG.error("Master file missing: %s", master_path)
        raise SystemExit(1)

    LOG.info("Loading master: %s", master_path)
    df = pd.read_csv(master_path, low_memory=False)

    # Make index order deterministic for rolling/lag operations
    df = _ensure_index_order(df)

        # 1) Trade total + exposure (canonical, robust)
    #    - trade_total = exports + imports (zero-fill for sum)
    #    - trade_total_ln and trade_total_ln_safe for downstream consistency
    #    - trade_exposure = trade_total / gdp_usd (NaN where GDP missing)
    #    - conservative fallback: if both exports & imports are zero -> exposure = 0
    if {"exports_usd", "imports_usd"}.issubset(df.columns):
        LOG.info("Creating trade_total and trade_exposure features")
        # ensure numeric
        df["exports_usd"] = pd.to_numeric(df["exports_usd"], errors="coerce")
        df["imports_usd"] = pd.to_numeric(df["imports_usd"], errors="coerce")

        # trade_total: sum; use zero for missing values in the sum so volume is preserved
        df["trade_total"] = df["exports_usd"].fillna(0) + df["imports_usd"].fillna(0)

        # safe log names to match other log features (picked up by later transform step)
        df["trade_total_ln"] = _safe_log1p_series(df["trade_total"])
        df["trade_total_ln_safe"] = df["trade_total_ln"]

        # compute normalized exposure if GDP present
        if "gdp_usd" in df.columns:
            df["gdp_usd"] = pd.to_numeric(df["gdp_usd"], errors="coerce")
            # normal exposure (NaN if gdp missing)
            df["trade_exposure"] = (df["trade_total"] / df["gdp_usd"]).where(
                (df["trade_total"].notna()) & (df["gdp_usd"].notna()),
                np.nan,
            )
            # conservative fallback: if both exports & imports zero -> exposure 0
            df["trade_exposure"] = df["trade_exposure"].where(
                df["trade_exposure"].notna(),
                np.where(
                    (df.get("exports_usd", 0).fillna(0) == 0) & (df.get("imports_usd", 0).fillna(0) == 0),
                    0,
                    np.nan,
                ),
            )

    # 2) Log transforms (safe log1p on positive economics variables)
    to_log = [
        "gdp_usd", "gdp_per_capita_usd", "fdi_inflow_usd", "total_reserves_usd",
        "exports_usd", "imports_usd", "external_debt_usd", "current_account_balance_usd"
    ]
    for c in to_log:
        if c in df.columns:
            df[c + "_ln"] = _safe_log1p_series(df[c])
            # Also create _ln_safe alias (explicitly named safe logs used in modeling)
            df[c + "_ln_safe"] = df[c + "_ln"]
            LOG.debug("Created log features: %s, %s", c + "_ln", c + "_ln_safe")

    # 3) Governance composite: zscore pillars then mean
    wgi_cols: List[str] = [
        c for c in [
            "voice_accountability", "political_stability", "gov_effectiveness",
            "reg_quality", "rule_of_law", "control_corruption"
        ] if c in df.columns
    ]
    if wgi_cols:
        LOG.info(
            "Computing governance z-scores and composite (gov_index_zmean) for %d pillars",
            len(wgi_cols),
        )
        zcols = []
        for c in wgi_cols:
            zc = c + "_z"
            df[zc] = _zscore_series(df[c])
            zcols.append(zc)
        df["gov_index_zmean"] = df[zcols].mean(axis=1)

    # 4) Rolling temperature anomaly (3-year country-level mean) if temp_anom_degC present
    if "temp_anom_roll" not in df.columns and "temp_anom_degC" in df.columns:
        LOG.info("Computing 3-year rolling temperature anomaly per country")
        df = df.sort_values(["iso3", "year"])
        df["temp_anom_roll"] = (
            df.groupby("iso3")["temp_anom_degC"]
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        )

    # 5) Lag features (1-year) for selected variables
    df = df.sort_values(["iso3", "year"])
    for c in ["gdp_usd", "gov_index_zmean", "gdp_per_capita_usd"]:
        if c in df.columns:
            df[c + "_lag1"] = df.groupby("iso3")[c].shift(1)

    # 6) Conservative imputations
    # - trade_exposure -> if exports&imports both zero -> set 0, else keep NaN
    if "trade_exposure" in df.columns:
        df["trade_exposure"] = df["trade_exposure"].where(
            df["trade_exposure"].notna(),
            np.where((df.get("exports_usd", 0).fillna(0) == 0) & (df.get("imports_usd", 0).fillna(0) == 0), 0, np.nan),
        )

    # - WGI pillars: country-mean imputation only if missing_fraction < 0.4
    for c in wgi_cols:
        miss_frac = df[c].isna().mean()
        if miss_frac < 0.4:
            col_imputed = c + "_imputed"
            LOG.info("Imputing %s by country mean (missing_frac=%.3f)", c, miss_frac)
            df[col_imputed] = df.groupby("iso3")[c].transform(lambda s: s.fillna(s.mean()))

    # Final housekeeping: write file and record provenance if possible
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Writing engineered features to %s", out_path)
    df.to_csv(out_path, index=False)

    md5 = None
    if record_artifact is not None:
        try:
            # record_artifact typically expects (path, canonical_id=...) signature
            md5 = record_artifact(str(out_path), canonical_id="features")
        except Exception as e:
            LOG.warning("Provenance recording failed: %s", e)
            md5 = None
    else:
        LOG.debug("record_artifact not available; skipping provenance recording")

    if md5:
        LOG.info("Saved engineered features -> %s (rows=%s, cols=%s) — md5=%s", out_path, f"{len(df):,}", len(df.columns), md5)
    else:
        LOG.info("Saved engineered features -> %s (rows=%s, cols=%s)", out_path, f"{len(df):,}", len(df.columns))

    return df


def _cli():
    p = argparse.ArgumentParser(description="Feature engineering for wgi_econ_master")
    p.add_argument("--in", dest="infile", help="Master CSV input path (default data/interim/wgi_econ_master.csv)")
    p.add_argument("--out", dest="outfile", help="Output features CSV path (default data/processed/features.csv)")
    args = p.parse_args()
    infile = Path(args.infile) if args.infile else None
    outfile = Path(args.outfile) if args.outfile else None
    build_features(master_path=infile, out_path=outfile)


if __name__ == "__main__":
    _cli()
