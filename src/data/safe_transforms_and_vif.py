"""
Safe transforms, monetary-scale diagnostics, correlations, and VIF exports.

Outputs (in data/interim):
  - monetary_scale_check.csv
  - top_correlations.csv
  - vif.csv (if statsmodels available)
  - safe_transforms_manifest.json (summary & md5s)

Design / Admissions notes:
 - Conservative numeric transforms only (log1p for strictly positive values)
 - Correlations exported as absolute correlations (top pairs)
 - VIF computed only when statsmodels available and sufficient full-case rows exist
 - Each artifact recorded via src.utils.data_registry.record_artifact (provenance)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# try to import record_artifact (optional). If import fails, fall back to no-op.
try:
    from src.utils.data_registry import record_artifact
except Exception:
    def record_artifact(*args, **kwargs):
        return None  # type: ignore

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
INTERIM.mkdir(parents=True, exist_ok=True)

FEATURES = PROCESSED / "features.csv"
MASTER = INTERIM / "wgi_econ_master.csv"

MONETARY_OUT = INTERIM / "monetary_scale_check.csv"
CORR_OUT = INTERIM / "top_correlations.csv"
VIF_OUT = INTERIM / "vif.csv"
MANIFEST_OUT = INTERIM / "safe_transforms_manifest.json"


def _load_source() -> pd.DataFrame:
    """Load processed features if available, otherwise fall back to master."""
    if FEATURES.exists():
        LOG.info("Loaded processed features: %s", FEATURES)
        return pd.read_csv(FEATURES, low_memory=False)
    if MASTER.exists():
        LOG.info("Loaded master dataset: %s", MASTER)
        return pd.read_csv(MASTER, low_memory=False)
    raise SystemExit("Missing processed/features.csv and wgi_econ_master.csv — run upstream steps.")


def safe_log(series: pd.Series) -> pd.Series:
    """Return log1p for strictly positive values; NaN otherwise."""
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index)
    mask = s > 0
    out.loc[mask] = np.log1p(s.loc[mask])
    return out


def monetary_scale_check(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Compute median / max / max_over_median diagnostics for monetary columns."""
    rows: List[Tuple[str, int, float, float, float]] = []
    for c in cols:
        s = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce").dropna()
        if len(s) == 0:
            rows.append((c, 0, float("nan"), float("nan"), float("nan")))
            continue
        med = float(s.median())
        mx = float(s.max())
        mn = float(s.min())
        ratio = mx / (med if med != 0 else 1)
        rows.append((c, len(s), med, mx, ratio))
    scale_df = pd.DataFrame(rows, columns=["column", "n_nonnull", "median", "max", "max_over_median"])
    scale_df.to_csv(MONETARY_OUT, index=False)
    try:
        md5 = record_artifact(MONETARY_OUT, canonical_id="monetary_scale_check")
    except Exception as e:
        LOG.warning("record_artifact failed for monetary_scale_check: %s", e)
        md5 = None
    LOG.info("monetary scale check -> %s (md5=%s)", MONETARY_OUT, md5)
    return scale_df


def create_safe_logs(df: pd.DataFrame, mon_cols: List[str]) -> pd.DataFrame:
    """Add *_ln_safe columns for monetary columns.

    Rules:
    - If a *_ln column already exists, do not create a duplicate *_ln_safe.
    - If 'trade_total' exists, skip creating exports/imports _ln_safe to avoid multicollinearity.
    """
    # if we already created trade_total upstream, prefer aggregated measure
    prefer_trade_total = "trade_total" in df.columns

    for c in mon_cols:
        # skip per-side logs when we prefer trade_total
        if prefer_trade_total and c in ("exports_usd", "imports_usd"):
            LOG.info("Skipping creation of %s_ln_safe because trade_total exists", c)
            # still create the plain _ln (if needed) but avoid _ln_safe for components
            ln_col = c + "_ln"
            if ln_col not in df.columns:
                df[ln_col] = safe_log(df.get(c))
            continue

        ln_col = c + "_ln"
        ln_safe_col = c + "_ln_safe"

        # create _ln if missing
        if ln_col not in df.columns:
            df[ln_col] = safe_log(df.get(c))

        # only create *_ln_safe if it doesn't already exist and it's not a duplicate of _ln
        if ln_safe_col not in df.columns:
            # copy _ln into _ln_safe (this keeps naming consistent for downstream code)
            df[ln_safe_col] = df[ln_col]
        else:
            LOG.debug("'%s' already exists; not overwriting", ln_safe_col)

    return df


def export_top_correlations(df: pd.DataFrame, top_k: int = 2000) -> pd.DataFrame:
    """Compute absolute pairwise correlations and export sorted pairs."""
    num = df.select_dtypes(include=[np.number]).copy()
    if "year" in num.columns:
        num = num.drop(columns=["year"], errors="ignore")
    if num.shape[1] < 2:
        LOG.info("Not enough numeric columns for correlation analysis.")
        pairs_df = pd.DataFrame(columns=["x", "y", "abs_corr"])
        pairs_df.to_csv(CORR_OUT, index=False)
        try:
            record_artifact(CORR_OUT, canonical_id="top_correlations")
        except Exception:
            pass
        return pairs_df

    corr = num.corr().abs()
    pairs: List[Tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
    pairs_df = pd.DataFrame(pairs, columns=["x", "y", "abs_corr"]).sort_values("abs_corr", ascending=False)
    # limit output (but still deterministic)
    pairs_df = pairs_df.head(top_k)
    pairs_df.to_csv(CORR_OUT, index=False)
    try:
        md5 = record_artifact(CORR_OUT, canonical_id="top_correlations")
    except Exception as e:
        LOG.warning("record_artifact failed for top_correlations: %s", e)
        md5 = None
    LOG.info("Saved top correlations -> %s (md5=%s)", CORR_OUT, md5)
    return pairs_df


def _drop_duplicate_logs_and_gov_pillars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate *_ln columns (keep *_ln_safe), and drop raw governance pillar columns
    so that VIF is computed on higher-level aggregates (gov_index_zmean) instead.
    """
    # Deduplicate log columns: prefer *_ln_safe
    log_dupes = [
        "gdp_usd_ln", "gdp_per_capita_usd_ln", "fdi_inflow_usd_ln",
        "total_reserves_usd_ln", "exports_usd_ln", "imports_usd_ln",
        "external_debt_usd_ln", "current_account_balance_usd_ln"
    ]
    for _c in log_dupes:
        if _c in df.columns and (_c + "_safe") not in df.columns:
            # if there is no _safe version, keep the _ln and continue (nothing to drop)
            continue
        if _c in df.columns:
            try:
                df.drop(columns=[_c], inplace=True, errors="ignore")
                LOG.info("Dropping duplicate log column %s (keeping %s_safe)", _c, _c)
            except Exception:
                LOG.warning("Failed to drop duplicate log column %s", _c)

    # Drop raw governance pillars to avoid VIF explosion (we keep gov_index_zmean)
    gov_pillars = [
        "voice_accountability", "voice_accountability_imputed", "voice_accountability_z",
        "political_stability", "political_stability_imputed", "political_stability_z",
        "gov_effectiveness", "gov_effectiveness_imputed", "gov_effectiveness_z",
        "reg_quality", "reg_quality_imputed", "reg_quality_z",
        "rule_of_law", "rule_of_law_imputed", "rule_of_law_z",
        "control_corruption", "control_corruption_imputed", "control_corruption_z"
    ]
    for _c in gov_pillars:
        if _c in df.columns:
            try:
                df.drop(columns=[_c], inplace=True, errors="ignore")
                LOG.info("Dropping governance pillar column %s from VIF inputs (keeping gov_index_zmean)", _c)
            except Exception:
                LOG.warning("Failed to drop governance pillar column %s", _c)

    return df


def compute_vif(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute VIF using cleaned numeric columns only.
    Drop raw governance pillars + duplicate logs for stability.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception:
        LOG.warning("statsmodels not available; skipping VIF calculation.")
        return None

    # Copy and clean for VIF computation
    df_clean = df.copy()
    df_clean = _drop_duplicate_logs_and_gov_pillars(df_clean)

    # numeric only, drop "year"
    num = df_clean.select_dtypes(include=[np.number]).copy()
    num = num.drop(columns=["year"], errors="ignore")

    # keep columns with at least 50% observed
    vif_cols = [c for c in num.columns if num[c].notna().mean() > 0.5]
    if not vif_cols:
        LOG.info("No numeric columns with >50% non-missing values.")
        return None

    X = num[vif_cols].dropna()
    if len(X) == 0:
        LOG.info("No full-case rows for VIF calculation.")
        return None

    vif_rows = []
    for i, col in enumerate(X.columns):
        try:
            v = float(variance_inflation_factor(X.values, i))
        except Exception as e:
            LOG.warning("VIF computation failed for %s: %s", col, e)
            v = float("nan")
        vif_rows.append((col, v))

    vif_df = pd.DataFrame(vif_rows, columns=["feature", "vif"]).sort_values("vif", ascending=False)
    vif_df.to_csv(VIF_OUT, index=False)

    try:
        md5 = record_artifact(VIF_OUT, canonical_id="vif")
    except Exception as e:
        LOG.warning("record_artifact failed for vif: %s", e)
        md5 = None

    LOG.info("Saved VIF -> %s (md5=%s)", VIF_OUT, md5)
    return vif_df


def main() -> None:
    df = _load_source()

    mon_cols = [c for c in [
        "gdp_usd", "exports_usd", "imports_usd", "fdi_inflow_usd",
        "total_reserves_usd", "current_account_balance_usd", "external_debt_usd"
    ] if c in df.columns]

    LOG.info("Monetary columns considered: %s", mon_cols)

    scale_df = monetary_scale_check(df, mon_cols)
    df = create_safe_logs(df, mon_cols)

    # Prefer aggregated trade_total over per-side logs to avoid multicollinearity:
    _trade_cols_to_drop = [
        "exports_usd_ln_safe", "imports_usd_ln_safe",
        "exports_usd_ln", "imports_usd_ln"
    ]
    if "trade_total" in df.columns:
        for _c in _trade_cols_to_drop:
            if _c in df.columns:
                try:
                    df.drop(columns=[_c], inplace=True, errors="ignore")
                    LOG.info("Dropping %s (use trade_total instead to avoid multicollinearity)", _c)
                except Exception:
                    LOG.warning("Failed to drop %s; continuing", _c)

    pairs_df = export_top_correlations(df)
    vif_df = compute_vif(df)

    # Console summary
    if not pairs_df.empty:
        LOG.info("Top 10 correlations:\n%s", pairs_df.head(10).to_string(index=False))
    else:
        LOG.info("No correlation pairs available.")

    if vif_df is not None:
        LOG.info("Top VIF features:\n%s", vif_df.head(20).to_string(index=False))
    else:
        LOG.info("VIF not computed (missing statsmodels or insufficient data).")

    # Save small JSON manifest and try to record artifact
    manifest: Dict[str, Optional[str]] = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "monetary_check": str(MONETARY_OUT) if MONETARY_OUT.exists() else None,
        "top_correlations": str(CORR_OUT) if CORR_OUT.exists() else None,
        "vif": str(VIF_OUT) if VIF_OUT.exists() else None,
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
    try:
        record_artifact(MANIFEST_OUT, canonical_id="safe_transforms_manifest")
    except Exception as e:
        LOG.warning("record_artifact failed for safe_transforms_manifest: %s", e)

    LOG.info("Wrote manifest -> %s", MANIFEST_OUT)


if __name__ == "__main__":
    main()
