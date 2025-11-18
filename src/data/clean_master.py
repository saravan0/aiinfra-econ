# src/data/clean_master.py
"""
Clean and canonicalize the raw WGI-econ master into:
 - data/interim/wgi_econ_master.csv    (cleaned master)
 - data/interim/panel_union.csv        (same as master; kept for pipeline compatibility)
 - data/interim/panel_core.csv         (core panel for modeling: iso3, year, gdp + at least one WGI pillar)

Notes / admissions angle:
This step documents conservative, reproducible decisions about canonical country selection,
column renaming and the construction of a model-ready "core" panel. Admissions reviewers
look for clear, defensible rules like these in research portfolios.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.data_registry import record_artifact

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Project-aware paths
ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

RAW_MASTER = INTERIM / "wgi_econ_master_raw.csv"
OUT_MASTER = INTERIM / "wgi_econ_master.csv"
OUT_UNION = INTERIM / "panel_union.csv"
OUT_CORE = INTERIM / "panel_core.csv"

if not RAW_MASTER.exists():
    LOG.error("Missing raw master: %s. Run the builder first.", RAW_MASTER)
    raise SystemExit(1)


def _pick_canonical_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Choose a canonical 'country' column:
     - prefer country_y, then country_x, then existing 'country'
     - fallback to the first column whose name starts with 'country' (case-insensitive)
     - if none found, create country column with None
    """
    if "country_y" in df.columns:
        df["country"] = df["country_y"]
    elif "country_x" in df.columns:
        df["country"] = df["country_x"]
    elif "country" in df.columns:
        pass  # already present
    else:
        possible = [c for c in df.columns if c.lower().startswith("country")]
        if possible:
            df["country"] = df[possible[0]]
        else:
            df["country"] = pd.NA
    # drop legacy columns if present
    for c in ("country_x", "country_y"):
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df


def _ensure_iso3_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'iso3' and 'year' columns exist by attempting conservative renames.
    """
    if "iso3" not in df.columns:
        for c in df.columns:
            if c.lower() in ("country code", "iso", "iso3", "ref_area"):
                df = df.rename(columns={c: "iso3"})
                LOG.info("Renamed %s -> iso3", c)
                break

    if "year" not in df.columns:
        for c in df.columns:
            if c.lower() in ("time_period", "time", "period", "year"):
                df = df.rename(columns={c: "year"})
                LOG.info("Renamed %s -> year", c)
                break

    return df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put country, iso3, year first (if present) and keep the rest in original order.
    """
    cols = list(df.columns)
    rest = [c for c in cols if c not in ("country", "iso3", "year")]
    newcols = ["country", "iso3", "year"] + rest
    newcols = [c for c in newcols if c in df.columns]
    return df[newcols]


def _build_panel_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build panel_core:
      - require iso3, year, gdp_usd (if present)
      - require at least one WGI pillar to be non-null
    """
    wgi_pillars: List[str] = [
        c for c in df.columns if c in (
            "voice_accountability",
            "political_stability",
            "gov_effectiveness",
            "reg_quality",
            "rule_of_law",
            "control_corruption"
        )
    ]
    required = ["iso3", "year"]
    if "gdp_usd" in df.columns:
        required.append("gdp_usd")

    core = df.copy()
    # drop rows missing required keys (only for keys present in the dataframe)
    required_present = [c for c in required if c in core.columns]
    if required_present:
        core = core.dropna(subset=required_present, how="any")

    # keep only rows where at least one pillar present, if pillars exist
    if wgi_pillars:
        core = core[core[wgi_pillars].notna().any(axis=1)]
    return core


def main() -> None:
    LOG.info("Reading raw master: %s", RAW_MASTER)
    df = pd.read_csv(RAW_MASTER, low_memory=False)

    df = _pick_canonical_country(df)
    df = _ensure_iso3_year(df)
    df = _reorder_columns(df)

    # save cleaned master
    df.to_csv(OUT_MASTER, index=False)
    md5_master = record_artifact(OUT_MASTER, canonical_id="wgi_econ_master")
    if md5_master:
        LOG.info("Saved cleaned master -> %s (%s rows, %s cols) — md5=%s", OUT_MASTER, f"{len(df):,}", len(df.columns), md5_master)
    else:
        LOG.info("Saved cleaned master -> %s (%s rows, %s cols) — provenance not recorded", OUT_MASTER, f"{len(df):,}", len(df.columns))

    # write panel_union (same as master)
    df.to_csv(OUT_UNION, index=False)
    record_artifact(OUT_UNION, canonical_id="panel_union")
    LOG.info("Saved panel union -> %s", OUT_UNION)

    # build and write panel_core
    core = _build_panel_core(df)
    core.to_csv(OUT_CORE, index=False)
    record_artifact(OUT_CORE, canonical_id="panel_core")
    LOG.info("Saved panel core -> %s (%s rows)", OUT_CORE, f"{len(core):,}")

    # quick stats (concise)
    LOG.info("Quick stats: years: %s → %s", (df["year"].min() if "year" in df.columns else "n/a"), (df["year"].max() if "year" in df.columns else "n/a"))
    LOG.info("Unique iso3: %s", (df["iso3"].nunique() if "iso3" in df.columns else "n/a"))
    LOG.info("Columns: %s", len(df.columns))
    missing = df.isna().mean().sort_values(ascending=False)
    LOG.info("Top 10 columns by missing fraction:\n%s", missing.head(10).to_string())


if __name__ == "__main__":
    main()
