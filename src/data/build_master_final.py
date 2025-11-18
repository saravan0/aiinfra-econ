# src/data/build_master_final.py
"""
Final master builder: combine WGI (governance) + WDI (economic) datasets
into a single canonical master table.

Produces:
 - data/interim/wgi_econ_master_raw.csv
 - data/interim/wgi_econ_master_missingness.csv

Design choices (brief):
 - mappings are read from data/raw/mappings/column_map.csv and are
   used to translate WDI indicator codes -> master column names.
 - WGI merged table (data/interim/wgi_merged.csv) and WDI long (data/interim/wdi_long.csv)
   are the canonical upstream artifacts produced earlier in the pipeline.
 - This step records provenance for the final master artifact (md5 + sources.yaml),
   providing a single point a reviewer can verify.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Set

import pandas as pd

from src.utils.data_registry import record_artifact

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Project paths (robust to run location)
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MAPPINGS = RAW / "mappings"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)


def _load_mapping() -> Dict[str, str]:
    """
    Load column_map.csv and return a mapping: indicator_code -> master_column_name.
    Only keep mappings for WDI (source_file containing worldbank_wdi).
    """
    mfile = MAPPINGS / "column_map.csv"
    if not mfile.exists():
        LOG.error("Mapping file not found: %s", mfile)
        raise SystemExit(1)

    m = pd.read_csv(mfile)
    # normalize column names in the mapping file
    m.columns = [c.strip().lower() for c in m.columns]
    # guard: expect columns 'source_file','source_column','master_column'
    required = {"source_file", "source_column", "master_column"}
    if not required.issubset(set(m.columns)):
        LOG.error("Mapping file %s missing required columns: %s", mfile, required - set(m.columns))
        raise SystemExit(1)

    wdi_rows = m[m["source_file"].str.contains("worldbank_wdi", case=False, na=False)]
    # keys: indicator code in data (source_column), values: desired master column name
    mapping = dict(zip(wdi_rows["source_column"].str.strip(), wdi_rows["master_column"].str.strip()))
    LOG.info("Loaded %d mappings for WDI indicators", len(mapping))
    return mapping


def _load_wgi() -> pd.DataFrame:
    path = INTERIM / "wgi_merged.csv"
    if not path.exists():
        LOG.error("WGI merged artifact missing: %s", path)
        raise SystemExit(1)
    df = pd.read_csv(path, low_memory=False)
    LOG.info("Loaded WGI merged: %s", df.shape)
    return df


def _load_wdi_as_wide(mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Load wdi_long.csv, filter to mapped indicators, pivot to wide by indicator_code,
    and rename pivoted columns using the mapping (indicator_code -> master_name).
    """
    path = INTERIM / "wdi_long.csv"
    if not path.exists():
        LOG.error("WDI long artifact missing: %s", path)
        raise SystemExit(1)

    df = pd.read_csv(path, low_memory=False)
    # Ensure the key columns exist
    for req in ("indicator_code", "country", "iso3", "year", "value"):
        if req not in df.columns:
            LOG.error("Expected column %s not in WDI long", req)
            raise SystemExit(1)

    mapping_keys: Set[str] = set(mapping.keys())
    present_codes = set(df["indicator_code"].unique()).intersection(mapping_keys)
    if not present_codes:
        LOG.warning("No indicator codes from mapping are present in wdi_long.csv (mapping keys may not match).")
    # Filter to mapped codes only (keeps the dataset compact)
    df_filtered = df[df["indicator_code"].isin(present_codes)].copy()

    # pivot to wide: one row per (country, iso3, year), indicator codes -> columns
    wide = df_filtered.pivot_table(
        index=["country", "iso3", "year"],
        columns="indicator_code",
        values="value",
        aggfunc="first"
    ).reset_index()

    # rename wide columns from indicator_codes to master names using mapping
    rename_map = {code: mapping[code] for code in present_codes}
    wide = wide.rename(columns=rename_map)
    LOG.info("Loaded WDI wide: %s (mapped indicators=%d)", wide.shape, len(rename_map))
    return wide


def build_master() -> pd.DataFrame:
    """Main assembly: load mappings, load WGI and WDI, merge, write master and missingness."""
    mapping = _load_mapping()
    wgi = _load_wgi()
    wdi = _load_wdi_as_wide(mapping)

    # merge on iso3, year (outer to preserve coverage)
    LOG.info("Merging WGI and WDI on ['iso3', 'year']")
    master = pd.merge(wgi, wdi, on=["iso3", "year"], how="outer", copy=False)

    out = INTERIM / "wgi_econ_master_raw.csv"
    master.to_csv(out, index=False)
    md5_master = record_artifact(out, canonical_id="wgi_econ_master_raw")
    if md5_master:
        LOG.info("Saved combined master -> %s %s rows, cols=%s — md5=%s", out, f"{master.shape[0]:,}", master.shape[1], md5_master)
    else:
        LOG.info("Saved combined master -> %s (%s,%s) — provenance not recorded", out, master.shape[0], master.shape[1])

    # write missingness summary
    miss = master.isna().mean().sort_values(ascending=False)
    miss_df = miss.reset_index()
    miss_df.columns = ["column", "missing_fraction"]
    miss_out = INTERIM / "wgi_econ_master_missingness.csv"
    miss_df.to_csv(miss_out, index=False)
    md5_miss = record_artifact(miss_out, canonical_id="wgi_econ_master_missingness")
    LOG.info("Saved missingness summary -> %s (md5=%s)", miss_out, md5_miss)

    # optional: record upstream WGI artifact here as a single canonical provenance point
    try:
        wgi_artifact = INTERIM / "wgi_merged.csv"
        if wgi_artifact.exists():
            record_artifact(wgi_artifact, canonical_id="wgi")
            LOG.info("Recorded provenance for upstream wgi_merged.csv (canonical_id='wgi').")
    except Exception:
        LOG.debug("Upstream wgi provenance recording skipped or failed; continuing.")

    return master


if __name__ == "__main__":
    build_master()
