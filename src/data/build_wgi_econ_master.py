# src/data/build_wgi_econ_master.py
"""
Build a merged WGI table from individual WGI indicator CSVs.

This module:
- loads each WGI csv (voice, political stability, gov effectiveness, ...)
- renames detected key columns to canonical names (country, iso3, year, <indicator>)
- reduces duplicates and performs an outer merge on (iso3, year)
- writes the merged table to data/interim/wgi_merged.csv

Note: we intentionally DO NOT record canonical provenance (sources.yaml) here.
Provenance for the final econ master will be recorded later in build_master_final.
"""

from __future__ import annotations

import logging
from functools import reduce
from pathlib import Path
from typing import Dict, List

import pandas as pd

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Resolve project-root-aware paths (robust to different cwd)
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

# Files expected (filename -> canonical indicator column)
WGI_FILES: Dict[str, str] = {
    "wgi_voice.csv": "voice_accountability",
    "wgi_polstab.csv": "political_stability",
    "wgi_goveff.csv": "gov_effectiveness",
    "wgi_regqual.csv": "reg_quality",
    "wgi_rulelaw.csv": "rule_of_law",
    "wgi_corrupt.csv": "control_corruption",
}


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV with pandas with a best-effort approach; raise on hard errors."""
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        LOG.error("WGI file missing: %s", path)
        raise
    except Exception as exc:
        LOG.exception("Failed reading %s: %s", path, exc)
        raise


def _canonicalize_columns(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """
    Rename plausible columns to a canonical set:
      - country
      - iso3
      - year
      - <target_name> (the indicator)
    This function uses conservative heuristics to avoid accidental renames.
    """
    rename: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower().strip()
        if "ref_area_label" in lc or (("country" in lc) and ("code" not in lc)):
            rename[c] = "country"
        elif ("ref_area" in lc and "label" not in lc) or ("country code" in lc) or (lc == "iso3"):
            rename[c] = "iso3"
        elif lc in ("time_period", "year", "time"):
            rename[c] = "year"
        elif any(tok in lc for tok in ("obs_value", "estimate", "value", "score")):
            # map value-like columns to the target indicator name
            rename[c] = target_name
    df = df.rename(columns=rename)
    # Keep only canonical subset if present
    keep = [c for c in ["country", "iso3", "year", target_name] if c in df.columns]
    return df[keep].copy()


def load_wgi_frame(path: Path, target_name: str) -> pd.DataFrame:
    """Load and canonicalize a single WGI CSV into a small dataframe with the target column."""
    df = _read_csv_safe(path)
    # trim column names
    df.columns = [str(c).strip() for c in df.columns]
    df_clean = _canonicalize_columns(df, target_name)
    # Drop duplicate observations (iso3, year) keeping the first observed value
    if set(("iso3", "year")).issubset(df_clean.columns):
        df_clean = df_clean.drop_duplicates(subset=["iso3", "year"])
    return df_clean


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: List[str]) -> pd.DataFrame:
    """
    Merge two frames on `on` keys, dropping redundant 'country' to avoid suffixes.
    Outer merge is used to preserve coverage.
    """
    # Drop country from right if both frames have it (prefer left's country)
    if "country" in left.columns and "country" in right.columns:
        right = right.drop(columns=["country"])
    return pd.merge(left, right, on=on, how="outer")


def build_master() -> pd.DataFrame:
    """Load, canonicalize, and merge all WGI components. Returns the merged frame."""
    frames: List[pd.DataFrame] = []
    for fname, colname in WGI_FILES.items():
        fpath = RAW / fname
        LOG.info("Loading %s", fpath)
        frame = load_wgi_frame(fpath, colname)
        frames.append(frame)
        LOG.info("%s loaded (%d rows, cols=%s)", fname, len(frame), list(frame.columns))

    if not frames:
        LOG.error("No WGI frames loaded; aborting.")
        raise SystemExit(1)

    join_keys = ["iso3", "year"]
    LOG.info("Merging WGI frames on %s", join_keys)

    merged = reduce(lambda L, R: _safe_merge(L, R, join_keys), frames)
    LOG.info("Merged WGI shape: %s", merged.shape)

    out = INTERIM / "wgi_merged.csv"
    merged.to_csv(out, index=False)
    LOG.info("Saved WGI merged -> %s", out)

    # NOTE: we do not write canonical provenance here; that is recorded later after
    # the econ master is built (single canonical update). This keeps provenance
    # concentrated and easy for reviewers to audit.

    return merged


if __name__ == "__main__":
    build_master()
