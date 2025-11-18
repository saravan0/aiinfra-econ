# src/data/verify_master_structure.py
"""
Generate verification reports for the master table (wgi_econ_master.csv).

Produces:
 - data/interim/verify_missingness.csv
 - data/interim/verify_year_coverage.csv
 - data/interim/verify_country_coverage.csv
 - data/interim/verify_value_stats.csv

Each artifact is checksum-recorded via src.utils.data_registry.record_artifact
so provenance is available for reviewers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.data_registry import record_artifact

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Paths (project-root aware)
ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

MASTER = INTERIM / "wgi_econ_master.csv"
if not MASTER.exists():
    LOG.error("Missing master file: %s. Run build_master_final / clean_master first.", MASTER)
    raise SystemExit(1)


def _write_and_record(df: pd.DataFrame, out_path: Path, canonical_id: str | None = None) -> None:
    """Write dataframe to CSV and record provenance (md5 + sources.yaml) if canonical_id provided."""
    df.to_csv(out_path, index=False)
    md5 = record_artifact(out_path, canonical_id=canonical_id)
    if md5:
        LOG.info("Wrote %s (rows=%s) — md5=%s", out_path.name, f"{len(df):,}", md5)
    else:
        LOG.warning("Wrote %s but provenance recording failed or returned None.", out_path.name)


def run_verification() -> None:
    """Main logic: compute missingness, coverage, and numeric stats and write outputs."""
    df = pd.read_csv(MASTER, low_memory=False)

    # basic info
    rows = len(df)
    cols = len(df.columns)
    years = (int(df["year"].min()), int(df["year"].max())) if "year" in df.columns else ("n/a", "n/a")
    n_countries = int(df["iso3"].nunique()) if "iso3" in df.columns else "n/a"

    # missingness per column
    missing = df.isna().mean().sort_values(ascending=False)
    missing_df = missing.reset_index()
    missing_df.columns = ["column", "missing_fraction"]

    # coverage by year (unique countries per year)
    if "year" in df.columns and "iso3" in df.columns:
        year_counts = df.groupby("year")["iso3"].nunique().reset_index().rename(columns={"iso3": "n_countries"})
    else:
        year_counts = pd.DataFrame(columns=["year", "n_countries"])

    # coverage by country (min/max year, count)
    if "iso3" in df.columns and "year" in df.columns:
        country_years = df.groupby("iso3")["year"].agg(["min", "max", "count"]).reset_index().rename(
            columns={"count": "n_obs"}
        )
    else:
        country_years = pd.DataFrame(columns=["iso3", "min", "max", "n_obs"])

    # numeric value stats
    num = df.select_dtypes(include=["number"])
    # keep pandas' describe output, but format consistently
    stats = num.describe().T.reset_index().rename(columns={"index": "column"})

    # detect suspicious numeric columns: very large scale or huge max/median ratio
    suspicious: List[str] = []
    if not stats.empty:
        # ensure required columns exist
        for _, r in stats.iterrows():
            col = r["column"]
            mean = r.get("mean", 0)
            med = r.get("50%", 0)
            mx = r.get("max", 0)
            try:
                if abs(mean) > 1e9 or (abs(mx) / (abs(med) + 1e-9) > 1e6):
                    suspicious.append(col)
            except Exception:
                # be conservative and skip columns with unexpected stats
                LOG.debug("Skipping suspiciousness test for %s due to unexpected stats", col)

    # write outputs + record provenance
    _write_and_record(missing_df, INTERIM / "verify_missingness.csv", canonical_id="verify_missingness")
    _write_and_record(year_counts, INTERIM / "verify_year_coverage.csv", canonical_id="verify_year_coverage")
    _write_and_record(country_years, INTERIM / "verify_country_coverage.csv", canonical_id="verify_country_coverage")
    _write_and_record(stats, INTERIM / "verify_value_stats.csv", canonical_id="verify_value_stats")

    # Print summary to console for quick inspection (also in logs)
    LOG.info("MASTER: %s", MASTER)
    LOG.info("rows: %s cols: %s", rows, cols)
    LOG.info("years: %s", years)
    LOG.info("unique iso3: %s", n_countries)
    LOG.info("")
    LOG.info("Top 12 most-missing columns:\n%s", missing.head(12).to_string())
    LOG.info("")
    LOG.info("Suspicious numeric columns (very large scale or outliers): %s", suspicious)
    LOG.info("Saved reports to: %s", INTERIM)


if __name__ == "__main__":
    run_verification()
