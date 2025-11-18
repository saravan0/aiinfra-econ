# src/data/generate_metadata.py
"""
Metadata & Provenance Generator

Creates:
  - data/raw/mappings/column_map_with_provenance.csv
  - data/interim/column_provenance_summary.csv

Combines:
  (a) User-defined column mapping file
  (b) Missingness statistics from the cleaned master table

Intended usage:
  from src.data.generate_metadata import make_metadata_card
  make_metadata_card(master_path, mapping_file, out_dir)

This module is kept intentionally modular and lightweight so it can be
called from any pipeline stage without creating tight coupling.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from datetime import date
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def _validate_mapping_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize the column_map.csv structure.
    Ensures the columns:
       - source_file
       - source_column
       - master_column
    exist, otherwise tries safe fallbacks.
    """
    df = df.copy()
    required = ["source_file", "source_column", "master_column"]

    # Normalize lower-case mapping
    lower = {c.lower(): c for c in df.columns}

    if not all(req in lower for req in required):
        # Attempt tolerant rename from lower-case
        cols = {c.lower(): c for c in df.columns}
        if all(req in cols for req in required):
            df = df.rename(columns={
                cols["source_file"]: "source_file",
                cols["source_column"]: "source_column",
                cols["master_column"]: "master_column"
            })
        elif len(df.columns) >= 3:
            # Fallback: assume first 3 columns are correct mapping
            df = df.rename(columns={
                df.columns[0]: "source_file",
                df.columns[1]: "source_column",
                df.columns[2]: "master_column"
            })
        else:
            raise ValueError("column_map.csv has unexpected headers; expected "
                             "source_file, source_column, master_column.")

    # Final normalization (strip whitespace)
    for col in ["source_file", "source_column", "master_column"]:
        df[col] = df[col].astype(str).str.strip()

    return df


def make_metadata_card(
    master_path: Path,
    mapping_file: Path,
    out_dir: Path
) -> tuple[Path, Path]:
    """
    Construct a provenance-aware metadata table linking:
      - master dataset columns
      - original raw source columns
      - missingness fraction
      - optional: download dates, URLs, units

    Returns:
      (path_to_column_map_with_provenance, path_to_provenance_summary)
    """

    master_path = Path(master_path)
    mapping_file = Path(mapping_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mapping_file.exists():
        raise FileNotFoundError(f"Missing mapping file: {mapping_file}")
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master file: {master_path}")

    logger.info("Loading mapping file: %s", mapping_file)
    mapping = pd.read_csv(mapping_file, dtype=str).fillna("")
    mapping = _validate_mapping_file(mapping)

    # Optional metadata enrichment placeholders
    if "download_date" not in mapping.columns:
        mapping["download_date"] = str(date.today())
    if "source_url" not in mapping.columns:
        mapping["source_url"] = ""
    if "units" not in mapping.columns:
        mapping["units"] = ""

    logger.info("Loading master dataset: %s", master_path)
    master = pd.read_csv(master_path, low_memory=False)

    # Missingness table
    missing = master.isna().mean().reset_index()
    missing.columns = ["master_column", "missing_fraction"]

    # Merge mapping + missingness
    provenance = pd.merge(
        mapping,
        missing,
        on="master_column",
        how="right"   # ensures we include columns in master even if unmapped
    )

    # Fill unmapped columns with placeholders
    provenance["source_file"] = provenance["source_file"].replace("", "UNKNOWN")
    provenance["source_column"] = provenance["source_column"].replace("", "")
    provenance["download_date"] = provenance["download_date"].replace("", str(date.today()))
    provenance["source_url"] = provenance["source_url"].replace("", "")
    provenance["units"] = provenance["units"].replace("", "")

    # Outputs
    colmap_out = mapping_file.parent / "column_map_with_provenance.csv"
    summary_out = out_dir / "column_provenance_summary.csv"

    provenance.to_csv(colmap_out, index=False)
    provenance.sort_values("missing_fraction", ascending=False).to_csv(summary_out, index=False)

    logger.info("Wrote provenance mapping -> %s", colmap_out)
    logger.info("Wrote provenance summary -> %s", summary_out)

    # Print small readable preview
    preview = (
        provenance[["master_column", "missing_fraction"]]
        .sort_values("missing_fraction", ascending=False)
        .head(10)
    )
    logger.info("Top 10 columns by missing_fraction:\n%s", preview.to_string(index=False))

    return colmap_out, summary_out


if __name__ == "__main__":
    """
    Allows running as:
        python -m src.data.generate_metadata

    Uses project defaults.
    """
    ROOT = Path.cwd()
    DEFAULT_MASTER = ROOT / "data" / "interim" / "wgi_econ_master.csv"
    DEFAULT_MAP = ROOT / "data" / "raw" / "mappings" / "column_map.csv"
    DEFAULT_OUT = ROOT / "data" / "interim"

    make_metadata_card(
        master_path=DEFAULT_MASTER,
        mapping_file=DEFAULT_MAP,
        out_dir=DEFAULT_OUT
    )
