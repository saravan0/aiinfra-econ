# src/data/wdi_clean_long.py
"""
Create a canonical long-form WDI table (data/interim/wdi_long.csv).

Behavior:
- Detect file encoding and header row heuristically (robust to World Bank CSV variants).
- Read the CSV with the detected header and melt wide year-columns to long format.
- Normalize column names to a small canonical set: country, iso3, indicator_name, indicator_code, year, value.
- Record provenance (md5 + sources.yaml) using src.utils.data_registry.record_artifact.

Usage (recommended):
    # from project root
    python -m src.data.wdi_clean_long

Admissions angle (why this matters):
This routine shows careful, reproducible handling of messy cross-country time-series exports
— a clear signal of real-world research readiness expected from top applicants.
"""

from __future__ import annotations

import logging
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd

from src.utils.data_registry import record_artifact

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Project paths (resolve relative to repo root when run as a module)
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

FILE = RAW / "worldbank_wdi.csv"

COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]


def _read_sample_lines(path: Path, encoding: str, n_lines: int) -> List[str]:
    """Read up to n_lines from file using the given encoding (best-effort)."""
    lines: List[str] = []
    try:
        with path.open("r", encoding=encoding, errors="replace") as fh:
            for _ in range(n_lines):
                line = fh.readline()
                if not line:
                    break
                lines.append(line)
    except Exception as exc:
        LOG.debug("Failed to read sample lines with encoding %s: %s", encoding, exc)
    return lines


def detect_encoding_and_header(encodings: List[str] = COMMON_ENCODINGS, n_lines: int = 200) -> Tuple[str, int, Optional[str]]:
    """
    Return (encoding, header_row_index (0-based), header_line_text).

    Heuristics:
    - scan the first n_lines using several encodings
    - look for header-like tokens (country, indicator, series, 1960/2023 year markers)
    - fallback to cp1252 and header row 0
    """
    for enc in encodings:
        lines = _read_sample_lines(FILE, enc, n_lines)
        if not lines:
            continue
        for i, line in enumerate(lines):
            ln = line.lower()
            if ("country" in ln and ("indicator" in ln or "series" in ln)) or ("indicator code" in ln) or ("country code" in ln):
                return enc, i, line.strip()
            if re.search(r"\b(19|20)\d{2}\b", ln):
                return enc, i, line.strip()
    # fallback
    return "cp1252", 0, None


def read_with_header(enc: str, header_row: int) -> pd.DataFrame:
    """Read CSV with pandas using provided encoding and header row."""
    # pandas header parameter is 0-based index of header row
    return pd.read_csv(FILE, encoding=enc, header=header_row, low_memory=False)


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt a WDI-style wide table to long.

    Strategy:
    - identify id_vars heuristically (country, iso/code, indicator name/code)
    - fall back to first 3-4 columns if heuristics fail
    - melt all other columns, extract year (4-digit) and coerce values to numeric
    """
    cols = list(df.columns)
    lower = [c.lower() for c in cols]

    id_candidates: List[str] = []
    targets = [
        "country name", "country", "country code",
        "indicator name", "indicator code", "series name", "series code"
    ]
    for target in targets:
        for c in cols:
            if target in c.lower() and c not in id_candidates:
                id_candidates.append(c)

    # keep order, unique
    id_vars = []
    for c in id_candidates:
        if c not in id_vars:
            id_vars.append(c)

    if len(id_vars) < 3:
        # conservative fallback: first 4 columns
        id_vars = cols[:4]

    year_cols = [c for c in cols if c not in id_vars]
    df_long = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="year", value_name="value")

    # Rename to canonical names where possible
    rename_map = {}
    if len(id_vars) >= 1:
        rename_map[id_vars[0]] = "country"
    if len(id_vars) >= 2:
        rename_map[id_vars[1]] = "iso3"
    if len(id_vars) >= 3:
        rename_map[id_vars[2]] = "indicator_name"
    if len(id_vars) >= 4:
        rename_map[id_vars[3]] = "indicator_code"
    df_long = df_long.rename(columns=rename_map)

    # Normalize year to integer (extract first 4-digit group)
    df_long["year"] = df_long["year"].astype(str).str.extract(r"(\d{4})")
    df_long = df_long[df_long["year"].notna()].copy()
    df_long["year"] = df_long["year"].astype(int)

    # Force numeric values (coerce errors -> NaN)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    return df_long


def main() -> None:
    if not FILE.exists():
        LOG.error("WDI file not found at %s", FILE)
        raise SystemExit(1)

    enc, header_idx, header_line = detect_encoding_and_header()
    LOG.info("Detected encoding=%s, header_row_index=%s", enc, header_idx)
    if header_line:
        LOG.debug("Header preview: %s", header_line)

    try:
        df = read_with_header(enc, header_idx)
    except Exception as exc:
        LOG.exception("Error reading CSV with detected header. Try increasing sample size or inspect file manually.")
        raise

    LOG.info("Columns preview (first 20): %s", list(df.columns)[:20])

    df_long = melt_to_long(df)
    out = INTERIM / "wdi_long.csv"
    df_long.to_csv(out, index=False)

    # record provenance (md5 + sources.yaml) using the project helper
    md5 = record_artifact(out, canonical_id="wdi_worldbank")
    if md5:
        LOG.info("WDI long saved to %s (rows: %s) — md5=%s", out, f"{len(df_long):,}", md5)
    else:
        LOG.warning("WDI long saved to %s but provenance recording failed", out)


if __name__ == "__main__":
    main()
