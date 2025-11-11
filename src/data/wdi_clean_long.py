# src/data/wdi_clean_long.py
"""
Robust WDI reader:
- tries several encodings
- finds the header row by scanning file lines for keywords
- reads CSV using the detected header row, then melts to long format
"""
from pathlib import Path
import pandas as pd
import io
import sys

RAW = Path("data/raw")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)
FILE = RAW / "worldbank_wdi.csv"

COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

def detect_encoding_and_header(encodings=COMMON_ENCODINGS, n_lines=200):
    """
    Return (encoding, header_row_index (0-based), header_line_text).
    """
    for enc in encodings:
        try:
            with open(FILE, "r", encoding=enc, errors="replace") as f:
                lines = [next(f) for _ in range(n_lines)]
        except StopIteration:
            # file shorter than n_lines, that's fine
            pass
        except Exception as e:
            print(f"Encoding {enc} failed to open file: {e}", flush=True)
            continue

        # scan each line for header-like features
        for i, line in enumerate(lines):
            ln = line.lower()
            # common header token combos in WDI exports
            if ("country" in ln and ("indicator" in ln or "series" in ln)) or ("indicator code" in ln) or ("country code" in ln):
                return enc, i, line.strip()
            # alternatively, a header often contains '1960' as first year column
            if "1960" in ln or "2023" in ln:
                return enc, i, line.strip()
        # if we reached here, try next encoding
    # fallback: try to open with cp1252 and take first line as header
    return "cp1252", 0, None

def read_with_header(enc, header_row):
    # pandas read_csv header param is 0-based index of header row
    df = pd.read_csv(FILE, encoding=enc, header=header_row, low_memory=False)
    return df

def melt_to_long(df):
    # Identify the id_vars heuristically
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    id_candidates = []
    # prefer exact matches
    for target in ["country name", "country", "country code", "indicator name", "indicator code", "series name", "series code"]:
        for c in cols:
            if target in c.lower() and c not in id_candidates:
                id_candidates.append(c)
    # unique the ids keeping order
    id_vars = []
    for c in id_candidates:
        if c not in id_vars:
            id_vars.append(c)
    # fallback minimal id variables: first two cols assumed country + code + next is indicator
    if len(id_vars) < 3:
        # try to pick first four columns as id_vars if they look reasonable
        id_vars = cols[:4]
    # now melt using all non-id numeric columns as year columns
    # detect year columns by regex of four digits or columns convertible to int
    year_cols = [c for c in cols if c not in id_vars]
    # perform melt
    df_long = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="year", value_name="value")
    # rename id_vars to canonical names if possible
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
    # normalize year column (extract 4-digit year)
    import re
    df_long["year"] = df_long["year"].astype(str).str.extract(r"(\d{4})")
    df_long = df_long[df_long["year"].notna()].copy()
    df_long["year"] = df_long["year"].astype(int)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    return df_long

def main():
    if not FILE.exists():
        print(f"ERROR: file not found: {FILE}", file=sys.stderr)
        sys.exit(1)

    enc, header_idx, header_line = detect_encoding_and_header()
    print(f"Detected encoding={enc}, header_row_index={header_idx}")
    if header_line:
        print("Detected header line preview:", header_line)
    try:
        df = read_with_header(enc, header_idx)
    except Exception as e:
        print("Error reading CSV with detected header. Details:", e)
        print("Try increasing n_lines in detect or inspect file manually.", flush=True)
        raise

    print("Columns preview (first 20):", list(df.columns)[:20])
    df_long = melt_to_long(df)
    out = INTERIM / "wdi_long.csv"
    df_long.to_csv(out, index=False)
    print(f"✅ WDI long saved to {out} (rows: {len(df_long):,})")

if __name__ == "__main__":
    main()
