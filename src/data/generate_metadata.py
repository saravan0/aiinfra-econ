# src/data/generate_metadata.py
"""
Generates a provenance + missingness mapping for master columns.
Outputs:
 - data/raw/mappings/column_map_with_provenance.csv
 - data/interim/column_provenance_summary.csv
Requires: pandas
"""
from pathlib import Path
import pandas as pd
from datetime import date

ROOT = Path.cwd()
MAPPINGS_DIR = ROOT / "data" / "raw" / "mappings"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

COLMAP = MAPPINGS_DIR / "column_map.csv"
MASTER = INTERIM / "wgi_econ_master.csv"
OUT_MAP = MAPPINGS_DIR / "column_map_with_provenance.csv"
OUT_SUM = INTERIM / "column_provenance_summary.csv"

if not COLMAP.exists():
    raise SystemExit(f"Missing mapping file: {COLMAP}")
if not MASTER.exists():
    raise SystemExit(f"Missing master file: {MASTER}")

m = pd.read_csv(COLMAP, dtype=str).fillna("")
m.columns = [c.strip() for c in m.columns]
# normalize header names
lower_map = {c.lower(): c for c in m.columns}
required = ['source_file','source_column','master_column']
if not set(k in lower_map for k in required):
    # attempt tolerant rename
    cols = {c.lower(): c for c in m.columns}
    if not set(required).issubset(set(cols.keys())):
        # fallback: try assumed positions
        if len(m.columns) >= 3:
            m = m.rename(columns={m.columns[0]:'source_file', m.columns[1]:'source_column', m.columns[2]:'master_column'})
        else:
            raise SystemExit("column_map.csv has unexpected headers; please fix.")
    else:
        m = m.rename(columns={cols['source_file']:'source_file', cols['source_column']:'source_column', cols['master_column']:'master_column'})

# compute missingness from master
master = pd.read_csv(MASTER, low_memory=False)
missing = master.isna().mean().reset_index()
missing.columns = ['master_column','missing_fraction']

# attach download_date, url and units placeholders if absent
if 'download_date' not in m.columns:
    m['download_date'] = str(date.today())
if 'source_url' not in m.columns:
    m['source_url'] = ""
if 'units' not in m.columns:
    m['units'] = ""

# merge and save
prov = pd.merge(m, missing, on='master_column', how='right')
# if some master columns have no row in mapping, add stub rows
prov['source_file'] = prov['source_file'].fillna("UNKNOWN")
prov['source_column'] = prov['source_column'].fillna("")
prov['download_date'] = prov['download_date'].fillna(str(date.today()))
prov['source_url'] = prov['source_url'].fillna("")
prov['units'] = prov['units'].fillna("")

prov.to_csv(OUT_MAP, index=False)
prov.sort_values('missing_fraction', ascending=False).to_csv(OUT_SUM, index=False)
print("Wrote provenance mapping ->", OUT_MAP)
print("Wrote provenance summary ->", OUT_SUM)
print("Top 10 by missing_fraction:")
print(prov[['master_column','missing_fraction']].sort_values('missing_fraction', ascending=False).head(10).to_string(index=False))
