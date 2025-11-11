# src/data/build_master_final.py
"""
Final builder:
 - Loads governance (WGI) and economic (WDI) datasets
 - Applies mappings from column_map.csv
 - Merges them into wgi_econ_master_raw.csv
 - Writes missingness summary
"""

from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
MAPPINGS = RAW / "mappings"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

def load_mapping():
    mfile = MAPPINGS / "column_map.csv"
    m = pd.read_csv(mfile)
    m.columns = [c.strip().lower() for c in m.columns]
    m = m[m["source_file"].str.contains("worldbank_wdi", case=False, na=False)]
    mapping = dict(zip(m["source_column"].str.strip(), m["master_column"].str.strip()))
    logging.info("Loaded %d mappings", len(mapping))
    return mapping

def load_wgi():
    path = INTERIM / "wgi_merged.csv"
    df = pd.read_csv(path)
    logging.info("Loaded WGI merged: %s", df.shape)
    return df

def load_wdi(mapping):
    path = INTERIM / "wdi_long.csv"
    df = pd.read_csv(path)
    df = df[df["indicator_code"].isin(mapping.keys())]
    wide = df.pivot_table(index=["country","iso3","year"], 
                          columns="indicator_code", 
                          values="value", 
                          aggfunc="first").reset_index()
    wide = wide.rename(columns=mapping)
    logging.info("Loaded WDI wide: %s", wide.shape)
    return wide

def build_master():
    mapping = load_mapping()
    wgi = load_wgi()
    wdi = load_wdi(mapping)

    master = pd.merge(wgi, wdi, on=["iso3","year"], how="outer")
    out = INTERIM / "wgi_econ_master_raw.csv"
    master.to_csv(out, index=False)
    logging.info("Saved combined master -> %s (%s)", out, master.shape)

    miss = master.isna().mean().sort_values(ascending=False)
    miss.to_csv(INTERIM / "wgi_econ_master_missingness.csv", header=["missing_fraction"])
    logging.info("Saved missingness summary -> %s", INTERIM / "wgi_econ_master_missingness.csv")

if __name__ == "__main__":
    build_master()
