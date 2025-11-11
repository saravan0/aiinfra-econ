# src/data/build_wgi_econ_master.py
from pathlib import Path
import pandas as pd
import logging
import sys
from functools import reduce

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

WGI_FILES = {
    "wgi_voice.csv": "voice_accountability",
    "wgi_polstab.csv": "political_stability",
    "wgi_goveff.csv": "gov_effectiveness",
    "wgi_regqual.csv": "reg_quality",
    "wgi_rulelaw.csv": "rule_of_law",
    "wgi_corrupt.csv": "control_corruption",
}

def load_wgi_frame(path: Path, target_name: str):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # detect key columns
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if "ref_area_label" in lc or "country" in lc:
            rename[c] = "country"
        elif "ref_area" in lc and "label" not in lc:
            rename[c] = "iso3"
        elif lc in ("time_period", "year", "time"):
            rename[c] = "year"
        elif lc in ("obs_value", "estimate", "value"):
            rename[c] = target_name
    df = df.rename(columns=rename)
    keep = [c for c in ["country", "iso3", "year", target_name] if c in df.columns]
    return df[keep].copy()

def build_master():
    wgi_frames = []
    for fname, master_col in WGI_FILES.items():
        fpath = RAW / fname
        frame = load_wgi_frame(fpath, master_col)
        wgi_frames.append(frame)
        logging.info("%s loaded (%d rows)", fname, len(frame))

    # --- 🩵 PATCH: unify country names before merge to avoid suffixes ---
    for f in wgi_frames:
        if "country" in f.columns:
            f.drop_duplicates(subset=["iso3", "year"], inplace=True)
    # choose join keys
    join_keys = ["iso3", "year"]
    logging.info("Merging WGI frames on %s", join_keys)

    # merge safely ignoring duplicate country columns
    def safe_merge(L, R):
        if "country" in L.columns and "country" in R.columns:
            R = R.drop(columns=["country"])
        return pd.merge(L, R, on=join_keys, how="outer")

    wgi_merged = reduce(safe_merge, wgi_frames)
    logging.info("Merged WGI shape: %s", wgi_merged.shape)

    out = INTERIM / "wgi_merged.csv"
    wgi_merged.to_csv(out, index=False)
    logging.info("Saved WGI merged -> %s", out)

if __name__ == "__main__":
    build_master()
