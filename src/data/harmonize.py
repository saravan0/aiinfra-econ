# src/data/harmonize.py
"""
Orchestrator for data harmonization.

Steps:
 1. Validate raw files & mappings exist
 2. Run WDI reshape -> data/interim/wdi_long.csv
 3. Run build master -> data/interim/wgi_econ_master_raw.csv
 4. Run simple QA + write missingness report
 5. Produce panel union/core placeholders
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
MAPPINGS = RAW / "mappings"
INTERIM = ROOT / "data" / "interim"

REQUIRED_RAW = [
    "worldbank_wdi.csv",
    "wgi_voice.csv",
    "wgi_polstab.csv",
    "wgi_goveff.csv",
    "wgi_regqual.csv",
    "wgi_rulelaw.csv",
    "wgi_corrupt.csv",
]
REQUIRED_MAP_FILES = [
    "column_map.csv",
    "country_map.csv",
    "units_map.csv"
]

def check_files():
    missing = []
    for f in REQUIRED_RAW:
        if not (RAW / f).exists():
            missing.append(str(RAW / f))
    for f in REQUIRED_MAP_FILES:
        if not (MAPPINGS / f).exists():
            missing.append(str(MAPPINGS / f))
    if missing:
        logging.error("Missing required files:\n" + "\n".join(missing))
        return False
    logging.info("All required raw files and mappings present.")
    return True

def run_script(relpath):
    """
    Helper to run a Python script relative to project root using the same interpreter.
    It captures and prints stdout/stderr.
    """
    py = sys.executable
    script = ROOT / relpath
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    logging.info(f"Running: {script}")
    proc = subprocess.run([py, str(script)], capture_output=True, text=True)
    logging.info(proc.stdout)
    if proc.returncode != 0:
        logging.error(proc.stderr)
        raise RuntimeError(f"Script {script} failed (exit code {proc.returncode}). See logs above.")
    logging.info(f"Completed: {script}")
    return True

def simple_qa(master_path: Path):
    df = pd.read_csv(master_path, low_memory=False)
    # Basic stats
    stats = {
        "rows": len(df),
        "unique_countries": int(df['iso3'].nunique(dropna=True)) if 'iso3' in df.columns else None,
        "year_min": int(df['year'].min()) if 'year' in df.columns else None,
        "year_max": int(df['year'].max()) if 'year' in df.columns else None,
        "columns": list(df.columns)
    }
    logging.info(f"Master stats: rows={stats['rows']}, countries={stats['unique_countries']}, years={stats['year_min']}-{stats['year_max']}")

    # Missingness by column
    missing = df.isna().mean().sort_values(ascending=False)
    missing_file = INTERIM / "wgi_econ_master_missingness.csv"
    missing.to_csv(missing_file, header=["missing_fraction"])
    logging.info(f"Saved missingness report -> {missing_file}")

    # Save top-10 worst columns for quick glance
    top_worst = missing.head(10)
    logging.info(f"Top 10 columns by missingness:\n{top_worst.to_string()}")

    return stats, missing_file

def write_manifest(files_written: dict):
    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "files": files_written
    }
    out = INTERIM / "harmonize_manifest.json"
    import json
    out.write_text(json.dumps(manifest, indent=2))
    logging.info(f"Wrote manifest to {out}")

def main():
    INTERIM.mkdir(parents=True, exist_ok=True)

    if not check_files():
        logging.error("Fix missing files and re-run harmonize.py")
        sys.exit(1)

    # 1. wdi clean long
    run_script("src/data/wdi_clean_long.py")

    # 2. build master
    run_script("src/data/build_wgi_econ_master.py")

    # 3. simple QA
    master_path = INTERIM / "wgi_econ_master_raw.csv"
    # fallback to wgi_econ_master.csv (older script naming)
    if not master_path.exists():
        master_path = INTERIM / "wgi_econ_master.csv"
    if not master_path.exists():
        logging.error("Master dataset not found after build step. Aborting.")
        sys.exit(1)

    stats, missing_file = simple_qa(master_path)

    # 4. produce union/core placeholders (no modification, just copies)
    union_path = INTERIM / "panel_union.csv"
    core_path = INTERIM / "panel_core.csv"
    df_master = pd.read_csv(master_path, low_memory=False)
    df_master.to_csv(union_path, index=False)
    # panel_core: drop rows missing ALL governance pillars or GDP
    core = df_master.dropna(subset=["gov_effectiveness","gdp_usd"], how="any")
    core.to_csv(core_path, index=False)

    files_written = {
        "wdi_long": str(INTERIM / "wdi_long.csv"),
        "master_raw": str(master_path),
        "missingness": str(missing_file),
        "panel_union": str(union_path),
        "panel_core": str(core_path),
    }
    write_manifest(files_written)
    logging.info("Harmonization pipeline complete. Inspect missingness report and panel_core/panel_union files.")

if __name__ == "__main__":
    main()
