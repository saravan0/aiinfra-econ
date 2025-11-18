# src/data/harmonize.py
"""
Data Harmonization Orchestrator

This module runs a minimal, reproducible sequence of steps:
  1. Validate raw WDI/WGI files and mapping tables
  2. Generate WDI long-format table
  3. Build WGI → ECON master (raw)
  4. Run a basic QA (missingness, ranges)
  5. Produce default panel_union and panel_core extractions
  6. Write a harmonization manifest (timestamps + created files)

The module is intentionally lightweight. It does not perform heavy
transformations itself; instead it calls underlying scripts that
handle their respective domain logic. This keeps the harmonization
stage easy to audit and reason about.
"""

from __future__ import annotations
import subprocess
import sys
import json
from pathlib import Path
import pandas as pd
import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
MAPPINGS = RAW / "mappings"
INTERIM = ROOT / "data" / "interim"

# Required raw datasets
REQUIRED_RAW = [
    "worldbank_wdi.csv",
    "wgi_voice.csv",
    "wgi_polstab.csv",
    "wgi_goveff.csv",
    "wgi_regqual.csv",
    "wgi_rulelaw.csv",
    "wgi_corrupt.csv",
]

# Required mapping files
REQUIRED_MAP_FILES = [
    "column_map.csv",
    "country_map.csv",
    "units_map.csv"
]


def check_required_files() -> bool:
    """
    Verify that all required raw files and mapping files exist.
    Logs missing components and returns False if any are absent.
    """
    missing = []

    for fname in REQUIRED_RAW:
        if not (RAW / fname).exists():
            missing.append(str(RAW / fname))

    for fname in REQUIRED_MAP_FILES:
        if not (MAPPINGS / fname).exists():
            missing.append(str(MAPPINGS / fname))

    if missing:
        logger.error("Missing required files:\n%s", "\n".join(missing))
        return False

    logger.info("All required raw datasets and mapping files are present.")
    return True


def run_script(relpath: str):
    """
    Run a python script relative to the project root.

    Prefer running as a module (python -m <module>) so package imports like
    `from src...` work. Falls back to running the script file if module conversion fails.
    Raises RuntimeError on non-zero exit with captured stderr for easier debugging.
    """
    py = sys.executable
    script_path = ROOT / relpath

    # attempt to convert "src/data/foo.py" -> "src.data.foo" for -m invocation
    try:
        rel = Path(relpath)
        if rel.suffix == ".py":
            module_parts = rel.with_suffix("").parts
            # normalize to dot-notation only if it starts with 'src'
            if module_parts[0] == "src":
                module_name = ".".join(module_parts)
            else:
                module_name = None
        else:
            module_name = None
    except Exception:
        module_name = None

    # Prefer module invocation when possible
    if module_name:
        cmd = [py, "-m", module_name]
        logging.info("Running module: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
    else:
        # fallback to direct script execution
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        cmd = [py, str(script_path)]
        logging.info("Running script: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.stdout:
        logging.info(proc.stdout.strip())
    if proc.stderr:
        logging.warning(proc.stderr.strip())

    if proc.returncode != 0:
        logging.error("Script %s failed (exit code %s). Stderr:\n%s", script_path, proc.returncode, proc.stderr)
        raise RuntimeError(f"Script {script_path} failed (exit code {proc.returncode}). See logs above.")
    logging.info("Completed: %s", script_path)
    return True

def simple_quality_report(master_path: Path) -> tuple[dict, Path]:
    """
    Produce a minimal diagnostic summary of the master dataset:
       - row count
       - unique iso3 count
       - year range
       - missingness by variable

    Returns:
      stats_dict, path_to_missingness_file
    """
    df = pd.read_csv(master_path, low_memory=False)

    stats = {
        "rows": len(df),
        "unique_countries": int(df["iso3"].nunique(dropna=True)) if "iso3" in df.columns else None,
        "year_min": int(df["year"].min()) if "year" in df.columns else None,
        "year_max": int(df["year"].max()) if "year" in df.columns else None,
        "columns": list(df.columns)
    }

    logger.info(
        "Master stats: rows=%s, countries=%s, years=%s-%s",
        stats["rows"], stats["unique_countries"],
        stats["year_min"], stats["year_max"]
    )

    missing = df.isna().mean().sort_values(ascending=False)
    missing_file = INTERIM / "wgi_econ_master_missingness.csv"
    missing.to_csv(missing_file, header=["missing_fraction"])

    logger.info("Saved missingness report -> %s", missing_file)
    logger.info("Top 10 missing columns:\n%s", missing.head(10).to_string())

    return stats, missing_file


def write_manifest(files_written: dict) -> Path:
    """
    Write a JSON manifest documenting the harmonization stage —
    useful for reproducibility logs.
    """
    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "files": files_written
    }

    out = INTERIM / "harmonize_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote harmonization manifest -> %s", out)
    return out


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)

    # Check raw / map files
    if not check_required_files():
        logger.error("Fix missing files and re-run harmonize.py.")
        sys.exit(1)

    # Step 1 — Generate WDI long-format
    run_script("src/data/wdi_clean_long.py")

    # Step 2 — Build combined WGI → ECON master
    run_script("src/data/build_wgi_econ_master.py")

    # Determine which master file exists
    master_path = INTERIM / "wgi_econ_master_raw.csv"
    if not master_path.exists():
        master_path = INTERIM / "wgi_econ_master.csv"

    if not master_path.exists():
        logger.error("Master dataset not found after build step.")
        sys.exit(1)

    # Step 3 — Simple QA
    stats, missing_file = simple_quality_report(master_path)

    # Step 4 — Produce default panel outputs
    union_path = INTERIM / "panel_union.csv"
    core_path = INTERIM / "panel_core.csv"

    df_master = pd.read_csv(master_path, low_memory=False)
    df_master.to_csv(union_path, index=False)

    # Simple core extraction
    core = df_master.dropna(subset=["gov_effectiveness", "gdp_usd"], how="any")
    core.to_csv(core_path, index=False)

    # Step 5 — Write manifest
    files_written = {
        "wdi_long": str(INTERIM / "wdi_long.csv"),
        "master_raw": str(master_path),
        "missingness": str(missing_file),
        "panel_union": str(union_path),
        "panel_core": str(core_path)
    }
    write_manifest(files_written)

    logger.info("Harmonization pipeline complete.")


if __name__ == "__main__":
    main()
