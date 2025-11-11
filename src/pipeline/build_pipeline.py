# src/pipeline/build_pipeline.py
"""
Minimal orchestrator for Phase 1:
Runs data cleaning -> master build -> cleaning -> verify -> features -> lock schema -> metadata -> safe transforms
Produces data_manifest.json containing checksums for produced artifacts.
"""

import subprocess, json, hashlib, sys, time
from pathlib import Path

ROOT = Path.cwd()
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

def run(cmd):
    print("\n>>> RUNNING:", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")

def sha1(path: Path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()

def main():
    # 1. raw -> long WDI
    run("python src/data/wdi_clean_long.py")
    # 2. build WGI + WDI master
    run("python src/data/build_wgi_econ_master.py")
    # 3. clean master (produces wgi_econ_master.csv)
    run("python src/data/clean_master.py")
    # 4. verify
    run("python src/data/verify_master_structure.py")
    # 5. features
    run("python src/data/feature_engineer.py")
    # 6. generate metadata & provenance summary
    run("python src/data/generate_metadata.py")
    # 7. safe transforms & vif (optional; will require statsmodels)
    run("python src/data/safe_transforms_and_vif.py")
    # 8. lock schema (writes features_lean.csv)
    run("python src/data/lock_schema.py")

    # Build manifest
    manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    for p in [PROCESSED / "features.csv", PROCESSED / "features_lean.csv", INTERIM / "wgi_econ_master.csv", INTERIM / "wgi_merged.csv"]:
        if p.exists():
            manifest["files"][str(p.relative_to(ROOT))] = {"sha1": sha1(p), "size": p.stat().st_size}
    out = ROOT / "data_manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nWrote manifest ->", out)

if __name__ == "__main__":
    main()
