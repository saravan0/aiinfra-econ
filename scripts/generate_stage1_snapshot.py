#!/usr/bin/env python3
"""
generate_stage1_snapshot.py

Collects all Stage-1 artifacts into a single snapshot (JSON/CSV/MD).
Optionally uses artifacts/elasticnet_unscaled_coefs.json to rescale EN coefficients.

Produces:
 - reports/stage1_snapshot.json
 - reports/stage1_snapshot.csv
 - reports/stage1_snapshot.md
 - reports/model_comparison_table.csv

Usage:
    python scripts/generate_stage1_snapshot.py
    python scripts/generate_stage1_snapshot.py --rescale-en
"""

from pathlib import Path
import json
import argparse
import pandas as pd
from datetime import datetime

OUTDIR = Path("reports")
OUTDIR.mkdir(parents=True, exist_ok=True)
STANDARDIZED_DIR = Path("outputs") / "standardized"
OUTPUTS = Path("outputs")
ARTIFACTS = Path("artifacts")

VARS = ["trade_exposure", "gov_index_zmean", "inflation_consumer_prices_pct"]

def safe_load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf8"))
    except Exception:
        return None

def rescale_en_coeffs_map():
    p = ARTIFACTS / "elasticnet_unscaled_coefs.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf8"))
        except Exception:
            return None
    return None

def collect_for_var(var, en_unscaled_map=None):
    # load standardized summary
    std_p = STANDARDIZED_DIR / f"{var}_standardized.json"
    std = safe_load(std_p) or {}
    # load per model JSON
    fe = safe_load(OUTPUTS / f"fe_{var}.json") or {}
    ols = safe_load(OUTPUTS / f"ols_{var}.json") or {}
    en = safe_load(OUTPUTS / f"elasticnet_{var}.json") or {}
    # override EN coef with unscaled if requested
    if en_unscaled_map and var in en_unscaled_map:
        try:
            en["coef"] = float(en_unscaled_map[var])
            # recompute standardized in standardized summary if present
            sd_target = std.get("sd_target")
            sd_var = std.get("sd_var")
            if sd_target and sd_var:
                std_effect = float(en["coef"]) * float(sd_var) / float(sd_target)
                en_std = {"coef": en["coef"], "standardized": std_effect}
            else:
                en_std = {"coef": en["coef"]}
        except Exception:
            pass
    snapshot_entry = {
        "variable": var,
        "standardized_summary": std,
        "fe_json": fe,
        "ols_json": ols,
        "elasticnet_json": en
    }
    return snapshot_entry

def to_markdown(snapshot, outp: Path):
    lines = [f"# Stage-1 Snapshot", f"Generated: {datetime.utcnow().isoformat()}Z", ""]
    for entry in snapshot:
        v = entry["variable"]
        lines.append(f"## {v}")
        ss = entry["standardized_summary"]
        if ss:
            sd_target = ss.get("sd_target")
            sd_var = ss.get("sd_var")
            lines.append(f"- sd_target: {sd_target}")
            lines.append(f"- sd_var: {sd_var}")
        # quick table of models
        for m in ["FE", "OLS", "ElasticNet"]:
            stats = entry.get(f"{m.lower()}_json") if entry.get(f"{m.lower()}_json") else entry.get("fe_json") if m=="FE" else None
        # simplified line
        std = entry.get("standardized_summary", {}).get("summary", {})
        for mname, sval in std.items():
            lines.append(f"### {mname}")
            lines.append(f"- coef: {sval.get('coef')}")
            lines.append(f"- standardized: {sval.get('standardized')}")
            lines.append("")
    outp.write_text("\n".join(lines), encoding="utf8")

def build_model_comparison_table(snapshot):
    rows = []
    for entry in snapshot:
        var = entry["variable"]
        std = entry.get("standardized_summary", {})
        summ = std.get("summary") if isinstance(std, dict) else None
        fe = summ.get("FE") if summ else {}
        ols = summ.get("OLS") if summ else {}
        en = summ.get("ElasticNet") if summ else {}
        rows.append({
            "variable": var,
            "fe_coef": fe.get("coef"),
            "fe_std_effect": fe.get("standardized"),
            "ols_coef": ols.get("coef"),
            "ols_std_effect": ols.get("standardized"),
            "en_coef": en.get("coef"),
            "en_std_effect": en.get("standardized")
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescale-en", action="store_true", help="If present, override ElasticNet coefficients with artifacts/elasticnet_unscaled_coefs.json mapping (if exists).")
    args = parser.parse_args()

    en_map = rescale_en_coeffs_map() if args.rescale_en else None
    snapshot = []
    for var in VARS:
        entry = collect_for_var(var, en_unscaled_map=en_map)
        snapshot.append(entry)

    # write snapshot JSON
    out_json = OUTDIR / "stage1_snapshot.json"
    out_json.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf8")
    # write CSV comparison table
    df = build_model_comparison_table(snapshot)
    out_csv = OUTDIR / "model_comparison_table.csv"
    df.to_csv(out_csv, index=False)
    # write simple md summary
    out_md = OUTDIR / "stage1_snapshot.md"
    to_markdown(snapshot, out_md)

    print("Wrote:")
    print(" -", out_json)
    print(" -", out_csv)
    print(" -", out_md)
    print("Stage-1 snapshot generation complete.")

if __name__ == "__main__":
    main()
