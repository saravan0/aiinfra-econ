#!/usr/bin/env python3
"""
Compute standardized effects for any predictor using existing per-model JSON artifacts.

Usage:
    python scripts/compute_standardized_effects.py --var trade_exposure
    python scripts/compute_standardized_effects.py --var gov_index_zmean
    python scripts/compute_standardized_effects.py --var inflation_consumer_prices_pct

What it does:
 - Loads per-model JSONs: outputs/fe_<var>.json, outputs/ols_<var>.json, outputs/elasticnet_<var>.json
 - Loads features CSV to compute sd(target) and sd(var) (default path: data/processed/features_lean_imputed.csv)
 - Computes standardized effects: b * sd(var) / sd(target)
 - Produces outputs:
     - outputs/standardized/<var>_standardized.json
     - outputs/standardized/<var>_standardized.md
     - outputs/standardized/<var>_standardized.csv
 - Prints a compact console summary with ML-focused diagnostics.
"""
import argparse
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
from datetime import datetime

OUTDIR = Path("outputs") / "standardized"
OUTDIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FEATURES_PATH = Path("data/processed/features_lean_imputed.csv")
TARGET_NAME = "gdp_growth_pct"

def safe_load_json(p: Path):
    if not p.exists():
        return {"error": f"{p} not found"}
    try:
        with open(p, "r", encoding="utf8") as fh:
            return json.load(fh)
    except Exception as e:
        return {"error": f"failed to load {p}: {e}"}

def compute_sd_values(features_path: Path, varname: str):
    if not features_path.exists():
        raise FileNotFoundError(f"features file not found: {features_path}")
    df = pd.read_csv(features_path, low_memory=False)
    if TARGET_NAME not in df.columns:
        raise KeyError(f"target {TARGET_NAME} not found in features")
    if varname not in df.columns:
        # try case-insensitive match
        for c in df.columns:
            if c.lower() == varname.lower():
                varname = c
                break
        else:
            raise KeyError(f"variable {varname} not found in features columns")
    sd_target = float(df[TARGET_NAME].dropna().std(ddof=0))
    sd_var = float(df[varname].dropna().std(ddof=0))
    return sd_target, sd_var

def compute_standardized(coef, sd_var, sd_target):
    try:
        return float(coef) * float(sd_var) / float(sd_target)
    except Exception:
        return None

def summarize_model_stats(raw_stats: dict, sd_var: float, sd_target: float):
    if raw_stats is None:
        return {"error": "no stats"}
    if "error" in raw_stats:
        return raw_stats
    out = {}
    out["model_type"] = raw_stats.get("model_type")
    out["coef"] = raw_stats.get("coef")
    out["std_err"] = raw_stats.get("std_err")
    out["pvalue"] = raw_stats.get("pvalue")
    out["n_obs"] = raw_stats.get("n_obs")
    out["conf_int"] = raw_stats.get("conf_int")
    out["mapped_exog_name"] = raw_stats.get("mapped_exog_name")
    out["mapped_index"] = raw_stats.get("mapped_index")
    out["mapped_to_feature"] = raw_stats.get("mapped_to_feature")
    # standardized
    try:
        out["standardized"] = compute_standardized(out["coef"], sd_var, sd_target) if out.get("coef") is not None else None
    except Exception:
        out["standardized"] = None
    # effect magnitude description
    if out.get("standardized") is None:
        out["magnitude"] = None
    else:
        s = abs(out["standardized"])
        if s < 0.02:
            mag = "negligible"
        elif s < 0.05:
            mag = "small"
        elif s < 0.15:
            mag = "moderate"
        else:
            mag = "large"
        out["magnitude"] = mag
    return out

def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

def write_md(path: Path, varname: str, summary: dict):
    lines = [f"# Standardized effects — {varname}\n",
             f"Generated: {datetime.utcnow().isoformat()}Z\n"]
    for model_name, stats in summary.items():
        lines.append(f"## {model_name}\n")
        if "error" in stats:
            lines.append(f"- ERROR: {stats['error']}\n")
            continue
        lines.append(f"- model_type: {stats.get('model_type')}\n")
        lines.append(f"- coef: {stats.get('coef')}\n")
        if stats.get("std_err") is not None:
            lines.append(f"- std_err: {stats.get('std_err')}\n")
        if stats.get("pvalue") is not None:
            lines.append(f"- pvalue: {stats.get('pvalue')}\n")
        if stats.get("n_obs") is not None:
            lines.append(f"- n_obs: {stats.get('n_obs')}\n")
        lines.append(f"- standardized_effect (in sd of {TARGET_NAME}): {stats.get('standardized')}\n")
        lines.append(f"- magnitude: {stats.get('magnitude')}\n")
        if stats.get("mapped_to_feature"):
            lines.append(f"- mapped_to_feature: {stats.get('mapped_to_feature')} (exog: {stats.get('mapped_exog_name')}, idx: {stats.get('mapped_index')})\n")
        # ML hints
        if model_name.lower().startswith("elastic"):
            if stats.get("coef") is not None:
                lines.append(f"- (ML) model_type: {stats.get('model_type')} — penalized coef indicates shrinkage.\n")
    with open(path, "w", encoding="utf8") as fh:
        fh.write("\n".join(lines))

def write_csv(path: Path, varname: str, summary: dict):
    rows = []
    for model_name, s in summary.items():
        if "error" in s:
            rows.append({"model": model_name, "coef": None, "std_err": None, "pvalue": None, "n_obs": None, "standardized": None, "magnitude": None})
            continue
        rows.append({"model": model_name, "coef": s.get("coef"), "std_err": s.get("std_err"), "pvalue": s.get("pvalue"),
                     "n_obs": s.get("n_obs"), "standardized": s.get("standardized"), "magnitude": s.get("magnitude")})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", required=True, help="variable name (e.g. trade_exposure)")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH), help="path to features CSV")
    parser.add_argument("--outdir", default=str(OUTDIR), help="output folder")
    args = parser.parse_args()

    varname = args.var
    features_path = Path(args.features)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load per-model JSON files
    fe_stats = safe_load_json(Path("outputs") / f"fe_{varname}.json")
    ols_stats = safe_load_json(Path("outputs") / f"ols_{varname}.json")
    en_stats = safe_load_json(Path("outputs") / f"elasticnet_{varname}.json")

    # compute SDs
    try:
        sd_target, sd_var = compute_sd_values(features_path, varname)
    except Exception as e:
        print("Failed to compute SDs:", e)
        raise

    # summarize & compute standardized
    summary = {}
    summary["FE"] = summarize_model_stats(fe_stats, sd_var, sd_target)
    summary["OLS"] = summarize_model_stats(ols_stats, sd_var, sd_target)
    summary["ElasticNet"] = summarize_model_stats(en_stats, sd_var, sd_target)

    # additional ML-focused diagnostics
    ml_diag = {}
    # shrinkage: compare absolute coef sizes (EN vs OLS)
    try:
        en_coef = en_stats.get("coef") if isinstance(en_stats, dict) else None
        ols_coef = ols_stats.get("coef") if isinstance(ols_stats, dict) else None
        if en_coef is not None and ols_coef is not None:
            ml_diag["elasticnet_vs_ols_ratio"] = None
            try:
                ml_diag["elasticnet_vs_ols_ratio"] = float(en_coef) / float(ols_coef)
            except Exception:
                ml_diag["elasticnet_vs_ols_ratio"] = None
            ml_diag["elasticnet_shrinkage_abs"] = abs(float(en_coef)) < abs(float(ols_coef))
    except Exception:
        pass

    # write outputs
    write_json(outdir / f"{varname}_standardized.json", {"var": varname, "sd_target": sd_target, "sd_var": sd_var, "summary": summary, "ml_diag": ml_diag})
    write_md(outdir / f"{varname}_standardized.md", varname, summary)
    write_csv(outdir / f"{varname}_standardized.csv", varname, summary)

    # console summary
    print(f"Variable: {varname}")
    print(f"SDs -> target ({TARGET_NAME}): {sd_target:.6g}, var ({varname}): {sd_var:.6g}")
    for m, s in summary.items():
        if "error" in s:
            print(f"--- {m} : ERROR -> {s['error']}")
        else:
            print(f"--- {m} : coef={s.get('coef')} std_err={s.get('std_err')} p={s.get('pvalue')} n={s.get('n_obs')} std_effect={s.get('standardized')} magnitude={s.get('magnitude')}")
    if ml_diag:
        print("ML diagnostics:", ml_diag)
    print("\nWrote files:")
    print(" -", outdir / f"{varname}_standardized.json")
    print(" -", outdir / f"{varname}_standardized.md")
    print(" -", outdir / f"{varname}_standardized.csv")

if __name__ == "__main__":
    main()
