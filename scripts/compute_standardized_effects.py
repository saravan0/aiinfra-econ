#!/usr/bin/env python3
"""
Compute standardized effects for any predictor using existing per-model JSON artifacts.

Patched to load per-variable extractor output at:
  outputs/variables/<var>/<var>_summary.json

Usage:
    python scripts/compute_standardized_effects.py --var trade_exposure
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

def load_var_summary(varname: str):
    """
    Robust loader for model stats. Attempts multiple legacy locations then new
    extractor location: outputs/variables/<var>/<var>_summary.json

    Returns a dict with keys 'FE', 'OLS', 'ElasticNet' each mapping to a small dict
    containing at least 'coef','std_err','pvalue','n_obs' where available, or an
    error dict with key 'error'.
    """
    # legacy per-model files (older scripts)
    candidates = {
        "FE": Path(f"outputs/fe_{varname}.json"),
        "OLS": Path(f"outputs/ols_{varname}.json"),
        "ElasticNet": Path(f"outputs/elasticnet_{varname}.json"),
    }

    # first try legacy set
    loaded = {}
    for k, p in candidates.items():
        loaded[k] = safe_load_json(p)

    # If none found, try the new extractor combined file
    new_path = Path("outputs") / "variables" / varname / f"{varname}_summary.json"
    if new_path.exists():
        try:
            combined = safe_load_json(new_path)
            # map fields into expected simple dicts
            # fe_within -> FE (irrefutable within-demean result)
            if isinstance(combined, dict):
                if "fe_within" in combined:
                    loaded["FE"] = combined.get("fe_within") or {"error":"missing_fe_within"}
                elif "fe_artifact" in combined:
                    # if fe_within missing, but fe_artifact present, expose that under FE
                    loaded["FE"] = combined.get("fe_artifact") or {"error":"missing_fe_artifact"}
                if "ols" in combined:
                    loaded["OLS"] = combined.get("ols") or {"error":"missing_ols"}
                if "elasticnet" in combined:
                    loaded["ElasticNet"] = combined.get("elasticnet") or {"error":"missing_elasticnet"}
        except Exception as e:
            loaded["__loader_error"] = {"error": f"failed reading combined summary: {e}"}

    # normalize results: ensure each key is a dict (or error)
    for k in ("FE","OLS","ElasticNet"):
        if k not in loaded or loaded[k] is None:
            loaded[k] = {"error": f"{k} not found"}
    return loaded

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
    """
    Accepts either:
     - a raw per-model JSON dict with keys like 'coef','std_err','pvalue','n_obs'
     - or an error dict {'error': ...}
    Returns normalized summary with standardized effect and magnitude.
    """
    if raw_stats is None:
        return {"error": "no stats"}
    if "error" in raw_stats:
        return raw_stats
    out = {}
    # permissive extraction from different shapes
    # prefer 'coef'/'std_err' etc if present
    out["model_type"] = raw_stats.get("model_type") or raw_stats.get("model") or None
    # Some extractor outputs store coef under 'coef' (fe_within) or 'coef_raw' (fe_artifact/ols/en)
    coef = raw_stats.get("coef")
    if coef is None:
        coef = raw_stats.get("coef_raw") or raw_stats.get("estimate") or raw_stats.get("beta")
    out["coef"] = coef
    out["std_err"] = raw_stats.get("std_err") or raw_stats.get("se") or None
    out["pvalue"] = raw_stats.get("pvalue") or raw_stats.get("pval") or None
    out["n_obs"] = raw_stats.get("n_obs") or raw_stats.get("n_obs_used") or None
    # mapped info if present
    out["mapped_exog_name"] = raw_stats.get("mapped_name") or raw_stats.get("mapped_to_feature") or None
    out["mapped_index"] = raw_stats.get("mapped_index") or raw_stats.get("mapped_pos") or None

    # standardized
    try:
        out["standardized"] = compute_standardized(out["coef"], sd_var, sd_target) if out.get("coef") is not None else None
    except Exception:
        out["standardized"] = None

    # magnitude description
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
        if not isinstance(stats, dict):
            lines.append(f"- NOTE: unexpected stats format: {stats}\n")
            continue
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
    with open(path, "w", encoding="utf8") as fh:
        fh.write("\n".join(lines))

def write_csv(path: Path, varname: str, summary: dict):
    rows = []
    for model_name, s in summary.items():
        if not isinstance(s, dict) or "error" in s:
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

    # load per-model summaries (legacy or new combined extractor output)
    loaded = load_var_summary(varname)

    # compute SDs
    try:
        sd_target, sd_var = compute_sd_values(features_path, varname)
    except Exception as e:
        print("Failed to compute SDs:", e)
        raise

    # Build summary using loaded nested dicts
    summary = {}
    summary["FE"] = summarize_model_stats(loaded.get("FE", {"error":"missing"}), sd_var, sd_target)
    summary["OLS"] = summarize_model_stats(loaded.get("OLS", {"error":"missing"}), sd_var, sd_target)
    summary["ElasticNet"] = summarize_model_stats(loaded.get("ElasticNet", {"error":"missing"}), sd_var, sd_target)

    # additional ML-focused diagnostics (best-effort)
    ml_diag = {}
    try:
        en_coef = summary["ElasticNet"].get("coef")
        ols_coef = summary["OLS"].get("coef")
        if en_coef is not None and ols_coef is not None:
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
        if isinstance(s, dict) and "error" in s:
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
