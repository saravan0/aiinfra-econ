#!/usr/bin/env python3
"""
Deterministic baseline snapshot generator (final, reviewer-grade).

Usage:
  python scripts/generate_baseline_snapshot.py --config config/snapshot_config.json

This version:
 - extracts FE/OLS-style coefs from reports/model_table.csv (format: model,term,coef,...)
 - extracts ElasticNetCV coefs from reports/plot_elasticnet_paths/files/en_cv_selected_coefs.csv
 - maps rolling RMSE per-variable by column name matching
 - copies standardized / shap / lowess / provenance artifacts
 - writes baseline_<var>.json, generate_baseline_snapshot.json, model_comparison_table.csv, generate_baseline_snapshot.md, meta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(".").resolve()
OUT_BASE = ROOT / "reports" / "generate_baseline_snapshot"
OUT_FILES = OUT_BASE / "files"
OUT_STD = OUT_FILES / "standardized"
OUT_PROV = OUT_BASE / "provenance"
META_PATH = OUT_BASE / "meta.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(p: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def copy_file(src: Path, dst: Path) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": str(dst), "sha256": sha256(dst), "size": dst.stat().st_size}


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf8"))
    except Exception as e:
        print("ERROR reading JSON", p, e, file=sys.stderr)
        return None


def read_csv_df(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print("ERROR reading CSV", p, e, file=sys.stderr)
        return None


# ---------------- parsing model_table.csv & en_cv_selected_coefs.csv ----------------


def parse_model_table(model_table_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Parse reports/model_table.csv (format: model,term,coef,...) and return mapping:
      coeffs[model][term] = coef
    """
    df = read_csv_df(model_table_path)
    if df is None:
        return {}
    mapping: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        try:
            model = str(row.get("model", "")).strip()
            term = str(row.get("term", "")).strip()
            coef = row.get("coef")
            if pd.isna(coef):
                continue
            mapping.setdefault(model, {})[term] = float(coef)
        except Exception:
            continue
    return mapping


def parse_en_cv_coefs(en_cv_path: Path) -> Dict[str, float]:
    """
    Parse en_cv_selected_coefs.csv with columns: feature,coef_at_best_alpha
    """
    df = read_csv_df(en_cv_path)
    out: Dict[str, float] = {}
    if df is None:
        return out
    # normalize column names
    cols = [c.lower() for c in df.columns]
    feat_col = None
    coef_col = None
    for c in df.columns:
        lc = c.lower()
        if "feature" == lc or "term" == lc or "name" in lc:
            feat_col = c
        if "coef" in lc:
            coef_col = c
    if feat_col is None:
        feat_col = df.columns[0]
    if coef_col is None and df.shape[1] >= 2:
        coef_col = df.columns[1]
    if coef_col is None:
        return out
    for _, r in df.iterrows():
        f = str(r[feat_col]).strip()
        try:
            coef = float(r[coef_col])
            out[f] = coef
        except Exception:
            continue
    return out


# ---------------- rolling RMSE extraction ----------------


def extract_rmse_from_global_csv(p: Path, variable: str) -> Optional[float]:
    """
    If rolling csv is a wide table, try to find a column containing the variable token (case-insensitive).
    If found, return mean of that column. Otherwise try any column with 'rmse' in name.
    """
    df = read_csv_df(p)
    if df is None:
        return None
    var_token = variable.lower().replace(" ", "_")
    # find columns that contain var_token
    for col in df.columns:
        if var_token in str(col).lower().replace(" ", "_"):
            try:
                s = pd.to_numeric(df[col], errors="coerce")
                if not s.dropna().empty:
                    return float(s.mean())
            except Exception:
                pass
    # fallback: any column with rmse in name
    for col in df.columns:
        if "rmse" in str(col).lower():
            try:
                s = pd.to_numeric(df[col], errors="coerce")
                if not s.dropna().empty:
                    return float(s.mean())
            except Exception:
                pass
    return None


# ---------------- build baseline per variable ----------------


def build_baseline(
    var: str,
    cfg: Dict[str, Any],
    model_table_map: Dict[str, Dict[str, float]],
    en_cv_map: Dict[str, float],
) -> Dict[str, Any]:
    baseline = {
        "variable": var,
        "generated_at_utc": now_iso(),
        "provenance": {},
        "metrics": {},
        "notes": [],
    }
    vcfg = cfg["variables"].get(var, {})

    # copy standardized JSON if present
    std_path = (
        Path(vcfg.get("standardized_json")) if vcfg.get("standardized_json") else None
    )
    if std_path and std_path.exists():
        dst = OUT_STD / std_path.name
        baseline["provenance"]["standardized_json"] = copy_file(std_path, dst)
        sj = load_json(dst)
        if sj:
            baseline["metrics"]["sd_var"] = sj.get("sd_var")
            baseline["metrics"]["sd_target"] = sj.get("sd_target")
            if isinstance(sj.get("summary"), dict):
                baseline["metrics"]["standardized_summary_keys"] = list(
                    sj["summary"].keys()
                )
    else:
        baseline["notes"].append("missing_standardized_json")

    # copy lowess and shap if present
    for key in ("lowess_json", "shap_json"):
        pth = Path(vcfg.get(key)) if vcfg.get(key) else None
        if pth and pth.exists():
            dst = OUT_FILES / f"{key}_{var}{pth.suffix}"
            baseline["provenance"][key] = copy_file(pth, dst)
            if key == "shap_json":
                sj = load_json(dst)
                # extract mean shap
                if sj:
                    shap_mean = None
                    for k in (
                        "mean_abs_shap",
                        "mean(|shap|)",
                        "mean_abs_value",
                        "mean_shap",
                    ):
                        if k in sj and isinstance(sj[k], (int, float)):
                            shap_mean = float(sj[k])
                            break
                    if shap_mean is None and isinstance(sj.get("summary"), dict):
                        for k in ("mean_abs_shap", "mean(|shap|)"):
                            if k in sj["summary"]:
                                shap_mean = float(sj["summary"][k])
                                break
                    baseline["metrics"]["shap_mean_abs"] = shap_mean
        else:
            baseline["provenance"][key] = None

    # Rolling RMSEs (h1/h3) - use per-variable provided path else global provenance path
    for hkey in ("rolling_h1_csv", "rolling_h3_csv"):
        # prefer per-variable if provided
        p_local = Path(vcfg.get(hkey)) if vcfg.get(hkey) else None
        p_global = (
            Path(cfg.get("provenance", {}).get(hkey))
            if cfg.get("provenance", {})
            else None
        )
        chosen = None
        if p_local and p_local.exists():
            chosen = p_local
        elif p_global and p_global.exists():
            chosen = p_global
        if chosen:
            dst = OUT_PROV / chosen.name
            prov_info = copy_file(
                chosen, dst
            )  # prov_info usually a dict {"path": "...", ...}
            baseline["provenance"][hkey] = prov_info

            # Normalize to a Path object regardless of prov_info type
            if isinstance(prov_info, dict):
                path_str = prov_info.get("path")
            else:
                # prov_info could be a Path or a plain string in some environments
                path_str = str(prov_info)

            if path_str:
                rmse_mean = extract_rmse_from_global_csv(Path(path_str), var)
            else:
                rmse_mean = None

            baseline["metrics"][
                f"rmse_{'h1' if hkey.endswith('h1_csv') else 'h3'}"
            ] = rmse_mean

        else:
            baseline["provenance"][hkey] = None
            baseline["metrics"][
                f"rmse_{'h1' if hkey.endswith('h1_csv') else 'h3'}"
            ] = None
            baseline["notes"].append(f"missing_{hkey}")

    # extract en_coef from en_cv_map (prefer exact key match, else lowercase match)
    en_coef = None
    if var in en_cv_map:
        en_coef = en_cv_map[var]
    else:
        # try fuzzy key match (lowercase)
        for k, v in en_cv_map.items():
            if k.lower() == var.lower():
                en_coef = v
                break
    baseline["metrics"]["en_coef"] = en_coef

    # extract OLS/FE/DriscollKraay coefficients from model_table_map using priority
    def find_coef_priority(term: str) -> Dict[str, Optional[float]]:
        res = {"FE": None, "DriscollKraay": None, "OLS": None, "RandomEffects": None}
        for model_name in list(res.keys()):
            model_dict = model_table_map.get(model_name, {})
            if term in model_dict:
                res[model_name] = model_dict[term]
        # pick priority
        for priority in ("FE", "DriscollKraay", "OLS", "RandomEffects"):
            if res.get(priority) is not None:
                return {"source": priority, "coef": float(res[priority])}
        return {"source": None, "coef": None}

    coef_entry = find_coef_priority(var)
    baseline["metrics"]["fe_source"] = coef_entry["source"]
    baseline["metrics"]["fe_coef"] = coef_entry["coef"]

    # if OLS exists, record separately
    ols_val = None
    if "OLS" in model_table_map and var in model_table_map["OLS"]:
        ols_val = model_table_map["OLS"][var]
    baseline["metrics"]["ols_coef"] = ols_val

    # sd & n from standardized copied earlier already
    return baseline


# ---------------- main ----------------


def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print("Config not found:", cfg_path, file=sys.stderr)
        sys.exit(2)
    cfg = json.loads(cfg_path.read_text(encoding="utf8"))

    # outputs
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    OUT_FILES.mkdir(parents=True, exist_ok=True)
    OUT_STD.mkdir(parents=True, exist_ok=True)
    OUT_PROV.mkdir(parents=True, exist_ok=True)

    # parse model_table.csv
    model_table_path = Path(
        cfg.get("extra_files", {}).get("model_table_csv", "reports/model_table.csv")
    )
    model_table_map = {}
    if model_table_path.exists():
        model_table_map = parse_model_table(model_table_path)
    else:
        print(
            "WARNING: model_table.csv not found at", model_table_path, file=sys.stderr
        )

    # parse en_cv_selected_coefs
    en_cv_path = Path(
        cfg.get("extra_files", {}).get(
            "en_cv_selected_coefs",
            "reports/plot_elasticnet_paths/files/en_cv_selected_coefs.csv",
        )
    )
    en_cv_map = {}
    if en_cv_path.exists():
        en_cv_map = parse_en_cv_coefs(en_cv_path)
    else:
        print(
            "WARNING: en_cv_selected_coefs.csv not found at",
            en_cv_path,
            file=sys.stderr,
        )

    # copy global extras into OUT_BASE root if provided
    extras = cfg.get("extra_files", {})
    for name, path in extras.items():
        if not path:
            continue
        p = Path(path)
        if p.exists():
            copy_file(p, OUT_BASE / p.name)

    # copy provenance artifacts declared in config.provenance
    prov_cfg = cfg.get("provenance", {})
    prov_copies = {}
    for k, pstr in prov_cfg.items():
        if not pstr:
            continue
        p = Path(pstr)
        if p.exists():
            prov_copies[k] = copy_file(p, OUT_PROV / p.name)
        else:
            print(f"WARNING provenance missing {k}: {p}", file=sys.stderr)

    # Build baseline entries
    master_entries = []
    rows_for_csv = []
    variables = list(cfg.get("variables", {}).keys())
    for var in variables:
        b = build_baseline(var, cfg, model_table_map, en_cv_map)
        master_entries.append(b)
        row = {
            "variable": var,
            "fe_coef": b["metrics"].get("fe_coef"),
            "fe_source": b["metrics"].get("fe_source"),
            "ols_coef": b["metrics"].get("ols_coef"),
            "en_coef": b["metrics"].get("en_coef"),
            "shap_mean_abs": b["metrics"].get("shap_mean_abs"),
            "nonlinearity_strength": b["metrics"].get("nonlinearity_strength"),
            "turning_points": None,
            "rmse_h1": b["metrics"].get("rmse_h1"),
            "rmse_h3": b["metrics"].get("rmse_h3"),
            "rmse_shock": b["metrics"].get("rmse_shock"),
            "rmse_nonshock": b["metrics"].get("rmse_nonshock"),
        }
        rows_for_csv.append(row)
        # write per-variable baseline JSON
        outp = OUT_BASE / f"baseline_{var}.json"
        outp.write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding="utf8")

    # write master json
    master_path = OUT_BASE / "generate_baseline_snapshot.json"
    master_obj = {"entries": master_entries, "generated_at_utc": now_iso()}
    master_path.write_text(
        json.dumps(master_obj, indent=2, ensure_ascii=False), encoding="utf8"
    )

    # write model_comparison_table.csv
    df = pd.DataFrame(rows_for_csv)
    df_cols = [
        "variable",
        "fe_coef",
        "fe_source",
        "ols_coef",
        "en_coef",
        "shap_mean_abs",
        "nonlinearity_strength",
        "turning_points",
        "rmse_h1",
        "rmse_h3",
        "rmse_shock",
        "rmse_nonshock",
    ]
    for c in df_cols:
        if c not in df.columns:
            df[c] = None
    df = df[df_cols]
    csv_path = OUT_BASE / "model_comparison_table.csv"
    df.to_csv(csv_path, index=False)

    # write markdown summary
    md_lines = [f"# Baseline snapshot\nGenerated: {now_iso()}\n"]
    for e in master_entries:
        md_lines.append(f"## {e['variable']}\n")
        md_lines.append(f"- fe_source: {e['metrics'].get('fe_source')}\n")
        md_lines.append(f"- fe_coef: {e['metrics'].get('fe_coef')}\n")
        md_lines.append(f"- ols_coef: {e['metrics'].get('ols_coef')}\n")
        md_lines.append(f"- en_coef: {e['metrics'].get('en_coef')}\n")
        md_lines.append(f"- shap_mean_abs: {e['metrics'].get('shap_mean_abs')}\n")
        md_lines.append(f"- rmse_h1: {e['metrics'].get('rmse_h1')}\n")
        md_lines.append(f"- rmse_h3: {e['metrics'].get('rmse_h3')}\n")
        md_lines.append("\n")
    md_path = OUT_BASE / "generate_baseline_snapshot.md"
    md_path.write_text("\n".join(md_lines), encoding="utf8")

    # meta
    produced = [str(p.resolve()) for p in OUT_BASE.rglob("*") if p.is_file()]
    meta = {
        "script": str(Path(__file__).resolve()),
        "generated_at_utc": now_iso(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "requirements_hash": (
            sha256(Path("requirements.txt"))
            if Path("requirements.txt").exists()
            else None
        ),
        "features_sha256": (
            sha256(Path(cfg.get("features_csv")))
            if cfg.get("features_csv") and Path(cfg.get("features_csv")).exists()
            else None
        ),
        "provenance_files": prov_copies,
        "produced_files": produced,
        "elapsed_seconds": round(time.time() - start_ts, 2),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf8")

    print("Baseline snapshot written to:", OUT_BASE)
    print(" -", master_path)
    print(" -", csv_path)
    print(" -", md_path)
    print(" -", META_PATH)


if __name__ == "__main__":
    start_ts = time.time()
    main()
