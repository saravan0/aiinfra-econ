#!/usr/bin/env python3
"""
scripts/extract_variable_effects.py

Hybrid variable effect extraction (research-grade, provenance-rich).

Produces per-variable outputs:
  outputs/variables/<var>/
    - <var>_summary.json
    - <var>_summary.md
    - <var>_summary.csv

Also:
  - outputs/variables/summary_table.csv
  - outputs/variables/manifest.json

Design:
 - FE_within: within-entity demean (aligned with config/model.yml if present) — IRREFUTABLE.
 - FE_artifact: attempt to extract coefficient from saved fe artifact (if available) using careful heuristics.
 - OLS: exact-name mapping from ols_result artifact.
 - ElasticNet: inspect pipeline, unscale coef_ to raw units and map by feature name.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

try:
    import yaml
except Exception:
    yaml = None

import statsmodels.api as sm

LOG = logging.getLogger("extract_variable_effects")
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

ROOT = Path(".")
DEFAULT_FEATURES = ROOT / "data" / "processed" / "features_lean_imputed.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
OUTBASE = ROOT / "outputs" / "variables"
CONFIG_PATH = ROOT / "config" / "model.yml"


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")


def write_md(p: Path, obj: Dict[str, Any]) -> None:
    lines = []
    lines.append(f"# Variable: {obj.get('variable')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k, v in obj.items():
        if k == "notes" or k == "manifest":
            continue
        lines.append(f"- **{k}**: `{v}`")
    if obj.get("notes"):
        lines.append("")
        lines.append("## Notes")
        for n in obj["notes"]:
            lines.append(f"- {n}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf8")


def safe_load_joblib(p: Path):
    if joblib is None:
        raise RuntimeError("joblib not available")
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        LOG.warning("Failed to load joblib %s: %s", p, e)
        return None


def load_artifacts(artifacts_dir: Path):
    artifacts = {}
    fe_candidates = list(artifacts_dir.glob("fe_result*")) + list(artifacts_dir.glob("fe_model*")) + list(artifacts_dir.glob("models/fe_result*"))
    ols_candidates = list(artifacts_dir.glob("ols_result*")) + list(artifacts_dir.glob("ols_model*")) + list(artifacts_dir.glob("models/ols_result*"))
    en_candidates = list(artifacts_dir.glob("en_model*")) + list(artifacts_dir.glob("elasticnet_cv*")) + list(artifacts_dir.glob("models/en_model*")) + list(artifacts_dir.glob("models/elasticnet*"))
    artifacts['fe'] = safe_load_joblib(fe_candidates[0]) if fe_candidates else None
    artifacts['ols'] = safe_load_joblib(ols_candidates[0]) if ols_candidates else None
    artifacts['en'] = safe_load_joblib(en_candidates[0]) if en_candidates else None
    artifacts['feature_names'] = None
    if (artifacts_dir / "feature_names.json").exists():
        try:
            artifacts['feature_names'] = json.loads((artifacts_dir / "feature_names.json").read_text(encoding="utf8"))
        except Exception:
            artifacts['feature_names'] = None
    # load train_index if present (subset parity)
    artifacts['train_index'] = None
    if (artifacts_dir / "train_index.csv").exists():
        try:
            artifacts['train_index'] = pd.read_csv(artifacts_dir / "train_index.csv", header=None).iloc[:, 0].astype(int).tolist()
        except Exception:
            artifacts['train_index'] = None
    return artifacts


def load_baseline_predictors(cfg_path: Path) -> Optional[List[str]]:
    if not cfg_path.exists() or yaml is None:
        return None
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf8"))
        baseline = (cfg.get("predictors", {}) .get("baseline") or []) + (cfg.get("predictors", {}).get("extra_controls") or [])
        out = []
        for b in baseline:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                for k in ("predictors", "features", "name", "term", "predictor"):
                    if k in b and isinstance(b[k], str):
                        out.append(b[k]); break
        return out
    except Exception:
        return None


def compute_within_demean(df: pd.DataFrame, group: str, target: str, var: str) -> Dict[str, Any]:
    cols = [group, target, var]
    sub = df[cols].dropna().copy()
    if sub.empty:
        raise ValueError("No rows after dropna for within-demean")
    sub["y_w"] = sub[target] - sub.groupby(group)[target].transform("mean")
    sub["x_w"] = sub[var] - sub.groupby(group)[var].transform("mean")
    X = sm.add_constant(sub["x_w"], has_constant="add")
    res = sm.OLS(sub["y_w"], X).fit()
    sd_target = float(sub[target].std(ddof=0))
    sd_var = float(sub[var].std(ddof=0))
    coef = float(res.params.iloc[1])
    std_err = float(res.bse.iloc[1]) if hasattr(res, "bse") else None
    pval = float(res.pvalues.iloc[1]) if hasattr(res, "pvalues") else None
    std_effect = coef * sd_var / (sd_target if sd_target != 0 else 1.0)
    return {
        "n_obs": int(len(sub)),
        "coef": coef,
        "std_err": std_err,
        "pvalue": pval,
        "sd_target": sd_target,
        "sd_var": sd_var,
        "standardized_effect": std_effect
    }


def map_ols_coef(ols_obj, var: str) -> Dict[str, Any]:
    out = {"present": False}
    if ols_obj is None:
        out["notes"] = ["ols_artifact_missing"]
        return out

    try:
        # locate params Series in multiple artifact shapes
        params = None
        bse = None
        pvalues = None

        # direct statsmodels results object
        if hasattr(ols_obj, "params"):
            params = getattr(ols_obj, "params")
            bse = getattr(ols_obj, "bse", None)
            pvalues = getattr(ols_obj, "pvalues", None)

        # dict-like wrapper with 'fitted' or 'result'
        if params is None and isinstance(ols_obj, dict):
            for key in ("fitted", "result", "model", "estimator"):
                candidate = ols_obj.get(key)
                if candidate is not None and hasattr(candidate, "params"):
                    params = getattr(candidate, "params")
                    bse = getattr(candidate, "bse", None)
                    pvalues = getattr(candidate, "pvalues", None)
                    break

        # fallback: attribute .fitted
        if params is None and hasattr(getattr(ols_obj, "fitted", None), "params"):
            params = getattr(ols_obj.fitted, "params")
            bse = getattr(ols_obj.fitted, "bse", None)
            pvalues = getattr(ols_obj.fitted, "pvalues", None)

        if params is None:
            out["notes"] = ["could_not_read_ols_params"]
            return out

        # If params is DataFrame with single column, convert to Series
        import pandas as _pd, numpy as _np
        if isinstance(params, _pd.DataFrame):
            if params.shape[1] == 1:
                params = params.iloc[:, 0]
            else:
                out["notes"] = ["ols_params_is_dataframe_multi_column"]
                return out

        # If params is ndarray-like, try to read param names from .index-like metadata if available
        if isinstance(params, (list, _np.ndarray)):
            # we can't map raw ndarray without names -> abort
            out["notes"] = ["ols_params_is_ndarray_no_names"]
            return out

        # At this point params should be Series-like with .index
        if not hasattr(params, "index"):
            out["notes"] = ["ols_params_missing_index"]
            return out

        # Exact match
        if var in params.index:
            coef = float(params[var]) if params.get(var) is not None else float("nan")
            se = None
            pv = None
            try:
                if bse is not None and hasattr(bse, "get"):
                    se = float(bse.get(var)) if bse.get(var) is not None else None
                if pvalues is not None and hasattr(pvalues, "get"):
                    pv = float(pvalues.get(var)) if pvalues.get(var) is not None else None
            except Exception:
                se = None
                pv = None
            out.update({"present": True, "mapped_name": var, "coef_raw": coef, "se": se, "pvalue": pv})
            return out

        # Substring / case-insensitive fallback
        candidates = [n for n in list(params.index) if var.lower() in str(n).lower()]
        if candidates:
            name = candidates[0]
            out.update({"present": True, "mapped_name": name, "coef_raw": float(params[name])})
            return out

        out["notes"] = ["no_matching_param_name_in_ols"]
        return out

    except Exception as e:
        out["notes"] = [f"ols_map_error: {e}"]
        return out




def _load_fe_param_map(artifacts_dir: Path):
    try:
        p = artifacts_dir / "fe_param_map.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf8"))
    except Exception:
        pass
    return None


def map_fe_artifact(fe_obj, var: str, feat_names: Optional[List[str]], artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Robust FE artifact to variable mapping. Returns dict with detailed notes and mapping_method.
    """
    out = {"present": False, "notes": []}
    try:
        params = None
        param_names = None

        if fe_obj is None:
            out["notes"].append("fe_artifact_missing")
            return out

        # Unwrap possible dict wrapper
        if isinstance(fe_obj, dict):
            if "params" in fe_obj:
                params = fe_obj["params"]
            elif "fitted" in fe_obj and hasattr(fe_obj["fitted"], "params"):
                params = fe_obj["fitted"].params
            elif "result" in fe_obj and hasattr(fe_obj["result"], "params"):
                params = fe_obj["result"].params
            else:
                for k in ("fitted", "result", "model"):
                    candidate = fe_obj.get(k)
                    if candidate is not None and hasattr(candidate, "params"):
                        params = getattr(candidate, "params")
                        break
        else:
            if hasattr(fe_obj, "params"):
                params = getattr(fe_obj, "params")
            elif hasattr(fe_obj, "fitted") and hasattr(fe_obj.fitted, "params"):
                params = getattr(fe_obj.fitted, "params")
            elif hasattr(fe_obj, "result") and hasattr(fe_obj.result, "params"):
                params = getattr(fe_obj.result, "params")

        if params is not None and hasattr(params, "index"):
            param_names = list(params.index)
            if var in param_names:
                idx = param_names.index(var)
                val = float(params[var])
                out.update({"present": True, "mapped_name": var, "mapped_index": idx, "coef_raw": val, "mapping_method": "exact_param_name"})
                return out
            # case-insensitive exact or substring
            for i, pn in enumerate(param_names):
                if var.lower() == str(pn).lower():
                    out.update({"present": True, "mapped_name": pn, "mapped_index": i, "coef_raw": float(params[pn]), "mapping_method": "case_insensitive_param_name"})
                    return out
            for i, pn in enumerate(param_names):
                if var.lower() in str(pn).lower():
                    out.update({"present": True, "mapped_name": pn, "mapped_index": i, "coef_raw": float(params[pn]), "mapping_method": "substring_param_name"})
                    return out
            # heuristic: xN mapping using fe_param_map or feat_names
            simple_names = [str(n).lower() for n in param_names[: min(50, len(param_names))]]
            if any(n.startswith("x") and n[1:].isdigit() for n in simple_names) or any(n.startswith("fe_") for n in simple_names):
                fe_map = _load_fe_param_map(Path(artifacts_dir)) if artifacts_dir is not None else None
                if fe_map:
                    for pn, mapped in fe_map.items():
                        if mapped == var and pn in param_names:
                            val = float(params[pn]) if pn in params else None
                            out.update({"present": True, "mapped_name": pn, "mapped_index": param_names.index(pn), "coef_raw": val, "mapping_method": "fe_param_map"})
                            return out
                if feat_names and "const" in param_names:
                    const_pos = param_names.index("const")
                    if var in feat_names:
                        feat_idx = feat_names.index(var)
                        param_idx = const_pos + 1 + feat_idx
                        if param_idx < len(param_names):
                            pn = param_names[param_idx]
                            val = float(params[pn])
                            out.update({"present": True, "mapped_name": pn, "mapped_index": param_idx, "coef_raw": val, "mapping_method": "heuristic_index_via_feature_names"})
                            return out

        # ndarray-like params
        if params is not None and (isinstance(params, (list, tuple, np.ndarray)) or (hasattr(params, "shape") and not hasattr(params, "index"))):
            arr = np.asarray(params).ravel()
            fe_map = _load_fe_param_map(Path(artifacts_dir)) if artifacts_dir is not None else None
            if fe_map:
                for pn, mapped in fe_map.items():
                    if mapped == var:
                        idx = None
                        if pn == "const":
                            idx = 0
                        elif pn.lower().startswith("x") and pn[1:].isdigit():
                            idx = int(pn[1:])
                        else:
                            import re
                            m = re.search(r"(\d+)$", pn)
                            if m:
                                idx = int(m.group(1))
                        if idx is not None and idx < arr.size:
                            out.update({"present": True, "mapped_name": pn, "mapped_index": idx, "coef_raw": float(arr[idx]), "mapping_method": "fe_param_map_positional"})
                            return out
            if feat_names:
                if var in feat_names and arr.size >= 1 + len(feat_names):
                    feat_idx = feat_names.index(var)
                    param_idx = 1 + feat_idx
                    if param_idx < arr.size:
                        out.update({"present": True, "mapped_name": var, "mapped_index": param_idx, "coef_raw": float(arr[param_idx]), "mapping_method": "positional_via_feature_names"})
                        return out

        out["notes"].append("could_not_map_feature_to_fe_params_safely")
        return out

    except Exception as e:
        out["notes"].append(f"fe_map_error: {repr(e)}")
        return out


def map_en_coef(en_obj, feat_names: Optional[List[str]], var: str) -> Dict[str, Any]:
    out = {"present": False}
    if en_obj is None:
        out["notes"] = ["elasticnet_artifact_missing"]
        return out
    try:
        if hasattr(en_obj, "named_steps"):
            steps = en_obj.named_steps
            scaler = None
            en_est = None
            for k, v in steps.items():
                clsname = type(v).__name__.lower()
                if "standardscaler" in clsname:
                    scaler = v
                if "elasticnet" in clsname:
                    en_est = v
            coef_scaled = getattr(en_est, "coef_", None)
            scale_ = getattr(scaler, "scale_", None)
            if feat_names is None:
                feat_names = getattr(en_obj, "feature_names_in_", None) or feat_names
            if coef_scaled is not None and scale_ is not None and feat_names:
                coef_raws = np.asarray(coef_scaled) / np.asarray(scale_)
                if var in feat_names:
                    idx = feat_names.index(var)
                    coef_scaled_val = float(coef_scaled[idx])
                    coef_raw_val = float(coef_raws[idx])
                    out.update({"present": True, "mapped_name": var, "mapped_index": idx,
                                "coef_scaled": coef_scaled_val, "coef_raw": coef_raw_val})
                    return out
                else:
                    for i, n in enumerate(feat_names):
                        if var.lower() in n.lower():
                            out.update({"present": True, "mapped_name": n, "mapped_index": i,
                                        "coef_scaled": float(coef_scaled[i]), "coef_raw": float(coef_raws[i])})
                            return out
            if hasattr(en_est, "coef_"):
                coef_arr = getattr(en_est, "coef_")
                if feat_names and len(coef_arr) == len(feat_names):
                    for i, n in enumerate(feat_names):
                        if n == var:
                            out.update({"present": True, "mapped_name": var, "mapped_index": i,
                                        "coef_scaled": float(coef_arr[i]), "coef_raw": None})
                            return out
            out["notes"] = ["could_not_map_elasticnet_cleanly"]
            return out
        else:
            coef = getattr(en_obj, "coef_", None)
            if coef is not None and feat_names:
                if var in feat_names:
                    idx = feat_names.index(var)
                    out.update({"present": True, "mapped_name": var, "mapped_index": idx, "coef_scaled": float(coef[idx])})
                    return out
            out["notes"] = ["elasticnet_object_unrecognized"]
            return out
    except Exception as e:
        out["notes"] = [f"elasticnet_map_error: {e}"]
        return out


def to_csv_row(var: str, fe_within: Optional[Dict], fe_art: Optional[Dict], ols_m: Optional[Dict], en_m: Optional[Dict]):
    return {
        "variable": var,
        "fe_within_coef": fe_within.get("coef") if fe_within else None,
        "fe_within_std_effect": fe_within.get("standardized_effect") if fe_within else None,
        "fe_artifact_coef": fe_art.get("coef_raw") if fe_art else None,
        "fe_artifact_mapped_name": fe_art.get("mapped_name") if fe_art else None,
        "ols_coef": ols_m.get("coef_raw") if ols_m else None,
        "ols_se": ols_m.get("se") if ols_m else None,
        "ols_pvalue": ols_m.get("pvalue") if ols_m else None,
        "en_coef_raw": en_m.get("coef_raw") if en_m else None,
        "en_coef_scaled": en_m.get("coef_scaled") if en_m else None,
        "notes": ";".join((fe_within.get("notes",[]) if isinstance(fe_within,dict) and fe_within.get("notes") else [] ) + (fe_art.get("notes",[]) if fe_art else []) + (ols_m.get("notes",[]) if ols_m and ols_m.get("notes") else []) + (en_m.get("notes",[]) if en_m and en_m.get("notes") else []))
    }


def main():
    p = argparse.ArgumentParser(prog="extract_variable_effects")
    p.add_argument("--vars", nargs="+", required=True, help="Variable names to extract")
    p.add_argument("--features", default=str(DEFAULT_FEATURES), help="Features CSV")
    p.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR), help="Artifacts directory")
    p.add_argument("--outdir", default=str(OUTBASE), help="Outputs base dir")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = p.parse_args()

    feat_path = Path(args.features)
    artifacts_dir = Path(args.artifacts_dir)
    out_base = Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)

    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}")
    LOG.info("Loading features: %s", feat_path)
    df = pd.read_csv(feat_path, low_memory=False)
    artifacts = load_artifacts(artifacts_dir)
    LOG.info("Artifacts found: FE=%s  OLS=%s  EN=%s", bool(artifacts.get("fe")), bool(artifacts.get("ols")), bool(artifacts.get("en")))

    feat_names = artifacts.get("feature_names") or list(df.columns)
    baseline_preds = load_baseline_predictors(Path(CONFIG_PATH)) or None
    LOG.info("Baseline predictors from config: %s", baseline_preds)

    rows = []
    manifest = {
        "script": str(Path(__file__).resolve()),
        "features": str(feat_path.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "variables": args.vars,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat() + "Z"
    }

    # apply train index subset if available (ensures SD parity)
    train_index = artifacts.get("train_index")
    if train_index is not None:
        try:
            LOG.info("Applying train_index subset of length %d to features (to ensure SD/sample parity).", len(train_index))
            df = df.loc[train_index].reset_index(drop=True)
        except Exception:
            LOG.debug("Could not apply train_index; continuing with full df.")

    for var in args.vars:
        LOG.info("Processing variable: %s", var)
        var_out_dir = out_base / var
        if var_out_dir.exists() and not args.force:
            LOG.info("Skipping %s (exists). Use --force to overwrite.", var_out_dir)
            continue

        var_out_dir.mkdir(parents=True, exist_ok=True)
        notes: List[str] = []

        # 1) FE within-demean (aligned if baseline_preds available)
        try:
            if baseline_preds:
                req_cols = list({"iso3", "gdp_growth_pct"} | set(baseline_preds) | {var})
                if not set(req_cols).issubset(set(df.columns)):
                    notes.append("align_requested_but_missing_columns")
                    fe_within = compute_within_demean(df, "iso3", "gdp_growth_pct", var)
                else:
                    sub_df = df[req_cols].dropna().copy()
                    if sub_df.empty:
                        raise ValueError("No rows after aligning with baseline predictors")
                    sub_df["y_w"] = sub_df["gdp_growth_pct"] - sub_df.groupby("iso3")["gdp_growth_pct"].transform("mean")
                    sub_df["x_w"] = sub_df[var] - sub_df.groupby("iso3")[var].transform("mean")
                    X = sm.add_constant(sub_df["x_w"], has_constant="add")
                    res = sm.OLS(sub_df["y_w"], X).fit()
                    sd_target = float(sub_df["gdp_growth_pct"].std(ddof=0))
                    sd_var = float(sub_df[var].std(ddof=0))
                    coef = float(res.params.iloc[1])
                    std_err = float(res.bse.iloc[1]) if hasattr(res, "bse") else None
                    pval = float(res.pvalues.iloc[1]) if hasattr(res, "pvalues") else None
                    std_effect = coef * sd_var / (sd_target if sd_target != 0 else 1.0)
                    fe_within = {"n_obs": int(len(sub_df)), "coef": coef, "std_err": std_err, "pvalue": pval, "sd_target": sd_target, "sd_var": sd_var, "standardized_effect": std_effect}
            else:
                fe_within = compute_within_demean(df, "iso3", "gdp_growth_pct", var)
        except Exception as e:
            LOG.warning("FE within-demean failed for %s: %s", var, e)
            fe_within = {"notes": [f"fe_within_fail: {e}"]}

        # 2) FE artifact mapping (attempt) — pass artifacts_dir for fe_param_map lookup
        fe_art = map_fe_artifact(artifacts.get("fe"), var, feat_names, artifacts_dir=artifacts_dir)

        # 3) OLS mapping
        ols_m = map_ols_coef(artifacts.get("ols"), var)

        # 4) ElasticNet mapping/unscale
        en_m = map_en_coef(artifacts.get("en"), feat_names, var)

        summary = {
            "variable": var,
            "sd_var": fe_within.get("sd_var") if isinstance(fe_within, dict) else None,
            "sd_target": fe_within.get("sd_target") if isinstance(fe_within, dict) else None,
            "extracted_at_utc": pd.Timestamp.utcnow().isoformat() + "Z",
            "fe_within": fe_within,
            "fe_artifact": fe_art,
            "ols": ols_m,
            "elasticnet": en_m,
            "notes": notes
        }
        write_json(var_out_dir / f"{var}_summary.json", summary)
        write_md(var_out_dir / f"{var}_summary.md", summary)
        csv_row = to_csv_row(var, fe_within, fe_art, ols_m, en_m)
        pd.DataFrame([csv_row]).to_csv(var_out_dir / f"{var}_summary.csv", index=False)
        rows.append(csv_row)
        LOG.info("Wrote outputs for %s -> %s", var, var_out_dir)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(out_base / "summary_table.csv", index=False)
    write_json(out_base / "manifest.json", manifest)
    LOG.info("Wrote summary_table.csv and manifest to %s", out_base.resolve())
    print("Done. Variable outputs in:", out_base.resolve())


if __name__ == "__main__":
    main()
