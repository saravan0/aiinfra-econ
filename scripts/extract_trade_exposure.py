"""
scripts/extract_trade_exposure.py

Usage:
    python scripts/extract_trade_exposure.py --var trade_exposure
    python scripts/extract_trade_exposure.py --var trade_exposure --artifacts artifacts/ --out outputs/

Expectations:
- Your training pipeline should save fitted model objects to an 'artifacts/' folder:
    - FE result (statsmodels / linearmodels) -> artifacts/fe_result.pkl
    - OLS result (statsmodels OLSResult) -> artifacts/ols_result.pkl
    - ElasticNet / sklearn estimator -> artifacts/en_model.joblib or artifacts/en_model.pkl
    - If using sklearn ElasticNet, also save `feature_names.json` listing features in order.
- This script is defensive: if an artifact isn't present or fails to load, it will record an error entry in outputs/.
"""
import argparse
import json
from pathlib import Path
import pickle
import joblib
import numpy as np
import sys
import traceback

OUTDIR_DEFAULT = Path("outputs")
ARTIFACTS_DEFAULT = Path("artifacts")

def try_load(path: Path):
    """Try a range of loaders for a given path."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    # Try joblib first
    try:
        return joblib.load(path)
    except Exception:
        pass
    # Try pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    # Not loaded
    raise RuntimeError(f"Could not load artifact at {path} with joblib/pickle")

def safe_conf_int(res):
    try:
        ci = res.conf_int()
        # ci may be numpy array or DataFrame-like
        if hasattr(ci, "index"):
            return {str(k): [float(ci.loc[k, 0]), float(ci.loc[k, 1])] for k in ci.index}
        else:
            # fallback: numeric index
            return {str(i): [float(ci[i, 0]), float(ci[i, 1])] for i in range(ci.shape[0])}
    except Exception:
        return {}

def extract_stats_statsmodels(res, varname):
    """
    Robust extractor for statsmodels / linearmodels results.
    Handles:
      - pandas Series-like res.params with index
      - numpy ndarray res.params (no index) paired with res.model.exog_names or res.model.data.param_names
      - When res.model.exog_names are generic 'x1','x2',... this will attempt to load
        artifacts/feature_names.json and map x1->feature_names[0], x2->feature_names[1], etc.
    Returns dict with coef, std_err, pvalue, conf_int, n_obs, model_type or error.
    """
    out = {}
    try:
        # Basic attr pulls
        params = getattr(res, "params", None)
        bse = getattr(res, "bse", None)
        pvalues = getattr(res, "pvalues", None)
        nobs = getattr(res, "nobs", None)

        # conf_int (best-effort)
        ci = {}
        try:
            raw_ci = res.conf_int()
            if hasattr(raw_ci, "index"):
                ci = {str(k): [float(raw_ci.loc[k, 0]), float(raw_ci.loc[k, 1])] for k in raw_ci.index}
            else:
                for i in range(raw_ci.shape[0]):
                    ci[str(i)] = [float(raw_ci[i, 0]), float(raw_ci[i, 1])]
        except Exception:
            ci = {}

        # Case 1: pandas-like params (named)
        if params is not None and hasattr(params, "index"):
            if varname not in params.index:
                return {"error": f"variable '{varname}' not found in params"}
            return {
                "coef": float(params[varname]),
                "std_err": float(bse[varname]) if (bse is not None and varname in bse) else None,
                "pvalue": float(pvalues[varname]) if (pvalues is not None and varname in pvalues) else None,
                "conf_int": ci.get(varname, []),
                "n_obs": int(nobs) if nobs is not None else None,
                "model_type": type(res).__name__,
            }

        # Case 2: params is ndarray-like or unlabeled; attempt positional mapping
        model_obj = getattr(res, "model", None)
        exog_names = None
        try:
            if model_obj is not None:
                if hasattr(model_obj, "exog_names"):
                    exog_names = list(model_obj.exog_names)
                elif hasattr(model_obj, "data") and hasattr(model_obj.data, "param_names"):
                    exog_names = list(model_obj.data.param_names)
        except Exception:
            exog_names = None

        import numpy as _np
        try:
            arr = _np.asarray(params)
        except Exception:
            arr = None

        if arr is None or arr.size == 0:
            return {"error": f"res.params is empty or unconvertible; could not find '{varname}'"}

        # If exog_names present, check if they are generic like 'x1','x2'
        def is_generic_xnames(names):
            if not names:
                return False
            # check majority start with 'x' followed by digits or are 'const'
            cnt = 0
            total = 0
            for nm in names:
                total += 1
                if isinstance(nm, str) and (nm == "const" or (nm.startswith("x") and nm[1:].isdigit())):
                    cnt += 1
            return (cnt / float(total)) > 0.6  # heuristic

        # Attempt to load artifacts/feature_names.json if needed
        feature_names = None
        try:
            feat_path = Path("artifacts") / "feature_names.json"
            if feat_path.exists():
                import json
                with open(feat_path, "r", encoding="utf8") as fh:
                    feature_names = json.load(fh)
        except Exception:
            feature_names = None

        # If exog_names are generic and we have feature_names, build mapping x1->feature_names[0], etc.
        if exog_names and is_generic_xnames(exog_names) and feature_names:
            # Build mapping for exog_names -> human names
            # Find index in exog_names where data columns start (skip 'const' if present)
            mapped = {}
            # assume first after 'const' correspond to feature_names in order until len(feature_names)
            start_idx = 1 if exog_names and exog_names[0] == "const" else 0
            for i, feat in enumerate(feature_names):
                exog_i = start_idx + i
                if exog_i < len(exog_names):
                    mapped[exog_names[exog_i]] = feat
            # Now try to find which exog key maps to varname (exact or substring)
            found_exog = None
            for exog_key, human in mapped.items():
                if human == varname or varname.lower() in str(human).lower():
                    found_exog = exog_key
                    break
            if found_exog:
                idx = exog_names.index(found_exog)
                if idx < 0 or idx >= arr.size:
                    return {"error": f"mapped index for '{varname}' ({idx}) out of bounds (params size={arr.size})"}
                coef_val = float(arr[idx])
                bse_val = None
                try:
                    if bse is not None:
                        bse_arr = _np.asarray(bse)
                        if bse_arr.size > idx:
                            bse_val = float(bse_arr[idx])
                except Exception:
                    bse_val = None
                pval = None
                try:
                    if pvalues is not None:
                        p_arr = _np.asarray(pvalues)
                        if p_arr.size > idx:
                            pval = float(p_arr[idx])
                except Exception:
                    pval = None
                return {
                    "coef": coef_val,
                    "std_err": bse_val,
                    "pvalue": pval,
                    "conf_int": ci.get(str(idx), []),
                    "n_obs": int(nobs) if nobs is not None else None,
                    "model_type": type(res).__name__,
                    "mapped_exog_name": found_exog,
                    "mapped_index": idx,
                    "mapped_to_feature": mapped.get(found_exog)
                }
            # fallthrough to positional mapping if mapping did not find varname

        # If exog_names present, try to find direct or substring match in exog_names
        if exog_names:
            if varname in exog_names:
                idx = exog_names.index(varname)
            else:
                idx = None
                for i, nm in enumerate(exog_names):
                    if isinstance(nm, str) and varname.lower() in nm.lower():
                        idx = i
                        break
            if idx is None:
                # if exog_names are generic and we didn't map above, provide helpful message with sample names
                return {"error": f"variable '{varname}' not found in model.exog_names; available names sample: {exog_names[:30]}"}
            if idx < 0 or idx >= arr.size:
                return {"error": f"index for '{varname}' ({idx}) out of bounds for params array of size {arr.size}"}
            coef_val = float(arr[idx])
            bse_val = None
            try:
                if bse is not None:
                    bse_arr = _np.asarray(bse)
                    if bse_arr.size > idx:
                        bse_val = float(bse_arr[idx])
            except Exception:
                bse_val = None
            pval = None
            try:
                if pvalues is not None:
                    p_arr = _np.asarray(pvalues)
                    if p_arr.size > idx:
                        pval = float(p_arr[idx])
            except Exception:
                pval = None
            return {
                "coef": coef_val,
                "std_err": bse_val,
                "pvalue": pval,
                "conf_int": ci.get(str(idx), []),
                "n_obs": int(nobs) if nobs is not None else None,
                "model_type": type(res).__name__,
                "mapped_exog_name": exog_names[idx],
                "mapped_index": idx
            }

        # If no exog_names but array present, last-resort: try to map by positional assumption (const at 0)
        # Heuristic: varname might be in feature_names list if available; use that to compute index
        if feature_names:
            # assume const at 0, then first len(feature_names) positions correspond to features
            if len(arr) >= (1 + len(feature_names)):
                try:
                    feat_pos = feature_names.index(varname)
                    idx = 1 + feat_pos if arr.size > (1 + feat_pos) else None
                    if idx is None:
                        return {"error": f"could not locate '{varname}' positionally using feature_names; arr.shape={arr.shape}"}
                    coef_val = float(arr[idx])
                    bse_val = None
                    try:
                        if bse is not None:
                            bse_arr = _np.asarray(bse)
                            if bse_arr.size > idx:
                                bse_val = float(bse_arr[idx])
                    except Exception:
                        bse_val = None
                    pval = None
                    try:
                        if pvalues is not None:
                            p_arr = _np.asarray(pvalues)
                            if p_arr.size > idx:
                                pval = float(p_arr[idx])
                    except Exception:
                        pval = None
                    return {
                        "coef": coef_val,
                        "std_err": bse_val,
                        "pvalue": pval,
                        "conf_int": ci.get(str(idx), []),
                        "n_obs": int(nobs) if nobs is not None else None,
                        "model_type": type(res).__name__,
                        "mapped_index": idx,
                        "mapped_to_feature": varname
                    }
                except ValueError:
                    pass

        # Last resort: cannot map
        return {"error": f"could not map '{varname}' positionally; params array shape={arr.shape}"}

    except Exception as e:
        import traceback
        return {"error": f"extract_stats_statsmodels failed: {repr(e)}", "trace": traceback.format_exc()}



def extract_stats_sklearn_linear(estimator, feature_names, varname):
    """
    Extract coefficient from sklearn estimator or sklearn Pipeline.
    Accepts:
      - estimator: pipeline or final estimator
      - feature_names: list of features in same order as X used to fit the estimator (best-effort fallback)
    Returns dict with coef, shrinkage flag, model_type or error.
    """
    out = {}
    import numpy as _np
    try:
        # If pipeline, unwrap last estimator
        est = estimator
        if hasattr(estimator, "named_steps"):
            # get last step
            try:
                last_name = list(estimator.named_steps.keys())[-1]
                est = estimator.named_steps[last_name]
            except Exception:
                # fallback to pipeline[-1]
                try:
                    est = estimator.steps[-1][1]
                except Exception:
                    est = estimator

        # Try to get coef_ from estimator
        coef_attr = getattr(est, "coef_", None)
        if coef_attr is None:
            return {"error": "estimator.coef_ not found (not a linear estimator?)", "model_type": type(est).__name__}

        coefs = _np.asarray(coef_attr)
        # coefs might be scalar (0-D) in some edge cases: coerce to 1-D
        if coefs.ndim == 0:
            coefs = _np.atleast_1d(coefs)
        # if multi-output with shape (n_targets, n_features), attempt to flatten first row
        if coefs.ndim > 1:
            # pick first row (most common for regression with shape (1, n_features))
            coefs = coefs.ravel()
        # feature_names fallback: try estimator.feature_names_in_
        fnames = feature_names or []
        if not fnames and hasattr(est, "feature_names_in_"):
            try:
                fnames = list(getattr(est, "feature_names_in_"))
            except Exception:
                fnames = fnames

        if not fnames:
            return {"error": "feature_names not provided and estimator.feature_names_in_ missing; save artifacts/feature_names.json", "coef_shape": coefs.shape}

        if len(fnames) != coefs.size:
            # if mismatch, emit error but still attempt to find varname by substring in feature names
            if varname in fnames:
                idx = fnames.index(varname)
            else:
                # try fuzzy match
                candidates = [i for i, f in enumerate(fnames) if varname.lower() in str(f).lower()]
                if candidates:
                    idx = candidates[0]
                else:
                    return {"error": f"feature_names length ({len(fnames)}) != coef length ({coefs.size}); and '{varname}' not found in feature_names"}
        else:
            idx = fnames.index(varname) if varname in fnames else None
            if idx is None:
                candidates = [i for i, f in enumerate(fnames) if varname.lower() in str(f).lower()]
                idx = candidates[0] if candidates else None
            if idx is None:
                return {"error": f"variable '{varname}' not found in provided feature_names"}

        coef_val = float(coefs[idx])
        out = {
            "coef": coef_val,
            "shrinkage": True,
            "model_type": type(est).__name__,
            "feature_index": idx,
            "feature_name": fnames[idx]
        }
        return out
    except Exception as e:
        import traceback
        return {"error": f"extract_stats_sklearn_linear failed: {repr(e)}", "trace": traceback.format_exc()}


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def write_md(path: Path, data: dict, varname: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {varname} summary\n"]
    for model_name, stats in data.items():
        lines.append(f"## {model_name}\n")
        if "error" in stats:
            lines.append(f"- ERROR: {stats['error']}\n")
            if stats.get("trace"):
                lines.append("```\n" + stats["trace"] + "\n```\n")
            continue
        lines.append(f"- model_type: {stats.get('model_type')}\n")
        lines.append(f"- coef: {stats.get('coef')}\n")
        if stats.get("std_err") is not None:
            lines.append(f"- std_err: {stats.get('std_err')}\n")
        if stats.get("pvalue") is not None:
            lines.append(f"- pvalue: {stats.get('pvalue')}\n")
        if stats.get("conf_int"):
            lines.append(f"- conf_int: {stats.get('conf_int')}\n")
        if stats.get("n_obs") is not None:
            lines.append(f"- n_obs: {stats.get('n_obs')}\n")
        if stats.get("shrinkage"):
            lines.append(f"- shrinkage: {stats.get('shrinkage')}\n")
        lines.append("\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))

def main(args):
    varname = args.var
    artifacts = Path(args.artifacts)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {}

        # ---- Begin robust per-model extraction & per-file writes ----
    summary = {}

    # --- FE artifact extraction & write ---
    fe_path_candidates = [
        artifacts / "fe_result.pkl",
        artifacts / "fe_result.joblib",
        artifacts / "fe_result"
    ]
    fe_loaded = None
    for p in fe_path_candidates:
        if p.exists():
            try:
                fe_loaded = try_load(p)
                fe_path = p
                break
            except Exception:
                pass
    if fe_loaded is None:
        fe_stats = {"error": "FE artifact not found; tried: " + ", ".join(map(str, fe_path_candidates))}
    else:
        fe_stats = extract_stats_statsmodels(fe_loaded, varname)
    # write FE per-model file (guaranteed unique filename)
    write_json(outdir / f"fe_{varname}.json", fe_stats)
    summary["FE"] = fe_stats

    # --- OLS artifact extraction & write ---
    ols_path_candidates = [
        artifacts / "ols_result.pkl",
        artifacts / "ols_result.joblib",
        artifacts / "ols_result"
    ]
    ols_loaded = None
    for p in ols_path_candidates:
        if p.exists():
            try:
                ols_loaded = try_load(p)
                ols_path = p
                break
            except Exception:
                pass
    if ols_loaded is None:
        ols_stats = {"error": "OLS artifact not found; tried: " + ", ".join(map(str, ols_path_candidates))}
    else:
        ols_stats = extract_stats_statsmodels(ols_loaded, varname)
    write_json(outdir / f"ols_{varname}.json", ols_stats)
    summary["OLS"] = ols_stats

    # --- ElasticNet / sklearn artifact extraction & write ---
    en_path_candidates = [
        artifacts / "en_model.joblib",
        artifacts / "en_model.pkl",
        artifacts / "elasticnet_model.joblib",
        artifacts / "elasticnet_model.pkl"
    ]
    en_loaded = None
    en_path = None
    for p in en_path_candidates:
        if p.exists():
            try:
                en_loaded = try_load(p)
                en_path = p
                break
            except Exception:
                pass

    # feature names
    feature_names = []
    fn_path = artifacts / "feature_names.json"
    if fn_path.exists():
        try:
            with open(fn_path, "r", encoding="utf8") as f:
                feature_names = json.load(f)
        except Exception:
            feature_names = []
    if en_loaded is None:
        en_stats = {"error": "ElasticNet artifact not found; tried: " + ", ".join(map(str, en_path_candidates))}
    else:
        if not feature_names:
            en_stats = {"error": "ElasticNet loaded but feature_names not found. Place feature_names.json in artifacts/ (ordered list)."}
        else:
            en_stats = extract_stats_sklearn_linear(en_loaded, feature_names, varname)
    write_json(outdir / f"elasticnet_{varname}.json", en_stats)
    summary["ElasticNet"] = en_stats

    # --- Aggregate summary and human-readable markdown ---
    write_json(outdir / f"{varname}_summary.json", summary)
    write_md(outdir / f"{varname}_summary.md", summary, varname)

    # --- Quick verification print to console of what we wrote ---
    print(f"Wrote outputs to {outdir.resolve()}")
    for model_key, fname in [("FE", outdir / f"fe_{varname}.json"),
                             ("OLS", outdir / f"ols_{varname}.json"),
                             ("ElasticNet", outdir / f"elasticnet_{varname}.json"),
                             ("AGGREGATE", outdir / f"{varname}_summary.json")]:
        try:
            with open(fname, "r", encoding="utf8") as fh:
                data = json.load(fh)
            # print top-level keys and a quick marker to ensure files differ
            keys = list(data.keys()) if isinstance(data, dict) else []
            print(f"--- {model_key} ({fname.name}) --- keys:", keys[:10])
        except Exception as e:
            print(f"--- {model_key} ({fname.name}) --- could not read: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", required=True, help="Variable name to extract (e.g. trade_exposure)")
    parser.add_argument("--artifacts", default=str(ARTIFACTS_DEFAULT), help="Folder where model artifacts are stored (default: artifacts/)")
    parser.add_argument("--outdir", default=str(OUTDIR_DEFAULT), help="Output folder (default: outputs/)")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
        sys.exit(1)
