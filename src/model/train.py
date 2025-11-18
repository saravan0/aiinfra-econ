# src/model/train.py
"""
Model training & robustness runner — polished research-grade version.

Produces:
 - models/*.joblib                (saved model artifacts - canonical)
 - artifacts/*.joblib | *.pkl     (artifact copies for extraction)
 - reports/ols_summary.txt        (statsmodels text)
 - reports/fe_summary.txt
 - reports/model_table.csv        (coef table across models)
 - reports/model_plots.png
 - reports/model_artifacts_manifest.json
 - reports/model_metadata.json    (config snapshot)
 - reports/artifact_save_summary.json

Usage:
    python -m src.model.train --config config/model.yml
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, NamedTuple

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from joblib import dump as joblib_dump
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Project imports (assumed present in your repo)
from src.model import model_defs as mdefs
from src.model import utils as mutils
# stability gate (ensures coverage/sample checks run prior)
from src.model.stability_gate import main as run_stability_gate

LOG = logging.getLogger("src.model.train")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)


# -----------------------------
# Lightweight datatypes
# -----------------------------
@dataclass
class RunResult:
    model_name: str
    terms: List[str]
    n_obs: int
    rows: List[Dict[str, Any]]
    fitted: Optional[Any] = None


class ElasticNetResult(NamedTuple):
    """Light-weight return object used by main."""
    fitted: Any
    rows: List[Dict[str, Any]]
    n_obs: int


# -----------------------------
# Utilities: config / io
# -----------------------------
def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def load_data(path: Path, sample_filter: Optional[str] = None, max_rows: Optional[int] = None) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if sample_filter:
        try:
            df = df.query(sample_filter)
        except Exception as e:
            LOG.warning("Sample filter failed to apply: %s. Proceeding without filter.", e)
    if max_rows:
        df = df.head(max_rows)
    df = df.reset_index(drop=True)
    return df


def _dropna_for_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return dataframe with rows having non-null for all specified cols."""
    if len(cols) == 0:
        return df.copy()
    return df.loc[df[cols].notna().all(axis=1)].copy()


# -----------------------------
# Summarization helpers
# -----------------------------
def _summarize_sm_results_fallback(sm_obj: Any, cols: List[str], n_obs: int, model_name: str) -> List[Dict[str, Any]]:
    """Fallback summarizer for arbitrary statsmodels-like results."""
    rows: List[Dict[str, Any]] = []
    params = getattr(sm_obj, "params", {})
    bse = getattr(sm_obj, "bse", {})
    pvalues = getattr(sm_obj, "pvalues", {})
    for c in cols:
        rows.append({
            "model": model_name,
            "term": str(c),
            "coef": float(params.get(c, np.nan)) if c in params else np.nan,
            "std_err": float(bse.get(c, np.nan)) if c in bse else np.nan,
            "pvalue": float(pvalues.get(c, np.nan)) if c in pvalues else np.nan,
            "n_obs": int(n_obs)
        })
    return rows


def summarize_sm_results(sm_obj: Any, cols: List[str], n_obs: int, model_name: str) -> List[Dict[str, Any]]:
    """Use mutils.summarize_sm_results if present, else fallback."""
    if hasattr(mutils, "summarize_sm_results"):
        try:
            return mutils.summarize_sm_results(sm_obj, cols, n_obs, model_name)
        except Exception:
            LOG.debug("mutils.summarize_sm_results failed, using fallback.")
    return _summarize_sm_results_fallback(sm_obj, cols, n_obs, model_name)


# -----------------------------
# Core modeling functions
# -----------------------------
def run_ols(
    df: pd.DataFrame,
    target: str,
    predictors: List[str],
    cluster_on: Optional[str] = None,
    model_name: str = "OLS",
) -> RunResult:
    """Run OLS; optionally cluster standard errors by `cluster_on` column."""
    required_cols = [target] + [p for p in predictors if p in df.columns]
    if cluster_on and cluster_on in df.columns:
        required_cols.append(cluster_on)
    sdf = _dropna_for_columns(df, required_cols)
    n_obs = len(sdf)
    if n_obs == 0:
        raise ValueError("No observations after dropping NA for OLS.")
    X = sm.add_constant(sdf[[c for c in predictors if c in sdf.columns]].astype(float), has_constant="add")
    y = sdf[target].astype(float)
    if cluster_on and cluster_on in sdf.columns and sdf[cluster_on].nunique() > 1:
        groups = sdf[cluster_on]
        LOG.info("Fitting OLS with cluster-robust SEs (cluster_on=%s).", cluster_on)
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        if cluster_on and cluster_on not in sdf.columns:
            LOG.warning("Requested cluster_on '%s' not available after filtering -> falling back to normal SEs.", cluster_on)
        LOG.info("Fitting OLS (no clustering).")
        res = sm.OLS(y, X).fit()
    rows = summarize_sm_results(res, X.columns.tolist(), n_obs, model_name)
    return RunResult(model_name=model_name, terms=X.columns.tolist(), n_obs=n_obs, rows=rows, fitted=res)


def group_small_countries(df: pd.DataFrame, entity_col: str = "iso3", min_obs: int = 3, group_name: str = "OTHER") -> pd.DataFrame:
    """
    Replace iso3 codes that appear < min_obs times with group_name.
    Keeps dtype stable (string) and returns a copy.
    """
    if entity_col not in df.columns:
        return df.copy()
    counts = df[entity_col].value_counts(dropna=True)
    small = counts[counts < min_obs].index.tolist()
    if not small:
        return df.copy()
    out = df.copy()
    out[entity_col] = out[entity_col].fillna(group_name).astype(str)
    mask = out[entity_col].isin(small)
    if mask.any():
        out.loc[mask, entity_col] = group_name
    return out


def run_fixed_effects(
    df: pd.DataFrame,
    target: str,
    predictors: List[str],
    entity_col: str = "iso3",
    time_col: Optional[str] = None,
    drop_first: bool = True,
    prefix: str = "FE",
    min_obs_for_iso: int = 3,
    group_name: str = "OTHER",
) -> RunResult:
    """
    Fixed-effects with defragmented FE creation and robust numeric coercion.
    Strategy:
      - group small countries
      - expand FE dummies once (via mdefs.country_fixed_effects)
      - coerce numeric predictors carefully
      - drop low-support / constant FE dummies
      - fit OLS on final design matrix
    """
    LOG.info("run_fixed_effects: starting (entity=%s, min_obs=%d)", entity_col, min_obs_for_iso)

    # 1) Group small countries
    dfg = group_small_countries(df, entity_col=entity_col, min_obs=min_obs_for_iso, group_name=group_name)

    # 2) required cols and strict dropna
    cols_needed = [entity_col, target] + [p for p in predictors if p in dfg.columns]
    sdf = _dropna_for_columns(dfg, cols_needed)
    if len(sdf) == 0:
        raise ValueError("No observations after dropping NA for FE (entity/target/predictors).")

    # 3) Expand FE dummies (user-provided helper)
    sdf_fe = mdefs.country_fixed_effects(sdf, country_col=entity_col, drop_first=drop_first, prefix=prefix)
    LOG.info("run_fixed_effects: FE expansion produced %d cols (sample)", len([c for c in sdf_fe.columns if str(c).startswith(f"{prefix}_")]))

    # determine numeric predictor columns present after FE expansion
    numeric_preds = [p for p in predictors if p in sdf_fe.columns]
    fe_dummy_cols = [c for c in sdf_fe.columns if isinstance(c, str) and c.startswith(f"{prefix}_")]
    design_cols = numeric_preds + fe_dummy_cols
    if not design_cols:
        raise ValueError("No design columns found after FE expansion.")

    # 4) Build X_df WITHOUT repeated inserts (collect columns then concat)
    pieces: List[pd.Series] = []

    # numeric predictors -> try coercion once
    if numeric_preds:
        num_block = sdf_fe[numeric_preds].apply(pd.to_numeric, errors="coerce")
        pieces.append(num_block)

    # FE dummies: produce a dict of coerced series then concat once (prevents fragmentation)
    if fe_dummy_cols:
        fe_dict = {}
        for c in fe_dummy_cols:
            series = sdf_fe[c]
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.isna().all():
                coerced = series.astype(object).map({True: 1, False: 0})
            fe_dict[c] = pd.to_numeric(coerced, errors="coerce")
        if fe_dict:
            fe_block = pd.DataFrame(fe_dict, index=sdf_fe.index)
            pieces.append(fe_block)

    # concat all pieces once
    if pieces:
        X_df = pd.concat(pieces, axis=1)
    else:
        X_df = pd.DataFrame(index=sdf_fe.index)

    # 5) Drop columns that are entirely NaN
    col_non_na = X_df.notna().sum()
    drop_allnan = col_non_na[col_non_na == 0].index.tolist()
    if drop_allnan:
        LOG.info("run_fixed_effects: dropping all-NaN cols after coercion: %s", drop_allnan)
        X_df = X_df.drop(columns=drop_allnan)
        fe_dummy_cols = [c for c in fe_dummy_cols if c not in drop_allnan]
        numeric_preds = [c for c in numeric_preds if c not in drop_allnan]
        design_cols = [c for c in design_cols if c not in drop_allnan]

    if X_df.shape[1] == 0:
        raise ValueError("No columns remain after dropping all-NaN columns.")

    # 6) Drop FE dummies with extremely low support (<= 1 non-zero)
    to_drop = []
    for c in [c for c in X_df.columns if str(c).startswith(f"{prefix}_")]:
        nonzeros = (X_df[c].fillna(0).astype(float) != 0).sum()
        if nonzeros <= 1:
            to_drop.append(c)
    if to_drop:
        LOG.info("run_fixed_effects: dropping low-support FE dummies: %s", to_drop)
        X_df = X_df.drop(columns=to_drop)
        design_cols = [c for c in design_cols if c not in to_drop]

    # 7) Drop constant columns (zero variance)
    nunique = X_df.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        LOG.info("run_fixed_effects: dropping constant cols: %s", const_cols)
        X_df = X_df.drop(columns=const_cols)
        design_cols = [c for c in design_cols if c not in const_cols]

    if X_df.shape[1] == 0:
        raise ValueError("No design columns remain after dropping low-support FE and constant columns.")

    # 8) Keep only full-case rows (no NaNs in design matrix)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    full_idx = X_df.dropna().index
    if len(full_idx) == 0:
        raise ValueError("No full-case rows remain after coercion / dropping constant / low-support FE dummies.")
    X_df = X_df.loc[full_idx]
    y = sdf_fe.loc[full_idx, target].astype(float)

    print("DEBUG FE DESIGN COLS:", list(X_df.columns)[:60])

    # 9) Build final design (with constant)
    X_arr = sm.add_constant(X_df, has_constant="add")

    # 10) Convert to float numpy array for finite checks (robustly)
    try:
        arr = X_arr.to_numpy(dtype=float)
    except Exception:
        X_arr = X_arr.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        arr = X_arr.to_numpy(dtype=float)

    # find non-finite columns if any
    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        bad_cols = []
        for j, col in enumerate(X_arr.columns):
            col_vals = arr[:, j]
            if not np.isfinite(col_vals).all():
                bad_cols.append(col)
        raise ValueError(f"Design matrix contains non-finite values after cleaning. Bad cols: {bad_cols}")

    # 11) Fit OLS on cleaned design
    LOG.info("run_fixed_effects: FE fit starting (n_obs=%d, n_cols=%d)", arr.shape[0], arr.shape[1])
    res = sm.OLS(y.values, arr).fit()

    # 12) Summarize and return (wrap columns back into names)
    cols = list(X_arr.columns)
    rows = summarize_sm_results(res, cols, len(X_arr), "FE")
    LOG.info("run_fixed_effects: FE fit complete. n_obs=%d, n_cols=%d", len(X_arr), len(cols))
    return RunResult(model_name="FE", terms=cols, n_obs=len(X_arr), rows=rows, fitted=res)


# -----------------------------
# ElasticNet wrapper
# -----------------------------
def _baseline_to_predictors(baseline: Union[Dict, List, None]) -> List[str]:
    """Normalize different baseline representations into a list of predictor names."""
    if baseline is None:
        return []
    if isinstance(baseline, dict):
        preds = baseline.get("predictors") or baseline.get("features") or []
        return [p for p in preds if isinstance(p, str)]
    if isinstance(baseline, list):
        if all(isinstance(x, str) for x in baseline):
            return baseline
        if all(isinstance(x, dict) for x in baseline):
            out = []
            for d in baseline:
                if "name" in d and isinstance(d["name"], str):
                    out.append(d["name"])
                elif "term" in d and isinstance(d["term"], str):
                    out.append(d["term"])
                elif "predictor" in d and isinstance(d["predictor"], str):
                    out.append(d["predictor"])
            return out
    return []


def run_elasticnet(df: pd.DataFrame,
                   target: str,
                   baseline: Union[Dict, List, None],
                   l1_ratio: float = 0.5,
                   cv: int = 5,
                   random_state: int = 0,
                   save_path: Optional[Path] = None) -> ElasticNetResult:
    """
    Robust ElasticNet wrapper that accepts baseline either as:
      - dict with key "predictors": ["x1","x2"...]
      - list of predictor strings: ["x1","x2"...]
      - list of dicts [{'name': 'x1'}, ...]
    Returns ElasticNetResult(fitted, rows, n_obs).
    """
    preds = _baseline_to_predictors(baseline)
    if len(preds) == 0:
        raise ValueError("run_elasticnet: no predictors found in baseline (expected dict or list)")

    # Keep only predictors that exist in df
    preds = [p for p in preds if p in df.columns]
    if len(preds) == 0:
        raise ValueError("run_elasticnet: none of the baseline predictors are present in df.columns")

    # prepare numeric X,y and drop full-case rows
    subset = df[preds + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if subset.shape[0] == 0:
        raise ValueError("run_elasticnet: no full-case rows available to fit ElasticNet")

    X = subset[preds].to_numpy(dtype=float)
    y = subset[target].astype(float).to_numpy()

    # fit pipeline
    enet_cv = ElasticNetCV(l1_ratio=l1_ratio, cv=cv, random_state=random_state, n_jobs=-1)
    pipeline = make_pipeline(StandardScaler(), enet_cv)
    pipeline.fit(X, y)

    enet_obj = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
    coef = np.asarray(enet_obj.coef_).ravel()
    intercept = float(getattr(enet_obj, "intercept_", 0.0))

    # prepare rows consistent with main's expected schema
    rows = []
    for name, c in zip(preds, coef):
        rows.append({
            "model": "ElasticNetCV",
            "term": name,
            "coef": float(c),
            "std_err": None,
            "pvalue": None,
            "n_obs": int(X.shape[0]),
        })
    rows.append({
        "model": "ElasticNetCV",
        "term": "const",
        "coef": float(intercept),
        "std_err": None,
        "pvalue": None,
        "n_obs": int(X.shape[0]),
    })

    # optionally save (caller also attempts to save)
    if save_path is not None:
        try:
            joblib_dump(pipeline, str(save_path))
            LOG.info("Saved ElasticNet pipeline -> %s", save_path)
        except Exception:
            LOG.warning("Could not joblib.dump pipeline to %s (caller will handle saving).", save_path)

    return ElasticNetResult(fitted=pipeline, rows=rows, n_obs=int(X.shape[0]))


# -----------------------------
# Reporting helpers
# -----------------------------
def _write_text_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf8")


def _collect_and_write_table(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        LOG.warning("No result rows to write to %s", out_path)
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    LOG.info("Wrote model table -> %s", out_path)


def _make_manifest(models_dir: Path, reports_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "models": sorted([p.name for p in models_dir.glob("*") if p.is_file()]),
        "reports": sorted([p.name for p in reports_dir.glob("*") if p.is_file()]),
        "config_snapshot": "model_metadata.json",
    }
    return manifest


# -----------------------------
# Artifact saving (canonical)
# -----------------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _try_joblib_dump(obj, target: Path) -> bool:
    try:
        joblib_dump(obj, str(target))
        return True
    except Exception:
        return False


def _try_pickle_dump(obj, target: Path) -> bool:
    try:
        with open(target, "wb") as fh:
            pickle.dump(obj, fh)
        return True
    except Exception:
        return False


def save_artifacts(models_dir: Path,
                   artifacts_dir: Path,
                   ols_res_obj = None,
                   fe_res_obj = None,
                   en_pipeline_obj = None,
                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Save canonical artifacts for downstream extraction and reproducibility.
    Attempts joblib first, then pickle fallback. Writes manifest JSON.
    Returns a dict with saved paths and any errors.
    """
    artifacts_dir = _ensure_dir(artifacts_dir)
    models_dir = _ensure_dir(models_dir)

    results = {"saved": {}, "errors": {}}

    def _save(obj, name_base: str):
        if obj is None:
            results["errors"][name_base] = "object is None (not produced)"
            return None
        joblib_path = artifacts_dir / f"{name_base}.joblib"
        pkl_path = artifacts_dir / f"{name_base}.pkl"
        saved = False
        try:
            saved = _try_joblib_dump(obj, joblib_path)
            if saved:
                results["saved"][name_base] = str(joblib_path)
                # also write a copy in models_dir for provenance (best-effort)
                try:
                    joblib_dump(obj, str(models_dir / f"{name_base}.joblib"))
                except Exception:
                    pass
                return str(joblib_path)
        except Exception as e:
            results["errors"][name_base] = f"joblib dump failed: {repr(e)}"

        # fallback to pickle
        try:
            saved_pickle = _try_pickle_dump(obj, pkl_path)
            if saved_pickle:
                results["saved"][name_base] = str(pkl_path)
                try:
                    with open(models_dir / f"{name_base}.pkl", "wb") as fh:
                        pickle.dump(obj, fh)
                except Exception:
                    pass
                return str(pkl_path)
        except Exception as e:
            results["errors"][name_base] = f"pickle dump failed: {repr(e)}"
        results["errors"][name_base] = results["errors"].get(name_base, "all dumps failed")
        return None

    # Save OLS
    try:
        _save(ols_res_obj, "ols_result")
    except Exception as e:
        results["errors"]["ols_result"] = repr(e)

    # Save FE
    try:
        _save(fe_res_obj, "fe_result")
    except Exception as e:
        results["errors"]["fe_result"] = repr(e)

    # Save ElasticNet pipeline
    try:
        _save(en_pipeline_obj, "en_model")
    except Exception as e:
        results["errors"]["en_model"] = repr(e)

    # feature names (JSON) for ElasticNet
    if feature_names:
        try:
            fn_path = artifacts_dir / "feature_names.json"
            with open(fn_path, "w", encoding="utf8") as fh:
                json.dump(list(map(str, feature_names)), fh, indent=2, ensure_ascii=False)
            results["saved"]["feature_names.json"] = str(fn_path)
            # also copy to models_dir for convenience
            try:
                with open(models_dir / "feature_names.json", "w", encoding="utf8") as fh:
                    json.dump(list(map(str, feature_names)), fh, indent=2, ensure_ascii=False)
            except Exception:
                pass
        except Exception as e:
            results["errors"]["feature_names.json"] = repr(e)
    else:
        results["errors"]["feature_names.json"] = "feature_names list was empty or None"

    # write a small manifest inside reports_dir (best-effort)
    try:
        manifest = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "artifacts_dir": str(artifacts_dir.resolve()),
            "models_dir": str(models_dir.resolve()),
            "saved_files": results["saved"],
            "errors": results["errors"]
        }
        # attempt to save under reports_dir if available in outer scope; else write to artifacts_dir
        try:
            rpt_path = reports_dir / "model_artifacts_manifest.json"
        except Exception:
            rpt_path = artifacts_dir / "model_artifacts_manifest.json"
        with open(rpt_path, "w", encoding="utf8") as fh:
            json.dump(manifest, fh, indent=2)
        results["saved"]["manifest"] = str(rpt_path)
    except Exception as e:
        results["errors"]["manifest_write"] = repr(e)

    LOG.info("Artifact save results: saved=%s errors=%s", list(results.get("saved",{}).keys()), list(results.get("errors",{}).keys()))
    return results


# -----------------------------
# Main runner
# -----------------------------
def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/model.yml", help="Path to YAML config")
    args = p.parse_args(argv)

    cfg = load_config(Path(args.config))
    cfg.setdefault("seed", 2025)
    cfg.setdefault("debug", {})
    cfg.setdefault("outputs", {"reports_dir": "reports", "models_dir": "models", "model_table": "reports/model_table.csv", "model_plots": "reports/model_plots.png", "artifacts_dir": "artifacts"})

    reports_dir = Path(cfg["outputs"]["reports_dir"])
    models_dir = Path(cfg["outputs"]["models_dir"])
    artifacts_dir = Path(cfg["outputs"].get("artifacts_dir", "artifacts"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # snapshot config for provenance
    try:
        mutils.save_config_snapshot(cfg, reports_dir / "model_metadata.json")
    except Exception:
        with open(reports_dir / "model_metadata.json", "w", encoding="utf8") as fh:
            json.dump(cfg, fh, indent=2)
        LOG.info("Wrote config snapshot (fallback) -> %s", reports_dir / "model_metadata.json")

    # run stability gate (coverage / sample checks)
    try:
        run_stability_gate()
    except Exception as e:
        LOG.warning("Stability gate raised: %s", e)

    seed = int(cfg.get("seed", 2025))
    np.random.seed(seed)

    df = load_data(Path(cfg["data"]["features_path"]), sample_filter=cfg["data"].get("sample_filter"), max_rows=cfg["debug"].get("max_rows"))
    LOG.info("Loaded features: %s rows, %s cols", df.shape[0], df.shape[1])

    baseline_requested = (cfg.get("predictors", {}).get("baseline") or []) + (cfg.get("predictors", {}).get("extra_controls") or [])
    baseline = mdefs.safe_select_columns(df, baseline_requested)
    if not baseline:
        raise SystemExit("No predictors found in features for the requested baseline/controls. Check config.")

    results_rows: List[Dict[str, Any]] = []

    ols_res = None
    fe_res = None
    enet_res = None

    # OLS
    if cfg.get("models", {}).get("ols", {}).get("run", True):
        try:
            cluster_on = cfg["models"]["ols"].get("cluster_on")
            LOG.info("Running OLS baseline (cluster=%s)", cluster_on)
            ols_res = run_ols(df, cfg["target"]["name"], baseline, cluster_on=cluster_on, model_name="OLS")
            results_rows.extend(ols_res.rows)
            try:
                if getattr(ols_res.fitted, "summary", None):
                    _write_text_report(reports_dir / "ols_summary.txt", ols_res.fitted.summary().as_text())
            except Exception:
                LOG.debug("Could not write OLS summary text.")
            try:
                mutils.save_model(ols_res.fitted, models_dir / "ols_model.joblib")
            except Exception:
                joblib.dump(ols_res.fitted, models_dir / "ols_model.joblib")
            LOG.info("OLS completed. n_obs=%d", ols_res.n_obs)
        except Exception as e:
            LOG.exception("OLS failed: %s", e)

    # Fixed Effects
    if cfg.get("models", {}).get("fixed_effects", {}).get("run", True):
        try:
            fe_entity = cfg["models"]["fixed_effects"].get("entity", "iso3")
            LOG.info("Running Fixed Effects (entity=%s)", fe_entity)
            fe_res = run_fixed_effects(df, cfg["target"]["name"], baseline, entity_col=fe_entity)
            results_rows.extend(fe_res.rows)
            try:
                if getattr(fe_res.fitted, "summary", None):
                    _write_text_report(reports_dir / "fe_summary.txt", fe_res.fitted.summary().as_text())
            except Exception:
                LOG.debug("Could not write FE summary text.")
            try:
                mutils.save_model(fe_res.fitted, models_dir / "fe_model.joblib")
            except Exception:
                joblib.dump(fe_res.fitted, models_dir / "fe_model.joblib")
            LOG.info("FE completed. n_obs=%d", fe_res.n_obs)
        except Exception as e:
            LOG.exception("FE OLS fit failed: %s", e)
            LOG.error("Fixed-effects failed: %s", e)

    # ElasticNet
    if cfg.get("models", {}).get("elasticnet", {}).get("run", True):
        try:
            en_cfg = cfg["models"]["elasticnet"]
            LOG.info("Running ElasticNetCV (cv=%s l1_ratio=%s)", en_cfg.get("cv"), en_cfg.get("l1_ratio"))
            enet_res = run_elasticnet(df, cfg["target"]["name"], baseline, l1_ratio=en_cfg.get("l1_ratio", 0.5), cv=en_cfg.get("cv", 5), random_state=seed)
            results_rows.extend(enet_res.rows)
            try:
                mutils.save_model(enet_res.fitted, models_dir / "elasticnet_cv.joblib")
            except Exception:
                joblib.dump(enet_res.fitted, models_dir / "elasticnet_cv.joblib")
            LOG.info("ElasticNet completed. n_obs=%d", enet_res.n_obs)
        except Exception as e:
            LOG.exception("ElasticNet failed: %s", e)

    # write aggregated model table
    try:
        _collect_and_write_table(results_rows, Path(cfg["outputs"].get("model_table", "reports/model_table.csv")))
    except Exception as e:
        LOG.exception("Writing model table failed: %s", e)

    # simple diagnostic scatter plot (best-effort)
    try:
        if cfg["target"]["name"] in df.columns and "gov_index_zmean" in df.columns:
            plot_path = Path(cfg["outputs"].get("model_plots", "reports/model_plots.png"))
            mutils.plot_scatter_with_fit(df.dropna(subset=[cfg["target"]["name"], "gov_index_zmean"]), "gov_index_zmean", cfg["target"]["name"], fname=plot_path)
    except Exception as e:
        LOG.exception("Plotting failed: %s", e)

    # Make manifest and save
    manifest = _make_manifest(models_dir, reports_dir, cfg)
    try:
        mutils.save_json(manifest, reports_dir / "model_artifacts_manifest.json")
    except Exception:
        with open(reports_dir / "model_artifacts_manifest.json", "w", encoding="utf8") as fh:
            json.dump(manifest, fh, indent=2)
        LOG.info("Saved manifest (fallback) -> %s", reports_dir / "model_artifacts_manifest.json")

    # -----------------------
    # Save canonical artifacts (for extractor & reproducibility)
    # -----------------------
    try:
        # prepare objects to save (use fitted attributes where present)
        _ols_obj = ols_res.fitted if (ols_res is not None and getattr(ols_res, "fitted", None) is not None) else None
        _fe_obj = fe_res.fitted if (fe_res is not None and getattr(fe_res, "fitted", None) is not None) else None
        _en_obj = enet_res.fitted if (enet_res is not None and getattr(enet_res, "fitted", None) is not None) else None

        # feature names used for ElasticNet: baseline order (best-effort)
        _feature_names = None
        try:
            if isinstance(baseline, list) and baseline:
                _feature_names = baseline
            else:
                if _en_obj is not None and hasattr(_en_obj, "named_steps"):
                    last_step = list(_en_obj.named_steps.items())[-1][1]
                    if hasattr(last_step, "feature_names_in_"):
                        _feature_names = list(getattr(last_step, "feature_names_in_"))
        except Exception:
            _feature_names = None

        save_results = save_artifacts(models_dir=models_dir, artifacts_dir=artifacts_dir, ols_res_obj=_ols_obj, fe_res_obj=_fe_obj, en_pipeline_obj=_en_obj, feature_names=_feature_names)
        try:
            with open(reports_dir / "artifact_save_summary.json", "w", encoding="utf8") as fh:
                json.dump(save_results, fh, indent=2)
        except Exception:
            LOG.debug("Could not write artifact_save_summary.json (fallback).")
    except Exception as e:
        LOG.exception("Saving canonical artifacts failed: %s", e)

    LOG.info("Model run complete. Artifacts written to %s and %s", models_dir, reports_dir)


if __name__ == "__main__":
    main()
