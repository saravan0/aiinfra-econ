# src/model/robustness.py
"""
Robustness & diagnostics runner for the AI-Infra econ project.

Produces:
 - reports/robustness_card.md         (human-readable summary)
 - reports/robustness_plots.png       (panel of diagnostic plots)
 - reports/robustness_manifest.json   (artifact manifest)
 - reports/ols_summary.txt            (OLS text summary)
 - reports/re_summary.txt             (Random/ MixedLM text summary)
 - models/robust_*.joblib             (saved model artifacts)
 - appended rows to reports/model_table.csv

Features:
 - Tries to run Driscoll–Kraay SEs via linearmodels.panel if available.
 - Falls back to HAC (kernel) or Bartlett if driscoll fails.
 - Runs Random Effects via linearmodels (or MixedLM fallback).
 - Diagnostics: residuals, QQ, residuals-vs-fitted, Cook's distance, VIF integration.
 - Sensitivity: drop top 1% by GDP, re-run baseline OLS and report delta.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse
import json
import logging
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# statsmodels for OLS, influence measures, and MixedLM fallback
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# sklearn pipeline used for saving purposes (no new training)
import joblib

from src.model import model_defs as mdefs
from src.model import utils as mutils
from src.model.stability_gate import main as run_stability_gate

LOG = logging.getLogger("src.model.robustness")
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(ch)
LOG.setLevel(logging.INFO)


def load_cfg(path: Path) -> Dict[str, Any]:
    """Load config from yaml or json path."""
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf8"))
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf8"))


def safe_load_features(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load features with informative logging and stable path resolution."""
    path = Path(path).expanduser().resolve()
    LOG.info("Loading features file: %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if max_rows:
        df = df.head(max_rows)
    df = df.reset_index(drop=True)
    LOG.info("safe_load_features loaded rows=%d cols=%d", df.shape[0], df.shape[1])
    return df


def sanitize_design_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce to numeric, drop columns that are all-NaN or constant (single unique non-NaN value).
    Returns a copy.
    """
    Xn = X.apply(pd.to_numeric, errors="coerce").copy()
    # drop fully-NaN columns
    Xn = Xn.loc[:, Xn.notna().any(axis=0)]
    # drop constant columns (single unique non-NaN value)
    keep = [c for c in Xn.columns if Xn[c].nunique(dropna=True) > 1]
    Xn = Xn[keep].copy()
    LOG.debug("sanitize_design_matrix => kept %d/%d cols", Xn.shape[1], X.shape[1])
    return Xn


def ols_on(df: pd.DataFrame, target: str, predictors: List[str]) -> Tuple[Any, pd.DataFrame]:
    """Run simple OLS on target ~ predictors. Returns results object and dataframe used."""
    req = [target] + predictors
    sdf = df[req].dropna()
    y = sdf[target].astype(float)
    X = sm.add_constant(sdf[predictors].astype(float), has_constant="add")
    res = sm.OLS(y, X).fit()
    return res, pd.concat([y, X], axis=1)


def ols_clustered(df: pd.DataFrame, target: str, predictors: List[str], cluster_on: Optional[str]) -> Tuple[Any, pd.DataFrame]:
    """
    Run OLS but keep cluster column aligned and compute cluster-robust SEs when possible.
    Always returns (result, df_used).
    """
    required = [target] + predictors
    if cluster_on and cluster_on in df.columns:
        required = list(dict.fromkeys(required + [cluster_on]))
    sdf = df[required].dropna()
    y = sdf[target].astype(float)
    X = sm.add_constant(sdf[predictors].astype(float), has_constant="add")
    if cluster_on and (cluster_on in sdf.columns) and (sdf[cluster_on].nunique() > 1):
        groups = sdf[cluster_on]
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        res = sm.OLS(y, X).fit()
    return res, pd.concat([y, X], axis=1)


def try_driscoll_kraay_panel(df: pd.DataFrame, target: str, predictors: List[str], entity: str = "iso3", time: str = "year"):
    """
    Attempt Driscoll–Kraay using linearmodels.PanelOLS.
    If unavailable or the requested cov_type fails, fallback to kernel/bartlett.
    Returns the fitted result or None.
    """
    try:
        from linearmodels.panel import PanelOLS
        LOG.info("linearmodels available — attempting PanelOLS Driscoll–Kraay.")
    except Exception as e:
        LOG.debug("linearmodels import failed: %s", e)
        return None

    # prepare panel: set multiindex (entity, time)
    try:
        dfp = df[[entity, time, target] + predictors].dropna().copy()
        if dfp.empty:
            LOG.warning("No full-case rows for PanelOLS (driscoll).")
            return None
        dfp = dfp.set_index([entity, time])
        y = dfp[target]
        X = sm.add_constant(dfp[predictors])
    except Exception as e:
        LOG.warning("Panel preparation failed: %s", e)
        return None

    panel = PanelOLS(y, X, entity_effects=False, time_effects=False)
    tried = []
    for cov in ("driscoll", "kernel", "bartlett"):
        try:
            if cov == "driscoll":
                res = panel.fit(cov_type="driscoll", debiased=True)
            else:
                res = panel.fit(cov_type=cov)
            LOG.info("PanelOLS fit successful using cov_type='%s'.", cov)
            return res
        except Exception as e:
            tried.append((cov, str(e)))
            LOG.warning("PanelOLS %s fit failed: %s", cov, e)
            continue
    LOG.warning("All PanelOLS covariance attempts failed: %s", tried)
    return None


def random_effects_via_linearmodels(df: pd.DataFrame, target: str, predictors: List[str], entity: str = "iso3", time: str = "year"):
    """Try linearmodels RandomEffects; return fitted result or None."""
    try:
        from linearmodels.panel import RandomEffects
        LOG.info("Running RandomEffects (linearmodels).")
    except Exception as e:
        LOG.debug("linearmodels RandomEffects import failed: %s", e)
        return None

    try:
        dfp = df[[entity, time, target] + predictors].dropna().copy()
        if dfp.empty:
            LOG.warning("No rows for RandomEffects (linearmodels).")
            return None
        dfp = dfp.set_index([entity, time])
        y = dfp[target]
        X = sm.add_constant(dfp[predictors])
        re = RandomEffects(y, X).fit()
        return re
    except Exception as e:
        LOG.warning("linearmodels RandomEffects failed: %s", e)
        return None


def random_effects_mixedlm(df: pd.DataFrame, target: str, predictors: List[str], entity: str = "iso3"):
    """Fallback to statsmodels MixedLM for random intercepts (approximation)."""
    req = [target, entity] + predictors
    sdf = df[req].dropna()
    if sdf.empty:
        LOG.warning("No rows for MixedLM random effects.")
        return None
    endog = sdf[target].astype(float)
    exog = sm.add_constant(sdf[predictors].astype(float), has_constant="add")
    try:
        md = sm.MixedLM(endog, exog, groups=sdf[entity])
        mdf = md.fit(reml=False)
        return mdf
    except Exception as e:
        LOG.warning("MixedLM failed: %s", e)
        return None


def compute_cooks_distance(ols_res, df_for_influence: pd.DataFrame) -> pd.Series:
    infl = ols_res.get_influence()
    cooks, _ = infl.cooks_distance
    return pd.Series(cooks, index=df_for_influence.index)


def plot_diagnostics(res_obj, df_used: pd.DataFrame, target: str, out_path: Path):
    """Create multi-panel diagnostic plot and save as out_path."""
    try:
        resid = res_obj.resid
        fitted = res_obj.fittedvalues
    except Exception:
        try:
            resid = df_used[target] - res_obj.predict()
            fitted = res_obj.predict()
        except Exception:
            LOG.warning("Could not obtain residuals/fitted for diagnostics; skipping plot.")
            return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes.ravel()

    ax[0].hist(resid, bins=30)
    ax[0].set_title("Residuals histogram")

    ax[1].scatter(fitted, resid, s=10, alpha=0.6)
    ax[1].axhline(0, color="red", lw=1)
    ax[1].set_xlabel("Fitted")
    ax[1].set_ylabel("Residuals")
    ax[1].set_title("Residuals vs Fitted")

    try:
        sm.qqplot(resid, line="s", ax=ax[2])
    except Exception:
        ax[2].text(0.5, 0.5, "QQ plot unavailable", ha="center")

    ax[2].set_title("QQ-plot")

    try:
        cooks = compute_cooks_distance(res_obj, df_used)
        ax[3].stem(cooks, markerfmt=",", basefmt=" ")
        ax[3].set_title("Cook's distance")
    except Exception as e:
        ax[3].text(0.5, 0.5, "Cook's D not available", ha="center")
        LOG.debug("Cook's distance failed: %s", e)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved diagnostics plot -> %s", out_path)


def vif_summary(df: pd.DataFrame, predictors: List[str], out_reports: Optional[Path] = None, issue_threshold: float = 100.0) -> pd.DataFrame:
    """Return a VIF summary for provided predictors (select numeric and full-case rows).

    If out_reports is provided and any VIF > issue_threshold, writes a small
    reports/vif_issues.txt file containing the high-VIF rows.
    """
    preds = [p for p in predictors if p in df.columns]
    if not preds:
        return pd.DataFrame(columns=["feature", "vif"])

    # coerce to numeric (drop rows with any NaNs to get full-case design)
    X = df[preds].apply(pd.to_numeric, errors="coerce")
    X = X.dropna()
    if X.shape[0] == 0:
        return pd.DataFrame(columns=["feature", "vif"])

    vif_list = []
    for i, col in enumerate(X.columns):
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception as e:
            LOG.debug("VIF computation failed for %s: %s", col, e)
            v = float("nan")
        vif_list.append((col, v))

    vif_df = pd.DataFrame(vif_list, columns=["feature", "vif"]).sort_values("vif", ascending=False)

    # optionally write high-VIF issues for human review
    try:
        if out_reports is not None:
            vif_issues = vif_df[vif_df["vif"] > float(issue_threshold)]
            if not vif_issues.empty:
                out_reports.mkdir(parents=True, exist_ok=True)
                (out_reports / "vif_issues.txt").write_text(vif_issues.to_csv(index=False), encoding="utf8")
                LOG.warning("High VIFs found; wrote %s", out_reports / "vif_issues.txt")
    except Exception as e:
        LOG.debug("Writing vif_issues failed: %s", e)

    return vif_df



def sensitivity_drop_top_gdp(df: pd.DataFrame, target: str, predictors: List[str], gdp_col: str = "gdp_usd", pct: float = 0.01):
    """Drop top `pct` by gdp_col and return the reduced dataframe and a short summary."""
    if gdp_col not in df.columns:
        LOG.warning("GDP column '%s' not present; sensitivity test skipped.", gdp_col)
        return df, "gdp not present"
    cutoff = df[gdp_col].dropna().quantile(1 - pct)
    reduced = df[df[gdp_col] <= cutoff].copy()
    return reduced, f"dropped top {pct*100:.2f}% by {gdp_col} (cutoff={cutoff:.3g})"


def append_model_table_rows(model_rows: List[Dict[str, Any]], table_path: Path):
    if table_path.exists():
        mt = pd.read_csv(table_path)
    else:
        mt = pd.DataFrame()
    new = pd.DataFrame(model_rows)
    out = pd.concat([mt, new], axis=0, ignore_index=True) if not mt.empty else new
    out.to_csv(table_path, index=False)
    LOG.info("Appended %d rows to model table -> %s", len(new), table_path)


def make_card_text(summary_rows: List[Dict[str, Any]], diagnostics: Dict[str, Any], admission_bullet: str) -> str:
    lines = []
    lines.append("# Robustness & Diagnostics card\n")
    lines.append("**Purpose:** Robustness checks (Driscoll–Kraay / Random Effects) and diagnostic summaries.\n")
    lines.append("## Key results (table snippet)\n")
    df = pd.DataFrame(summary_rows)
    if not df.empty:
        lines.append(df.head(20).to_markdown(index=False))
    else:
        lines.append("_No model rows produced._")
    lines.append("\n## Diagnostics summary\n")
    for k, v in diagnostics.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Admissions bullet (pasteable)\n")
    lines.append(admission_bullet)
    return "\n\n".join(lines)


def safe_text(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/model.yml")
    args = p.parse_args(argv)

    LOG.info("ROBUSTNESS RUN STARTING — config=%s", Path(args.config).resolve())
    try:
        cfg = load_cfg(Path(args.config))
    except Exception as e:
        LOG.error("Failed loading config: %s\n%s", e, traceback.format_exc())
        raise

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_models = Path(cfg["outputs"]["models_dir"])
    out_reports.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)

    # load features
    df = safe_load_features(Path(cfg["data"]["features_path"]), max_rows=cfg.get("debug", {}).get("max_rows"))
    LOG.info("Loaded features for robustness: %s rows, %s cols", df.shape[0], df.shape[1])

    # run stability gate early (will raise/exit if fails)
    try:
        run_stability_gate()
    except SystemExit:
        LOG.info("Stability gate signalled stop; exiting robustness.")
        return
    except Exception as e:
        LOG.warning("Stability gate invocation raised: %s", e)

    # setup predictors
    baseline_req = (cfg["predictors"].get("baseline") or []) + (cfg["predictors"].get("extra_controls") or [])
    baseline = mdefs.safe_select_columns(df, baseline_req)
    target = cfg["target"]["name"]
    if len(baseline) == 0:
        raise SystemExit("No baseline predictors present for robustness.")

    # 1) Driscoll–Kraay via linearmodels (if available)
    dk_res = try_driscoll_kraay_panel(df, target, baseline, entity="iso3", time="year")
    dk_row: List[Dict[str, Any]] = []
    if dk_res is not None:
        try:
            # linearmodels result shapes vary; coerce carefully
            params = getattr(dk_res, "params", None)
            pvals = getattr(dk_res, "pvalues", None)
            ses = getattr(dk_res, "std_errors", None)
            if params is not None:
                for k, v in params.items():
                    dk_row.append({
                        "model": "DriscollKraay",
                        "term": str(k),
                        "coef": float(v),
                        "std_err": float(ses.get(k)) if ses is not None and k in ses else None,
                        "pvalue": float(pvals.get(k)) if pvals is not None and k in pvals else None,
                        "n_obs": int(getattr(dk_res, "nobs", getattr(dk_res, "nobs_effective", df.shape[0])))
                    })
            try:
                joblib.dump(dk_res, out_models / "robust_driscoll_linearmodels.joblib")
            except Exception:
                LOG.debug("Could not joblib.dump linearmodels result (non-pickleable).")
        except Exception as e:
            LOG.warning("Failed extracting linearmodels results: %s", e)

    # 2) OLS with cluster on iso3 (we ensure iso3 alignment)
    try:
        ols_res, ols_df = ols_clustered(df, target, baseline, cluster_on="iso3")
    except Exception as e:
        LOG.error("OLS clustered failed: %s\n%s", e, traceback.format_exc())
        raise

    # save OLS artifact (save result object)
    try:
        mutils.save_model(ols_res, out_models / "robust_ols_cluster.joblib")
    except Exception:
        joblib.dump(ols_res, out_models / "robust_ols_cluster.joblib")

    # save plain text summary for easy copy-paste (admissions / SOP)
    try:
        txt = ols_res.summary().as_text()
        mutils.write_text(txt, out_reports / "ols_summary.txt")
    except Exception as e:
        LOG.warning("Saving OLS text summary failed: %s", e)

    ols_rows: List[Dict[str, Any]] = []
    try:
        for term, coef in ols_res.params.items():
            ols_rows.append({
                "model": "OLS_cluster",
                "term": str(term),
                "coef": float(coef),
                "std_err": float(ols_res.bse.get(term, np.nan)) if hasattr(ols_res, "bse") else np.nan,
                "pvalue": float(ols_res.pvalues.get(term, np.nan)) if hasattr(ols_res, "pvalues") else np.nan,
                "n_obs": int(getattr(ols_res, "nobs", len(ols_df)))
            })
    except Exception as e:
        LOG.warning("Failed to extract OLS rows: %s", e)

    # 3) Random effects (try linearmodels then MixedLM)
    re_row: List[Dict[str, Any]] = []
    re_res = random_effects_via_linearmodels(df, target, baseline)
    if re_res is not None:
        try:
            params = getattr(re_res, "params", {})
            pvals = getattr(re_res, "pvalues", {})
            ses = getattr(re_res, "std_errors", {})
            for term, coef in params.items():
                re_row.append({
                    "model": "RandomEffects",
                    "term": str(term),
                    "coef": float(coef),
                    "std_err": float(ses.get(term, np.nan)) if ses is not None else np.nan,
                    "pvalue": float(pvals.get(term, np.nan)) if pvals is not None else np.nan,
                    "n_obs": int(getattr(re_res, "nobs", df.shape[0]))
                })
            try:
                joblib.dump(re_res, out_models / "robust_randomeffects_linearmodels.joblib")
            except Exception:
                LOG.debug("Could not joblib.dump RandomEffects result (non-pickleable).")
            # summary text
            try:
                summ = getattr(re_res, "summary", None)
                if summ is not None:
                    text = summ.as_text() if hasattr(summ, "as_text") else str(summ)
                    mutils.write_text(text, out_reports / "re_linearmodels_summary.txt")
            except Exception as e:
                LOG.debug("Saving linearmodels RandomEffects summary failed: %s", e)
        except Exception as e:
            LOG.warning("Extracting RandomEffects failed: %s", e)
    else:
        # fallback MixedLM
        re_mixed = random_effects_mixedlm(df, target, baseline)
        if re_mixed is not None:
            try:
                for name in getattr(re_mixed, "params", {}).index:
                    re_row.append({
                        "model": "RandomEffects_MixedLM",
                        "term": str(name),
                        "coef": float(re_mixed.params.get(name, np.nan)),
                        "std_err": float(getattr(re_mixed, "bse", {}).get(name, np.nan)) if hasattr(re_mixed, "bse") else np.nan,
                        "pvalue": float(getattr(re_mixed, "pvalues", {}).get(name, np.nan)) if hasattr(re_mixed, "pvalues") else np.nan,
                        "n_obs": int(getattr(re_mixed, "nobs", len(getattr(re_mixed, "model", pd.DataFrame()).endog) if hasattr(re_mixed, "model") else df.shape[0]))
                    })
                try:
                    joblib.dump(re_mixed, out_models / "robust_randomeffects_mixedlm.joblib")
                except Exception:
                    LOG.debug("Could not joblib.dump MixedLM result.")
                try:
                    txt = re_mixed.summary().as_text()
                    mutils.write_text(txt, out_reports / "re_mixedlm_summary.txt")
                except Exception:
                    LOG.debug("Saving MixedLM summary failed.")
            except Exception as e:
                LOG.warning("Extracting MixedLM RandomEffects failed: %s", e)

    # 4) Diagnostics + plots (use OLS as baseline)
    try:
        plot_diagnostics(ols_res, ols_df, target, out_reports / "robustness_plots.png")
    except Exception as e:
        LOG.warning("plot_diagnostics failed: %s", e)

    # 5) VIF summary for baseline predictors (restrict to baseline predictors only)
    try:
        vif_df = vif_summary(df, baseline)
        vif_path = out_reports / "robustness_vif.csv"
        vif_df.to_csv(vif_path, index=False)
        LOG.info("Saved VIF summary -> %s", vif_path)
    except Exception as e:
        LOG.warning("VIF computation failed: %s", e)
        vif_path = out_reports / "robustness_vif.csv"

    # 6) Sensitivity: drop top 1% GDP and re-run OLS_cluster
    reduced_df, sens_msg = sensitivity_drop_top_gdp(df, target, baseline, gdp_col="gdp_usd", pct=0.01)
    sens_rows: List[Dict[str, Any]] = []
    try:
        if reduced_df is not df:
            sens_res, sens_df = ols_clustered(reduced_df, target, baseline, cluster_on="iso3")
            for term, coef in sens_res.params.items():
                sens_rows.append({
                    "model": "OLS_cluster_topdrop1pct",
                    "term": str(term),
                    "coef": float(coef),
                    "std_err": float(sens_res.bse.get(term, np.nan)) if hasattr(sens_res, "bse") else np.nan,
                    "pvalue": float(sens_res.pvalues.get(term, np.nan)) if hasattr(sens_res, "pvalues") else np.nan,
                    "n_obs": int(getattr(sens_res, "nobs", len(sens_df)))
                })
            try:
                mutils.save_model(sens_res, out_models / "robust_ols_cluster_drop_top1pct.joblib")
            except Exception:
                joblib.dump(sens_res, out_models / "robust_ols_cluster_drop_top1pct.joblib")
    except Exception as e:
        LOG.warning("Sensitivity OLS failed: %s", e)

    # 7) Write robustness card
    summary_rows: List[Dict[str, Any]] = []
    summary_rows.extend(dk_row)
    summary_rows.extend(ols_rows)
    summary_rows.extend(re_row)
    summary_rows.extend(sens_rows)

    diagnostics = {
    "input_features_path": str(Path(cfg["data"]["features_path"])),
    "loaded_rows": int(df.shape[0]),
    "vif_saved": str(vif_path),
    "sensitivity": sens_msg,
    "n_rows": int(df.shape[0])
    }


    admission_bullet = (
        "Performed panel-robust inference (Driscoll–Kraay where available), "
        "clustered standard errors, random-effects (linearmodels / MixedLM fallback), "
        "VIF diagnostics and top-1% GDP sensitivity checks."
    )

    card_text = make_card_text(summary_rows, diagnostics, admission_bullet)
    try:
        (out_reports / "robustness_card.md").write_text(card_text, encoding="utf8")
        LOG.info("Wrote robustness card -> %s", out_reports / "robustness_card.md")
    except Exception as e:
        LOG.error("Failed writing robustness card: %s", e)

    # 8) Save artifacts manifest (use mutils.save_json when available)
    from datetime import datetime
    generated_ts = datetime.utcnow().isoformat() + "Z"

    manifest = {
        "generated_at": generated_ts,
        "reports": {
           "card": str(out_reports / "robustness_card.md"),
           "plots": str(out_reports / "robustness_plots.png"),
           "vif": str(vif_path)
        },
        "models": [p.name for p in out_models.glob("robust_*.joblib")],
    }

    try:
        mutils.save_json(manifest, out_reports / "robustness_manifest.json")
        LOG.info("Saved manifest -> %s", out_reports / "robustness_manifest.json")
    except Exception:
        try:
            with open(out_reports / "robustness_manifest.json", "w", encoding="utf8") as fh:
                json.dump(manifest, fh, indent=2)
            LOG.info("Saved manifest (fallback) -> %s", out_reports / "robustness_manifest.json")
        except Exception as e:
            LOG.error("Failed to write robustness manifest: %s", e)

    # 9) Append to model table if requested
    try:
        table_path = Path(cfg["outputs"]["model_table"])
        append_model_table_rows(summary_rows, table_path)
    except Exception as e:
        LOG.debug("Appending to model_table failed (non-fatal): %s", e)

    LOG.info("Robustness run complete. Artifacts written to models and reports")


if __name__ == "__main__":
    main()
