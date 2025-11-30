#!/usr/bin/env python3
"""
plot_fe_diagnostics — Research-grade fixed-effects diagnostics & robustness checker.

Produces (under reports/plot_fe_diagnostics_research/):
  - files/*.png|.pdf|.svg         (diagnostic figures)
  - files/standardized/*          (copied standardized jsons, if present)
  - files/artifacts/*             (copied joblib artifacts, if present)
  - files/fe_diagnostics_result.joblib
  - files/robustness/*.csv/.json  (robustness tables)
  - meta.json                     (provenance: args, hashes, produced files, git)
  - summary/fe_diagnostics_summary.json (human-readable summary)

Usage:
  python scripts/plot_fe_diagnostics.py --vars trade_exposure gov_index_zmean inflation_consumer_prices_pct
  python scripts/plot_fe_diagnostics.py --vars trade_exposure ... --cluster-col iso3 --winsor-pcts 0.01 0.02
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

# ---- logging / style ----
LOG = logging.getLogger("plot_fe_diagnostics_research")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

# ---- paths & defaults ----
ROOT = Path(".")
OUT_BASE = ROOT / "reports" / "plot_fe_diagnostics_research"
OUT_FILES = OUT_BASE / "files"
OUT_STD = OUT_FILES / "standardized"
OUT_ARTIFACTS = OUT_FILES / "artifacts"
OUT_PARTIALS = OUT_FILES / "partials"
OUT_ROB = OUT_FILES / "robustness"
SUMMARY_DIR = OUT_BASE / "summary"

for p in (OUT_FILES, OUT_STD, OUT_ARTIFACTS, OUT_PARTIALS, OUT_ROB, SUMMARY_DIR):
    p.mkdir(parents=True, exist_ok=True)

FEATURES_DEFAULT = Path("data") / "processed" / "features_lean_imputed.csv"
VARS_DEFAULT = ["trade_exposure", "gov_index_zmean", "inflation_consumer_prices_pct"]


# ---- utilities ----
def sha256_of_file(p: Optional[Path]) -> Optional[str]:
    if p is None or not Path(p).exists():
        return None
    try:
        h = hashlib.sha256()
        with Path(p).open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf8").strip()
    except Exception:
        return None


def safe_json_load(p: Path) -> Optional[Any]:
    try:
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf8"))
    except Exception as e:
        LOG.warning("safe_json_load failed for %s: %s", p, e)
        return None


def safe_copy_if_exists(src: Path, dst_dir: Path) -> Optional[str]:
    try:
        if not src.exists():
            return None
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return str(dst)
    except Exception as e:
        LOG.warning("safe_copy_if_exists %s -> %s failed: %s", src, dst_dir, e)
        return None


def safe_to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            continue
    return df



# ---- model / design helpers ----
def build_fe_design(df: pd.DataFrame, predictors: List[str], entity_col: str = "iso3",
                    drop_first: bool = True, min_obs_for_iso: int = 1) -> Tuple[pd.DataFrame, List[str]]:
    preds = [p for p in predictors if p in df.columns]
    if not preds:
        raise ValueError(f"No predictors found in features for requested predictors: {predictors}")

    # group small entities into OTHER if needed
    df_local = df
    if entity_col in df_local.columns and min_obs_for_iso > 1:
        counts = df_local[entity_col].value_counts(dropna=True)
        small = counts[counts < min_obs_for_iso].index.tolist()
        if small:
            df_local = df_local.copy()
            df_local[entity_col] = df_local[entity_col].fillna("OTHER").astype(str)
            df_local.loc[df_local[entity_col].isin(small), entity_col] = "OTHER"

    fe = pd.DataFrame(index=df_local.index)
    if entity_col in df_local.columns:
        fe = pd.get_dummies(df_local[entity_col].astype(str), prefix="FE", drop_first=drop_first)

    num_block = df_local[preds].apply(pd.to_numeric, errors="coerce")
    X_df = pd.concat([num_block, fe], axis=1)
    return X_df, preds


def fit_model(X: pd.DataFrame, y: pd.Series):
    Xc = sm.add_constant(X, has_constant="add")
    mask = Xc.notna().all(axis=1) & y.notna()
    Xc_clean = Xc.loc[mask]
    y_clean = y.loc[mask]
    if len(y_clean) == 0:
        raise ValueError("No full-case rows remain after dropna.")
    res = sm.OLS(y_clean.astype(float), Xc_clean.astype(float)).fit()
    return res, Xc_clean, y_clean


# ---- plotting & file-writer ----
def _write_fig(fig, path_prefix: Path):
    for ext in ("png", "pdf", "svg"):
        p = path_prefix.with_suffix("." + ext)
        try:
            fig.savefig(p, bbox_inches="tight", dpi=300)
            LOG.info("Wrote %s", p)
        except Exception as e:
            LOG.warning("Failed to write figure %s: %s", p, e)
    plt.close(fig)


def plot_resid_vs_fitted(res, outpath: Path):
    fitted = res.fittedvalues
    resid = res.resid
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(fitted, resid, s=20, alpha=0.7, edgecolor="k", linewidth=0.2)
    ax.axhline(0, color="0.3", linestyle="--")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    _write_fig(fig, outpath)


def plot_qq(res, outpath: Path):
    # compute studentized residuals and drop non-finite values before plotting
    infl = OLSInfluence(res)
    std_resid = np.asarray(infl.resid_studentized_internal)

    # filter non-finite values (can occur for exact-fit rows / zero sigma / leverage==1)
    finite_mask = np.isfinite(std_resid)
    std_resid_finite = std_resid[finite_mask]

    if std_resid_finite.size < 3:
        LOG.warning("Not enough finite studentized residuals for Q-Q plot (n=%d). Skipping plot.", int(std_resid_finite.size))
        return

    # use statsmodels qqplot but pass the filtered data and avoid refit if problematic
    try:
        fig = sm.graphics.qqplot(std_resid_finite, line="45", fit=True)
        fig.set_size_inches(6.5, 6.5)
        plt.title("Q-Q plot (studentized residuals)")
        _write_fig(fig, outpath)
    except Exception as e:
        LOG.warning("Q-Q plot failed: %s — attempting fallback (plot only).", e)
        # fallback: simple theoretical vs sample quantile scatter (no distribution fit)
        from scipy import stats
        prob = (np.arange(1, len(std_resid_finite) + 1) - 0.5) / len(std_resid_finite)
        theo_q = stats.norm.ppf(prob)
        sample_q = np.sort(std_resid_finite)
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(theo_q, sample_q, s=20, alpha=0.7)
        minv = min(theo_q.min(), sample_q.min())
        maxv = max(theo_q.max(), sample_q.max())
        ax.plot([minv, maxv], [minv, maxv], color="red", linewidth=1)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title("Q-Q plot (studentized residuals) — fallback")
        _write_fig(fig, outpath)



def plot_scale_location(res, outpath: Path):
    fitted = res.fittedvalues
    infl = OLSInfluence(res)
    std_resid = infl.resid_studentized_internal
    yvals = np.sqrt(np.abs(std_resid))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(fitted, yvals, s=20, alpha=0.7, edgecolor="k", linewidth=0.2)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Sqrt(|studentized residual|)")
    ax.set_title("Scale-Location")
    _write_fig(fig, outpath)


def plot_leverage_cooks(res, outpath: Path):
    infl = OLSInfluence(res)
    leverage = np.asarray(infl.hat_matrix_diag)
    cooks = np.asarray(infl.cooks_distance[0])
    std_resid = np.asarray(infl.resid_studentized_internal)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(leverage, std_resid, s=36, c=cooks, cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.25)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Cook's distance")

    n = int(res.nobs)
    p = float(res.df_model) + 1.0
    lev_thresh = 2 * p / n
    ax.axvline(lev_thresh, color="red", linestyle="--", lw=1, label="High leverage threshold")

    # annotate top influential points
    top_pos = np.argsort(-cooks)[:8]
    for pos_idx in top_pos:
        try:
            ax.annotate(f"#{pos_idx}", (leverage[pos_idx], std_resid[pos_idx]), textcoords="offset points", xytext=(6, -6), fontsize=8)
        except Exception:
            pass

    ax.set_xlabel("Leverage (hat)")
    ax.set_ylabel("Studentized residuals")
    ax.set_title("Leverage vs Studentized residuals (Cook's distance)")
    _write_fig(fig, outpath)


def plot_resid_hist(res, outpath: Path):
    resid = res.resid
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.histplot(resid, bins=40, kde=True, ax=ax)
    ax.set_title("Residuals distribution")
    ax.set_xlabel("Residual")
    _write_fig(fig, outpath)


def write_influence_table(res, outdir: Path, topk: int = 40) -> pd.DataFrame:
    infl = OLSInfluence(res)
    cooks = np.asarray(infl.cooks_distance[0])
    leverage = np.asarray(infl.hat_matrix_diag)
    std_resid = np.asarray(infl.resid_studentized_internal)
    row_labels = list(res.model.data.row_labels) if hasattr(res.model.data, "row_labels") else [f"pos_{i}" for i in range(len(cooks))]
    df_inf = pd.DataFrame({
        "pos_index": list(range(len(cooks))),
        "row_label": row_labels,
        "cooks_d": cooks,
        "leverage": leverage,
        "studentized_resid": std_resid,
    })
    df_inf = df_inf.sort_values("cooks_d", ascending=False).reset_index(drop=True)
    out_csv = outdir / "influence_top_by_cooks.csv"
    df_inf.head(topk).to_csv(out_csv, index=False)
    LOG.info("Wrote influence table %s", out_csv)
    return df_inf


def plot_partial_residuals(df_all, res, X_design, predictors, outdir: Path, target_col: str):
    rows_idx = X_design.index
    for pred in predictors:
        if pred not in X_design.columns:
            continue
        beta = res.params.get(pred, None)
        if beta is None:
            try:
                simple_res = sm.OLS(res.model.endog, sm.add_constant(X_design[[pred]].astype(float))).fit()
                beta = float(simple_res.params[1])
            except Exception:
                continue
        xvals = X_design.loc[rows_idx, pred].astype(float)
        partial = res.resid + beta * xvals
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(xvals, partial, s=28, alpha=0.85, edgecolor="k", linewidth=0.2)
        ax.axhline(0, color="0.3", linestyle="--")
        ax.set_xlabel(pred)
        ax.set_ylabel("Partial residual (resid + beta * x)")
        ax.set_title(f"Partial residuals — {pred}")
        _write_fig(fig, outdir / f"partial_resid_{pred}")


# ---- robust se / winsor ----
def compute_hc3_se(res):
    try:
        cov_hc3 = res.get_robustcov_results(cov_type="HC3").cov_params()
        se = np.sqrt(np.diag(cov_hc3))
        return se, cov_hc3
    except Exception as e:
        LOG.warning("compute_hc3_se failed: %s", e)
        return None, None


def compute_cluster_se(res, X_design_clean, cluster_series):
    try:
        clusters = cluster_series.loc[X_design_clean.index]
        cov = res.get_robustcov_results(cov_type="cluster", groups=clusters).cov_params()
        se = np.sqrt(np.diag(cov))
        return se, cov
    except Exception as e:
        LOG.warning("compute_cluster_se failed: %s", e)
        return None, None


def winsorize_df(df, cols, lower_pct, upper_pct):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            low = df2[c].quantile(lower_pct)
            high = df2[c].quantile(1 - upper_pct)
            df2[c] = df2[c].clip(lower=low, upper=high)
    return df2


def _safe_float(x):
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None

def write_summary(res, outdir: Path):
    infl = OLSInfluence(res)
    # capture potentially non-finite diagnostics, but convert Inf/nan -> None
    cooks_vals = np.asarray(infl.cooks_distance[0])
    leverage_vals = np.asarray(infl.hat_matrix_diag)

    cooks_d_max = None
    try:
        if np.any(np.isfinite(cooks_vals)):
            cooks_d_max = float(np.nanmax(np.where(np.isfinite(cooks_vals), cooks_vals, np.nan)))
    except Exception:
        cooks_d_max = None

    leverage_max = None
    try:
        if np.any(np.isfinite(leverage_vals)):
            leverage_max = float(np.nanmax(np.where(np.isfinite(leverage_vals), leverage_vals, np.nan)))
    except Exception:
        leverage_max = None

    summary = {
        "n_obs": int(res.nobs) if hasattr(res, "nobs") and np.isfinite(res.nobs) else None,
        "df_model": _safe_float(res.df_model),
        "aic": _safe_float(res.aic),
        "bic": _safe_float(res.bic),
        "rsquared": _safe_float(res.rsquared),
        "rsquared_adj": _safe_float(res.rsquared_adj),
        "resid_skewness": _safe_float(pd.Series(res.resid).skew()),
        "resid_kurtosis": _safe_float(pd.Series(res.resid).kurtosis()),
        "cooks_d_max": cooks_d_max,
        "leverage_max": leverage_max,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    outp = outdir / "fe_diagnostics_summary.json"
    outp.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote diagnostics summary -> %s", outp)
    return summary



def collect_robustness_rows(base_res, X_design_clean, y_clean, preds, label, cluster_col=None, cluster_series_full=None) -> Dict[str, Any]:
    row_dict = {"spec": label}
    for p in preds:
        coef = float(base_res.params.get(p, np.nan)) if p in base_res.params.index else np.nan
        se = float(base_res.bse.get(p, np.nan)) if hasattr(base_res, "bse") and p in base_res.params.index else np.nan
        pval = float(base_res.pvalues.get(p, np.nan)) if hasattr(base_res, "pvalues") and p in base_res.params.index else np.nan
        row_dict[f"{p}_coef"] = coef
        row_dict[f"{p}_se"] = se
        row_dict[f"{p}_pval"] = pval

    # HC3
    se_hc3, _ = compute_hc3_se(base_res)
    if se_hc3 is not None:
        for i, p in enumerate(base_res.params.index):
            if p in preds:
                row_dict[f"{p}_se_hc3"] = float(se_hc3[i])

    # Clustered
    if cluster_col and (cluster_series_full is not None):
        se_clust, _ = compute_cluster_se(base_res, X_design_clean, cluster_series_full)
        if se_clust is not None:
            for i, p in enumerate(base_res.params.index):
                if p in preds:
                    row_dict[f"{p}_se_cluster"] = float(se_clust[i])

    return row_dict


# ---- CLI runner ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=str(FEATURES_DEFAULT), help="Features CSV (processed)")
    p.add_argument("--vars", nargs="+", required=True, help="Predictor variables to diagnose")
    p.add_argument("--target", default="gdp_growth_pct", help="Target column name")
    p.add_argument("--entity-col", default="iso3", help="Entity column for FE (e.g., iso3)")
    p.add_argument("--outdir", default=str(OUT_BASE), help="Output directory root (overridden by script defaults)")
    p.add_argument("--min-obs-fe", type=int, default=1, help="Minimum observations for iso3 to keep separate; smaller grouped to OTHER")
    p.add_argument("--cluster-col", default=None, help="Optional: column to cluster SE by (e.g., iso3)")
    p.add_argument("--winsor-pcts", nargs="*", type=float, default=[0.01, 0.02], help="Winsorization fractions to try (e.g., 0.01 0.02)")
    args = p.parse_args()

    LOG.info("Starting FE diagnostics (research-grade). Vars=%s target=%s", args.vars, args.target)

    features_path = Path(args.features)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path, low_memory=False)
    df = safe_to_numeric_df(df)
    LOG.info("Loaded features: %d rows, %d cols", df.shape[0], df.shape[1])

    # copy standardized artifact if present (best-effort)
    std_dir = Path("outputs") / "standardized"
    produced_files: List[str] = []
    for v in args.vars:
        s = std_dir / f"{v}_standardized.json"
        cp = safe_copy_if_exists(s, OUT_STD)
        if cp:
            produced_files.append(cp)

    # build FE design
    X_df, used_preds = build_fe_design(df, args.vars, entity_col=args.entity_col, drop_first=True, min_obs_for_iso=args.min_obs_fe)

    # align y
    if args.target not in df.columns:
        raise ValueError(f"Target {args.target} not in features.")
    y = df[args.target]

    LOG.info("Fitting FE OLS with predictors: %s", used_preds)
    res, X_design_clean, y_clean = fit_model(X_df, y)
    LOG.info("Model fitted. n_obs=%d df_model=%s", int(res.nobs), res.df_model)

    # save main model artifact (joblib)
    try:
        import joblib
        joblib.dump(res, OUT_FILES / "fe_diagnostics_result.joblib")
        produced_files.append(str(OUT_FILES / "fe_diagnostics_result.joblib"))
    except Exception:
        LOG.info("joblib not available or save failed (optional).")

    # basic diagnostics
    plot_resid_vs_fitted(res, OUT_FILES / "resid_vs_fitted")
    plot_qq(res, OUT_FILES / "qq_studentized")
    plot_scale_location(res, OUT_FILES / "scale_location")
    plot_leverage_cooks(res, OUT_FILES / "leverage_cooks")
    plot_resid_hist(res, OUT_FILES / "resid_hist")

    # influence + partials
    df_inf = write_influence_table(res, OUT_FILES)
    produced_files.append(str(OUT_FILES / "influence_top_by_cooks.csv"))
    plot_partial_residuals(df, res, X_design_clean, used_preds, OUT_PARTIALS, args.target)

    # summary
    summary = write_summary(res, SUMMARY_DIR)
    produced_files.append(str(SUMMARY_DIR / "fe_diagnostics_summary.json"))

    # robustness collection
    robustness_rows = []
    base_label = "FE_full"
    LOG.info("Collecting robustness metrics (HC3 and optionally cluster).")
    cluster_series_full = None
    if args.cluster_col and args.cluster_col in df.columns:
        cluster_series_full = df[args.cluster_col]

    base_row = collect_robustness_rows(res, X_design_clean, y_clean, used_preds, base_label,
                                       cluster_col=args.cluster_col, cluster_series_full=cluster_series_full)
    robustness_rows.append(base_row)

    # HC3 annotated into base_row done in collect_robustness_rows

    # clustered
    if args.cluster_col and cluster_series_full is not None:
        LOG.info("Attempting cluster-robust SEs on %s", args.cluster_col)
        try:
            se_clust, cov_clust = compute_cluster_se(res, X_design_clean, cluster_series_full)
            if se_clust is not None:
                # annotate base_row (already handled inside collect_robustness_rows)
                LOG.info("Clustered SE computed.")
        except Exception as e:
            LOG.warning("Clustered SE computation failed: %s", e)

    # winsorization sensitivity
    for w in (args.winsor_pcts or []):
        try:
            pct = float(w)
            LOG.info("Applying winsorization at pct=%.4f", pct)
            df_w = df.copy()
            for col in used_preds + [args.target]:
                if col in df_w.columns:
                    low = df_w[col].quantile(pct)
                    high = df_w[col].quantile(1 - pct)
                    df_w[col] = df_w[col].clip(lower=low, upper=high)
            Xw_df, _ = build_fe_design(df_w, used_preds, entity_col=args.entity_col, drop_first=True, min_obs_for_iso=args.min_obs_fe)
            yw = df_w[args.target]
            res_w, Xw_clean, yw_clean = fit_model(Xw_df, yw)
            label = f"winsor_{int(pct*100)}pct"
            LOG.info("Winsorized fit complete (%s). n=%d", label, int(res_w.nobs))
            row = collect_robustness_rows(res_w, Xw_clean, yw_clean, used_preds, label,
                                          cluster_col=args.cluster_col, cluster_series_full=cluster_series_full)
            robustness_rows.append(row)
            try:
                import joblib
                joblib.dump(res_w, OUT_ROB / f"res_winsor_{int(pct*100)}.joblib")
                produced_files.append(str(OUT_ROB / f"res_winsor_{int(pct*100)}.joblib"))
            except Exception:
                pass
        except Exception as e:
            LOG.warning("Winsorization step %s failed: %s", w, e)

    # write robustness outputs
    rob_df = pd.DataFrame(robustness_rows)
    rob_csv = OUT_ROB / "robustness_summary.csv"
    rob_json = OUT_ROB / "robustness_summary.json"
    rob_df.to_csv(rob_csv, index=False)
    rob_json.write_text(rob_df.to_json(orient="records", indent=2), encoding="utf8")
    produced_files.extend([str(rob_csv), str(rob_json)])
    LOG.info("Wrote robustness summary -> %s and %s", rob_csv, rob_json)

    # final provenance meta
    meta = {
        "script": str(Path(__file__).resolve()),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "args": vars(args),
        "features_file": str(features_path),
        "features_sha256": sha256_of_file(features_path),
        "produced_files": produced_files,
        "git_commit": git_commit_hash(),
    }
    meta_path = OUT_BASE / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    LOG.info("Wrote metadata -> %s", meta_path)

    LOG.info("FE diagnostics & robustness complete. Files saved to: %s", OUT_BASE.resolve())
    print("Done — see", OUT_BASE.resolve())


if __name__ == "__main__":
    main()
