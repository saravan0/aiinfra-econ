#!/usr/bin/env python3
"""
scripts/plot_fe_diagnostics.py

Enhanced research-quality diagnostics for a fixed-effects specification,
with PhD-level robustness checks:
 - HC3 robust standard errors
 - Clustered standard errors (by --cluster-col)
 - Winsorization sensitivity (multiple pct thresholds)
 - Influence / Cook's table and robust plotting
 - Robustness summary CSV/JSON comparing coefficients across specifications

Usage:
    python scripts/plot_fe_diagnostics.py --vars trade_exposure gov_index_zmean inflation_consumer_prices_pct
    python scripts/plot_fe_diagnostics.py --vars trade_exposure ... --cluster-col iso3 --winsor-pcts 0.01 0.02
"""

from pathlib import Path
import argparse
import json
import logging
from datetime import datetime
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

sns.set_style("whitegrid")
LOG = logging.getLogger("fe_diag")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# -------------------------
# Helpers
# -------------------------
def safe_to_numeric_df(df: pd.DataFrame):
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    return df

def build_fe_design(df: pd.DataFrame, predictors: list, entity_col: str = "iso3", drop_first: bool = True, min_obs_for_iso: int = 1):
    """
    Build design matrix: numeric predictors + FE dummies (one-hot).
    Returns X_df (pandas.DataFrame), used_preds (list).
    """
    preds = [p for p in predictors if p in df.columns]
    if not preds:
        raise ValueError(f"No predictors found in features for requested predictors: {predictors}")

    # optionally group tiny iso3 into OTHER
    if entity_col in df.columns and min_obs_for_iso > 1:
        counts = df[entity_col].value_counts(dropna=True)
        small = counts[counts < min_obs_for_iso].index.tolist()
        if small:
            df = df.copy()
            df[entity_col] = df[entity_col].fillna("OTHER").astype(str)
            df.loc[df[entity_col].isin(small), entity_col] = "OTHER"

    # One-hot encode FE dummies
    if entity_col in df.columns:
        fe = pd.get_dummies(df[entity_col].astype(str), prefix="FE", drop_first=drop_first)
    else:
        LOG.warning("Entity column %s not found: no FE dummies added.", entity_col)
        fe = pd.DataFrame(index=df.index)

    num_block = df[preds].apply(pd.to_numeric, errors="coerce")
    X_df = pd.concat([num_block, fe], axis=1)
    return X_df, preds

def fit_model(X: pd.DataFrame, y: pd.Series):
    """Fit OLS with constant; return (res, Xc_clean, y_clean)."""
    Xc = sm.add_constant(X, has_constant="add")
    mask = Xc.notna().all(axis=1) & y.notna()
    Xc_clean = Xc.loc[mask]
    y_clean = y.loc[mask]
    if len(y_clean) == 0:
        raise ValueError("No full-case rows remain after dropna.")
    res = sm.OLS(y_clean.astype(float), Xc_clean.astype(float)).fit()
    return res, Xc_clean, y_clean

def _write_fig(fig, path_prefix: Path):
    for ext in ("png", "pdf", "svg"):
        p = path_prefix.with_suffix("." + ext)
        fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    LOG.info("Wrote figures with prefix: %s", path_prefix)

def fit_ols_line_with_ci(x, y, alpha=0.05):
    X = sm.add_constant(np.asarray(x))
    res = sm.OLS(np.asarray(y), X).fit()
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    Xpred = sm.add_constant(xx)
    pred = res.get_prediction(Xpred)
    mean = pred.predicted_mean
    ci_low, ci_high = pred.conf_int(alpha=alpha).T
    return xx, mean, ci_low, ci_high, res

# -------------------------
# Plotting functions
# -------------------------
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
    std_resid = res.get_influence().resid_studentized_internal
    fig = sm.graphics.qqplot(std_resid, line="45", fit=True)
    fig.set_size_inches(6.5, 6.5)
    plt.title("Q-Q plot (studentized residuals)")
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
    ax.set_xlabel("Leverage (hat)")
    ax.set_ylabel("Studentized residuals")
    ax.set_title("Leverage vs Studentized residuals (Cook's distance)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Cook's distance")

    n = int(res.nobs)
    p = float(res.df_model) + 1.0
    lev_thresh = 2 * p / n
    ax.axvline(lev_thresh, color="red", linestyle="--", lw=1, label="High leverage threshold")

    # annotate top influential points (positional indices)
    k = min(8, len(cooks))
    top_pos = np.argsort(-cooks)[:k]
    for pos_idx in top_pos:
        # annotate using positional index label to avoid missing-index KeyError
        ax.annotate(f"#{pos_idx}", (leverage[pos_idx], std_resid[pos_idx]), textcoords="offset points", xytext=(6, -6), fontsize=8)

    _write_fig(fig, outpath)

def plot_resid_hist(res, outpath: Path):
    resid = res.resid
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.histplot(resid, bins=40, kde=True, ax=ax)
    ax.set_title("Residuals distribution")
    ax.set_xlabel("Residual")
    _write_fig(fig, outpath)

def write_influence_table(res, outdir: Path, topk: int = 40):
    infl = OLSInfluence(res)
    cooks = np.asarray(infl.cooks_distance[0])
    leverage = np.asarray(infl.hat_matrix_diag)
    std_resid = np.asarray(infl.resid_studentized_internal)

    # row labels might be present, but we'll always include the positional index
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

def plot_partial_residuals(df, res, X_design, predictors, outdir: Path, target_col: str):
    """
    Partial residuals per predictor; compute on the rows used in the fitted model.
    Uses safe indexing: X_design.index are the fitted rows.
    """
    rows_idx = X_design.index

    for pred in predictors:
        if pred not in X_design.columns:
            LOG.warning("Predictor %s absent from design -> skip partial residual plot.", pred)
            continue

        beta = res.params.get(pred, None)
        if beta is None:
            # fallback simple slope
            try:
                simple_res = sm.OLS(res.model.endog, sm.add_constant(X_design[[pred]].astype(float))).fit()
                beta = float(simple_res.params[1])
            except Exception:
                LOG.warning("Could not estimate fallback slope for %s; skipping.", pred)
                continue

        y_fitted_rows = res.model.endog  # aligned with X_design
        resid = res.resid
        xvals = X_design.loc[rows_idx, pred].astype(float)
        partial = resid + beta * xvals

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(xvals, partial, s=28, alpha=0.85, edgecolor="k", linewidth=0.2)
        xx, mean, ci_low, ci_high, ols_r = fit_ols_line_with_ci(xvals.values, partial)
        ax.plot(xx, mean, color="#2c7bb6", lw=2, label="OLS (partial)")
        ax.fill_between(xx, ci_low, ci_high, color="#2c7bb6", alpha=0.18)
        ax.axhline(0, color="0.3", linestyle="--")
        ax.set_xlabel(pred)
        ax.set_ylabel("Partial residual (resid + beta * x)")
        ax.set_title(f"Partial residuals — {pred}")
        ax.legend()
        outp = outdir / f"partial_resid_{pred}"
        _write_fig(fig, outp)

# -------------------------
# Robustness utilities
# -------------------------
def compute_hc3_se(res):
    cov_hc3 = res.get_robustcov_results(cov_type="HC3").cov_params()
    se = np.sqrt(np.diag(cov_hc3))
    return se, cov_hc3

def compute_cluster_se(res, X_design_clean, cluster_series):
    """
    Compute cluster-robust covariance using statsmodels API.
    `cluster_series` must be aligned with X_design_clean index (same rows).
    """
    try:
        clusters = cluster_series.loc[X_design_clean.index]
        cov = res.get_robustcov_results(cov_type="cluster", groups=clusters).cov_params()
        se = np.sqrt(np.diag(cov))
        return se, cov
    except Exception as e:
        LOG.warning("Clustered SE computation failed: %s", e)
        return None, None

def winsorize_df(df, cols, lower_pct, upper_pct):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            low = df2[c].quantile(lower_pct)
            high = df2[c].quantile(1 - upper_pct)
            df2[c] = df2[c].clip(lower=low, upper=high)
    return df2

# -------------------------
# Summary writer
# -------------------------
def write_summary(res, outdir: Path):
    infl = OLSInfluence(res)
    summary = {
        "n_obs": int(res.nobs),
        "df_model": float(res.df_model),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "rsquared": float(res.rsquared),
        "rsquared_adj": float(res.rsquared_adj),
        "resid_skewness": float(pd.Series(res.resid).skew()),
        "resid_kurtosis": float(pd.Series(res.resid).kurtosis()),
        "cooks_d_max": float(np.max(infl.cooks_distance[0])),
        "leverage_max": float(np.max(infl.hat_matrix_diag)),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    outp = outdir / "fe_diagnostics_summary.json"
    outp.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote diagnostics summary -> %s", outp)
    return summary

# -------------------------
# Robustness table creation
# -------------------------
def collect_robustness_rows(base_res, X_design_clean, y_clean, preds, label, cluster_col=None, cluster_series_full=None):
    row_dict = {"spec": label}
    # Coefficients and baseline se
    for p in preds:
        coef = float(base_res.params.get(p, np.nan))
        se = float(base_res.bse.get(p, np.nan)) if hasattr(base_res, "bse") else np.nan
        pval = float(base_res.pvalues.get(p, np.nan)) if hasattr(base_res, "pvalues") else np.nan
        row_dict[f"{p}_coef"] = coef
        row_dict[f"{p}_se"] = se
        row_dict[f"{p}_pval"] = pval

    # HC3
    try:
        se_hc3, _ = compute_hc3_se(base_res)
        for i, p in enumerate(base_res.params.index):
            if p in preds:
                idx = list(base_res.params.index).index(p)
                row_dict[f"{p}_se_hc3"] = float(se_hc3[idx])
    except Exception:
        pass

    # Clustered if requested
    if cluster_col and (cluster_series_full is not None):
        try:
            se_clust, _ = compute_cluster_se(base_res, X_design_clean, cluster_series_full)
            if se_clust is not None:
                for i, p in enumerate(base_res.params.index):
                    if p in preds:
                        idx = list(base_res.params.index).index(p)
                        row_dict[f"{p}_se_cluster"] = float(se_clust[idx])
        except Exception as e:
            LOG.warning("Cluster robust se collection failed: %s", e)

    return row_dict

# -------------------------
# CLI / runner
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_lean_imputed.csv", help="Features CSV (processed)")
    p.add_argument("--vars", nargs="+", required=True, help="Predictor variables to diagnose")
    p.add_argument("--target", default="gdp_growth_pct", help="Target column name")
    p.add_argument("--entity-col", default="iso3", help="Entity column for FE (e.g., iso3)")
    p.add_argument("--outdir", default="reports/figs/diagnostics", help="Output directory for figures & tables")
    p.add_argument("--min-obs-fe", type=int, default=1, help="Minimum observations for iso3 to keep separate; smaller grouped to OTHER")
    p.add_argument("--cluster-col", default=None, help="Optional: column to cluster SE by (e.g., iso3)")
    p.add_argument("--winsor-pcts", nargs="*", type=float, default=[0.01, 0.02], help="Winsorization fractions to try (e.g., 0.01 0.02)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "partials").mkdir(exist_ok=True)
    (outdir / "robustness").mkdir(exist_ok=True)

    fpath = Path(args.features)
    if not fpath.exists():
        raise FileNotFoundError(f"Features file not found: {fpath}")

    df = pd.read_csv(fpath, low_memory=False)
    df = safe_to_numeric_df(df)
    LOG.info("Loaded features: %s rows, %s cols", df.shape[0], df.shape[1])

    # Build FE design (predictors numeric + FE dummies)
    X_df, used_preds = build_fe_design(df, args.vars, entity_col=args.entity_col, drop_first=True, min_obs_for_iso=args.min_obs_fe)

    # align y
    if args.target not in df.columns:
        raise ValueError(f"Target {args.target} not in features.")
    y = df[args.target]

    LOG.info("Fitting FE OLS with predictors: %s", used_preds)
    res, X_design_clean, y_clean = fit_model(X_df, y)

    LOG.info("Model fitted. n_obs=%d  df_model=%s", int(res.nobs), res.df_model)

    # Save main model artifact
    try:
        import joblib
        joblib.dump(res, outdir / "fe_diagnostics_result.joblib")
        LOG.info("Saved FE diagnostics model artifact.")
    except Exception:
        LOG.info("Could not save joblib artifact (optional).")

    # Basic diagnostics plots
    plot_resid_vs_fitted(res, outdir / "resid_vs_fitted")
    plot_qq(res, outdir / "qq_studentized")
    plot_scale_location(res, outdir / "scale_location")
    plot_leverage_cooks(res, outdir / "leverage_cooks")
    plot_resid_hist(res, outdir / "resid_hist")

    # influence table & partial residuals
    df_inf = write_influence_table(res, outdir, topk=40)
    plot_partial_residuals(df, res, X_design_clean, used_preds, outdir / "partials", args.target)

    # write summary for main fit
    summary = write_summary(res, outdir)

    # -------------------------
    # Robustness checks
    # -------------------------
    robustness_rows = []
    base_label = "FE_full"
    LOG.info("Collecting robustness metrics (HC3 and optionally cluster).")
    cluster_series_full = None
    if args.cluster_col and args.cluster_col in df.columns:
        cluster_series_full = df[args.cluster_col]

    base_row = collect_robustness_rows(res, X_design_clean, y_clean, used_preds, base_label, cluster_col=args.cluster_col, cluster_series_full=cluster_series_full)
    robustness_rows.append(base_row)

    # HC3 summary (we will attach HC3 SEs into the base row too)
    try:
        se_hc3, cov_hc3 = compute_hc3_se(res)
        # annotate base row with HC3 per predictor (if param present)
        for p in used_preds:
            if p in res.params.index:
                pos = list(res.params.index).index(p)
                base_row[f"{p}_se_hc3"] = float(se_hc3[pos])
    except Exception as e:
        LOG.warning("HC3 computation failed: %s", e)

    # Clustered SE (if requested)
    if args.cluster_col and cluster_series_full is not None:
        try:
            se_clust, cov_clust = compute_cluster_se(res, X_design_clean, cluster_series_full)
            if se_clust is not None:
                for p in used_preds:
                    if p in res.params.index:
                        pos = list(res.params.index).index(p)
                        base_row[f"{p}_se_cluster"] = float(se_clust[pos])
                LOG.info("Clustered SE computed for cluster col %s.", args.cluster_col)
        except Exception as e:
            LOG.warning("Clustered SE computation failed: %s", e)

    # Winsorization sensitivity
    winsor_pcts = args.winsor_pcts if args.winsor_pcts else []
    for w in winsor_pcts:
        try:
            # winsorize numeric predictors and target at symmetric tails
            pct = float(w)
            LOG.info("Applying winsorization at pct=%.3f", pct)
            numerical_predictors = used_preds.copy()
            # build a copy of df and winsorize numeric fields used in X_df building
            df_w = df.copy()
            for col in numerical_predictors + [args.target]:
                if col in df_w.columns:
                    low = df_w[col].quantile(pct)
                    high = df_w[col].quantile(1 - pct)
                    df_w[col] = df_w[col].clip(lower=low, upper=high)
            # rebuild design & refit
            Xw_df, _ = build_fe_design(df_w, used_preds, entity_col=args.entity_col, drop_first=True, min_obs_for_iso=args.min_obs_fe)
            yw = df_w[args.target]
            res_w, Xw_clean, yw_clean = fit_model(Xw_df, yw)
            label = f"winsor_{int(pct*100)}pct"
            LOG.info("Winsorized fit complete (%s). n=%d", label, int(res_w.nobs))
            # collect row
            row = collect_robustness_rows(res_w, Xw_clean, yw_clean, used_preds, label, cluster_col=args.cluster_col, cluster_series_full=cluster_series_full)
            robustness_rows.append(row)
            # save winsor fit artifact
            try:
                import joblib
                joblib.dump(res_w, outdir / "robustness" / f"res_winsor_{int(pct*100)}.joblib")
            except Exception:
                pass
        except Exception as e:
            LOG.warning("Winsorization step %s failed: %s", w, e)

    # Save robustness table (CSV + JSON)
    rob_df = pd.DataFrame(robustness_rows)
    rob_csv = outdir / "robustness" / "robustness_summary.csv"
    rob_json = outdir / "robustness" / "robustness_summary.json"
    rob_df.to_csv(rob_csv, index=False)
    rob_json.write_text(rob_df.to_json(orient="records", indent=2))
    LOG.info("Wrote robustness summary -> %s and %s", rob_csv, rob_json)

    LOG.info("FE diagnostics & robustness complete. Files saved to: %s", Path(outdir).resolve())
    print("Done — see", Path(outdir).resolve())

if __name__ == "__main__":
    main()
