"""
Generate added-variable (partial regression) diagnostic plots.

Produces:
 - reports/plot_added_variable_panel/plot_added_variable_panel.(png|pdf|svg)
 - reports/plot_added_variable_panel/individuals/av_<var>.(png|pdf|svg)
 - provenance metadata JSON describing inputs and generated files

Design notes:
 - Generates both per-variable and combined added-variable diagnostics.
 - Records provenance information for reproducibility.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import os
import subprocess
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

sns.set_style("whitegrid")

# ---------------------------
# Helpers
# ---------------------------
def sha256_file(p: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def git_commit_hash() -> Optional[str]:
    try:
        # try to get HEAD commit (silently fail)
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None

def zscore(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series - series  # preserve index but produce all-NaN
    mu = s.mean()
    sd = s.std(ddof=0)
    return (series - mu) / (sd if sd != 0 else 1.0)

def residualize(y: pd.Series, X: Optional[pd.DataFrame]):
    """
    Residuals of y on X. If X is None or empty -> demean y (intercept-only).
    Returns (resid_series, model_like_obj)
    model_like_obj: simple object with .resid, .params, .pvalues, .nobs when possible.
    """
    if X is None or (isinstance(X, (pd.DataFrame,)) and X.shape[1] == 0):
        mean_y = float(y.dropna().mean()) if not y.dropna().empty else 0.0
        resid = y - mean_y
        class Dummy:
            def __init__(self, resid, nobs):
                self.resid = resid
                self.params = np.array([mean_y, 0.0])
                self.pvalues = np.array([1.0, 1.0])
                self.nobs = nobs
        return resid, Dummy(resid, int(resid.dropna().shape[0]))
    Xc = sm.add_constant(X, has_constant="add")
    # statsmodels will drop NA rows automatically with missing='drop'
    model = sm.OLS(y, Xc, missing="drop").fit()
    return model.resid, model

def fit_ols_line_with_ci(x, y, alpha=0.05):
    X = sm.add_constant(x)
    res = sm.OLS(y, X, missing="drop").fit()
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    Xpred = sm.add_constant(xx)
    pred = res.get_prediction(Xpred)
    mean = pred.predicted_mean
    ci_low, ci_high = pred.conf_int(alpha=alpha).T
    return xx, mean, ci_low, ci_high, res

def save_fig_formats(fig, out_prefix: Path, dpi: int = 600):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    produced = []
    for ext in ("png", "pdf", "svg"):
        p = out_prefix.with_suffix("." + ext)   # <-- correct
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        produced.append(str(p))
    return produced


def make_pretty_name(col: str) -> str:
    mapping = {
        "trade_exposure": "Trade exposure (exports+imports / GDP)",
        "gov_index_zmean": "Governance index (z-mean)",
        "inflation_consumer_prices_pct": "Consumer inflation (pct)",
    }
    return mapping.get(col, col.replace("_", " "))

# ---------------------------
# Plotting primitives
# ---------------------------
def plot_individual(
    df: pd.DataFrame,
    pred: str,
    target_col: str,
    controls: Optional[List[str]],
    lowess_frac: float,
    outdir: Path,
    dpi: int,
) -> Optional[List[str]]:
    """
    Writes av_<pred>.(png|pdf|svg) into outdir/individuals/
    Returns list of produced filepaths or None if skipped.
    """
    indiv_dir = outdir / "individuals"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    sub_cols = [c for c in [target_col, pred] + (controls or []) if c in df.columns]
    if target_col not in sub_cols or pred not in sub_cols:
        print(f"Skipping {pred}: required cols missing in features")
        return None
    sub = df[sub_cols].dropna(subset=[target_col, pred]).copy()
    if sub.empty:
        print(f"No data for {pred} -> skipping individual")
        return None

    ctrl_df = sub[controls] if controls else None
    y_res, _ = residualize(sub[target_col], ctrl_df)
    x_res, _ = residualize(sub[pred], ctrl_df)

    y_z = zscore(y_res)
    x_z = zscore(x_res)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_z, y_z, s=36, edgecolor="k", linewidth=0.25, alpha=0.9)

    # OLS + CI
    ols_res = None
    try:
        xx, mean, ci_low, ci_high, ols_res = fit_ols_line_with_ci(x_z.values, y_z.values)
        ax.plot(xx, mean, lw=2.0, color="#2c7bb6", label="OLS fit")
        ax.fill_between(xx, ci_low, ci_high, color="#2c7bb6", alpha=0.18)
    except Exception:
        ols_res = None

    # LOWESS
    try:
        lw_out = lowess(y_z.values, x_z.values, frac=lowess_frac, return_sorted=True)
        ax.plot(lw_out[:, 0], lw_out[:, 1], linestyle="--", color="#fdae61", lw=2, label="LOWESS")
    except Exception:
        pass

    ax.axvline(0.0, color="0.6", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Residual (z-score)", fontsize=11)
    ax.set_ylabel(make_pretty_name(pred), fontsize=12)

    # stats box
    txt = ""
    if ols_res is not None:
        try:
            slope = float(ols_res.params[1])
            pval = float(ols_res.pvalues[1])
            nobs = int(ols_res.nobs)
            txt = f"N={nobs}  slope={slope:.3f}  p={pval:.3g}"
        except Exception:
            txt = ""
    if txt:
        ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha="right",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    plt.tight_layout()
    out_pref = indiv_dir / f"av_{pred}"
    files = save_fig_formats(fig, out_pref, dpi=dpi)
    plt.close(fig)
    print("Wrote individual plot for", pred, "->", files[0])
    return files

def plot_panel(
    df: pd.DataFrame,
    predictors: List[str],
    target_col: str,
    lowess_frac: float,
    outdir: Path,
    dpi: int,
    figsize=(10, 10)
) -> List[str]:
    """Create combined vertical panel and save to outdir/files/"""
    files_dir = outdir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    # pick numeric controls automatically (exclude predictors & meta)
    exclude = set(predictors + [target_col, "iso3", "country", "year", "time"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    controls = [c for c in numeric_cols if c not in exclude]
    if not controls:
        controls_df_cols = None
        print("Warning: no numeric controls found; using intercept-only residualization.")
    else:
        controls_df_cols = controls

    n = len(predictors)
    fig_h = max(3, n * 3.2)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    plt.suptitle("Added-variable (partial regression) panel", fontsize=18, y=0.96)
    palette = {"OLS": "#2c7bb6", "LOWESS": "#fdae61"}
    all_xvals = []

    for ax, pred in zip(axes, predictors):
        if pred not in df.columns:
            ax.text(0.5, 0.5, f"No column '{pred}' in features", ha="center", va="center")
            continue
        sub_cols = [c for c in [target_col, pred] + (controls_df_cols or []) if c in df.columns]
        sub = df[sub_cols].dropna(subset=[target_col, pred]).copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for {pred}", ha="center", va="center")
            continue

        ctrl_df = sub[controls_df_cols] if controls_df_cols else None
        y_res, _ = residualize(sub[target_col], ctrl_df)
        x_res, _ = residualize(sub[pred], ctrl_df)

        y_z = zscore(y_res)
        x_z = zscore(x_res)
        all_xvals.append(x_z.values)

        ax.scatter(x_z, y_z, s=28, edgecolor="k", linewidth=0.2, alpha=0.9)

        try:
            xx, mean, ci_low, ci_high, ols_res = fit_ols_line_with_ci(x_z.values, y_z.values)
            ax.plot(xx, mean, lw=2.0, color=palette["OLS"], label="OLS fit")
            ax.fill_between(xx, ci_low, ci_high, color=palette["OLS"], alpha=0.18)
        except Exception:
            ols_res = None

        try:
            lw_out = lowess(y_z.values, x_z.values, frac=lowess_frac, return_sorted=True)
            ax.plot(lw_out[:, 0], lw_out[:, 1], linestyle="--", color=palette["LOWESS"], lw=2, label="LOWESS")
        except Exception:
            pass

        ax.axvline(0.0, color="0.6", linestyle="--", linewidth=1.0)
        ax.set_ylabel(make_pretty_name(pred), fontsize=12, labelpad=12, rotation=0, ha="right")
        ax.grid(True, linewidth=0.5, alpha=0.6)

        txt = ""
        if ols_res is not None:
            try:
                slope = float(ols_res.params[1])
                pval = float(ols_res.pvalues[1])
                nobs = int(ols_res.nobs)
                txt = f"N={nobs}  slope={slope:.3f}  p={pval:.3g}"
            except Exception:
                txt = ""
        if txt:
            ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha="right",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    # symmetric x-limits across panels
    if all_xvals:
        all_concat = np.concatenate([a for a in all_xvals if a.size])
        if all_concat.size:
            max_abs = float(np.nanmax(np.abs(all_concat)))
            lim = max(1.0, max_abs * 1.05)
            for ax in axes:
                ax.set_xlim(-lim, lim)

    axes[-1].set_xlabel("Residual (z-score)", fontsize=13)

    # combined legend (from first axis)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.98, 0.55), fontsize=11, frameon=False)

    plt.subplots_adjust(left=0.14, right=0.88, top=0.93, bottom=0.12, hspace=0.45)
    foot = ("Note: Residuals z-scored (mean=0, sd=1). "
            "OLS line shown with 95% CI ribbon; LOWESS highlights nonlinearities.")
    fig.text(0.12, 0.02, foot, fontsize=10)

    out_pref = files_dir / "plot_added_variable_panel"
    produced = save_fig_formats(fig, out_pref, dpi=dpi)
    plt.close(fig)
    print("Wrote panel ->", produced[0])
    return produced

# ---------------------------
# CLI
# ---------------------------
def main():
    p = argparse.ArgumentParser(prog="plot_added_variable_panel")
    p.add_argument("--vars", nargs="+", required=True)
    p.add_argument("--features", default="data/processed/features_lean_imputed.csv")
    p.add_argument("--target", default="gdp_growth_pct")
    p.add_argument("--outdir", default="reports/plot_added_variable_panel")
    p.add_argument("--lowess-frac", type=float, default=0.3)
    p.add_argument("--dpi", type=int, default=600, help="DPI for saved figures (paper-grade default 600)")
    args = p.parse_args()

    feat_path = Path(args.features)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}")
    print("Loading features:", feat_path)
    df = pd.read_csv(feat_path, low_memory=False)

    # coerce numeric columns safely
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass

    # compute controls (numeric cols excluding predictors/target/ids)
    exclude = set(args.vars + [args.target, "iso3", "country", "year", "time"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    controls = [c for c in numeric_cols if c not in exclude]
    if not controls:
        controls = None

    # produce individual plots (best for inspection)
    produced_files = []
    for v in args.vars:
        try:
            files = plot_individual(df, v, args.target, controls, args.lowess_frac, outdir, dpi=args.dpi)
            if files:
                produced_files.extend(files)
        except Exception as e:
            print("Individual plot failed for", v, ":", e)

    # produce combined panel
    panel_files = []
    try:
        panel_files = plot_panel(df, args.vars, args.target, args.lowess_frac, outdir, dpi=args.dpi)
        produced_files.extend(panel_files)
    except Exception as e:
        print("Panel plotting failed:", e)

    # write metadata
    meta = {
        "script": str(Path(__file__).resolve()),
        "args": vars(args),
        "features_file": str(feat_path.resolve()),
        "features_sha256": sha256_file(feat_path),
        "git_commit": git_commit_hash(),
        "produced_files": produced_files,
    }
    meta_path = outdir / "plot_added_variable_panel_meta.json"
    with open(meta_path, "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=2)
    print("Wrote metadata ->", meta_path)
    print("Done. Wrote files:\n -", "\n - ".join(produced_files))

if __name__ == "__main__":
    main()
