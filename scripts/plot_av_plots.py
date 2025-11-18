#!/usr/bin/env python3
"""
scripts/plot_av_plots.py

Produce publication-quality Partial Regression / Added-Variable (AV) plots
for selected predictors.

Usage:
    python scripts/plot_av_plots.py --vars trade_exposure gov_index_zmean inflation_consumer_prices_pct
    python scripts/plot_av_plots.py --vars trade_exposure --features data/processed/features_lean_imputed.csv --entity_col iso3

What it does:
 - Loads features CSV (default: data/processed/features_lean_imputed.csv)
 - For each variable `v`:
     1. Builds "other" predictors = numeric columns except [target, v, entity, time]
     2. Optionally demeans by entity_col (within-country FE) before regression if --demean-fe
     3. Regresses target on other predictors -> residuals r_y
     4. Regresses predictor v on other predictors -> residuals r_x
     5. Scatter plot r_x vs r_y, OLS fit-line, lowess (smoothed) curve, annotate slope/p/padj/R2/n
 - Saves per-variable plots and a combined panel png/pdf/svg.

Requirements:
 - pandas, numpy, statsmodels, matplotlib, seaborn, scipy
 - Already part of your environment.

Notes on interpretation:
 - AV plot visualizes the marginal relationship between predictor and outcome
   after "partialing out" (controlling for) other covariates. For FE-style,
   the partialing happens in within-country-demeaned space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 12})

ROOT = Path(".")
DEFAULT_FEATURES = ROOT / "data" / "processed" / "features_lean_imputed.csv"
OUTDIR = ROOT / "reports" / "figs"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


def _select_other_predictors(df: pd.DataFrame, target: str, var: str, entity_col: Optional[str], time_col: Optional[str], max_cols: int = 200) -> List[str]:
    # numeric columns only, exclude target, var, entity_col, time_col
    exclude = {target, var}
    if entity_col:
        exclude.add(entity_col)
    if time_col:
        exclude.add(time_col)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    others = [c for c in numeric if c not in exclude]
    # if there are too many predictors, limit to a sensible number (keeps AV regressions stable)
    if len(others) > max_cols:
        # choose by variance (largest variance first) as heuristic for informative controls
        variances = df[others].var(numeric_only=True).sort_values(ascending=False)
        others = variances.index.tolist()[:max_cols]
    return others


def demean_within(df: pd.DataFrame, group_col: str, cols: List[str]) -> pd.DataFrame:
    # within-group demean: x - group_mean
    gmean = df.groupby(group_col)[cols].transform("mean")
    return df[cols] - gmean


def fit_residuals(y: pd.Series, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[sm.regression.linear_model.RegressionResultsWrapper]]:
    """
    Fit OLS y ~ X (with constant) and return residuals and fit object.
    If X is empty (no cols), residuals are y - mean(y).
    """
    if X is None or X.shape[1] == 0:
        resid = y - y.mean()
        return resid.to_numpy(), None
    Xc = sm.add_constant(X, has_constant="add")
    # coerce numeric
    Xc = Xc.apply(pd.to_numeric, errors="coerce")
    mask = Xc.notna().all(axis=1) & y.notna()
    if mask.sum() == 0:
        resid = pd.Series(np.full(len(y), np.nan), index=y.index)
        return resid.to_numpy(), None
    try:
        model = sm.OLS(y.loc[mask], Xc.loc[mask]).fit()
        resid = pd.Series(index=y.index, dtype=float)
        resid.loc[mask] = model.resid
        resid.loc[~mask] = np.nan
        return resid.to_numpy(), model
    except Exception:
        # fallback: use linear regression via numpy least squares
        Xm = Xc.fillna(0).to_numpy(dtype=float)
        ym = y.fillna(y.mean()).to_numpy(dtype=float)
        try:
            coef, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
            pred = Xm.dot(coef)
            resid = ym - pred
            return resid, None
        except Exception:
            resid = y - y.mean()
            return resid.to_numpy(), None


def compute_av_stats(rx: np.ndarray, ry: np.ndarray) -> dict:
    # remove nan pairs
    mask = np.isfinite(rx) & np.isfinite(ry)
    rxv = rx[mask]
    ryv = ry[mask]
    n = rxv.size
    if n == 0:
        return {"n": 0}
    # simple OLS ry ~ rx
    X = sm.add_constant(rxv, has_constant="add")
    res = sm.OLS(ryv, X).fit()
    slope = float(res.params[1]) if res.params.size > 1 else float(np.nan)
    pvalue = float(res.pvalues[1]) if res.pvalues.size > 1 else float(np.nan)
    r2 = float(res.rsquared)
    # robust stderr for slope
    se = float(res.bse[1]) if res.bse.size > 1 else float(np.nan)
    return {"n": n, "slope": slope, "pvalue": pvalue, "r2": r2, "se": se}


def plot_av_single(rx: np.ndarray, ry: np.ndarray, var: str, stats: dict, out_prefix: Path, figsize=(8, 6), dpi=300):
    """
    rx: residuals of predictor after regressing out other covariates
    ry: residuals of target after regressing out other covariates
    stats: dictionary with slope, pvalue, r2, n
    """
    # filter finite
    mask = np.isfinite(rx) & np.isfinite(ry)
    rxv = rx[mask]
    ryv = ry[mask]
    n = int(stats.get("n", np.sum(mask)))

    fig, ax = plt.subplots(figsize=figsize)
    # scatter
    # jitter is not desirable; plot with alpha
    ax.scatter(rxv, ryv, s=24, alpha=0.7)

    # OLS fit line
    if rxv.size > 1:
        X = sm.add_constant(rxv, has_constant="add")
        res = sm.OLS(ryv, X).fit()
        xs = np.linspace(rxv.min(), rxv.max(), 200)
        preds = res.predict(sm.add_constant(xs))
        ax.plot(xs, preds, color="#1f77b4", lw=2, label="OLS fit")

        # 95% CI for prediction of mean relation (approx)
        try:
            pred_se = res.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
            lower = pred_se["mean_ci_lower"].astype(float)
            upper = pred_se["mean_ci_upper"].astype(float)
            ax.fill_between(xs, lower, upper, color="#1f77b4", alpha=0.12)
        except Exception:
            pass

    # LOWESS smooth
    if rxv.size >= 5:
        try:
            lo = lowess(ryv, rxv, frac=0.25, return_sorted=True)
            ax.plot(lo[:, 0], lo[:, 1], color="#ff7f0e", lw=1.8, linestyle="--", label="LOWESS")
        except Exception:
            pass

    # Vertical line at 0
    ax.axvline(0, color="0.6", linestyle="--", zorder=0)

    # Labels & annotation
    ax.set_xlabel("Predictor (partialed) — residualized")
    ax.set_ylabel("Target (partialed) — residualized")
    ax.set_title(f"Added-variable (Partial regression) plot — {var}", fontsize=14)

    slope = stats.get("slope", None)
    pvalue = stats.get("pvalue", None)
    r2 = stats.get("r2", None)
    se = stats.get("se", None)

    ann = f"n = {n}\n"
    if slope is not None:
        ann += f"slope = {slope:.4f} "
        if se is not None:
            ann += f"(se={se:.3f})\n"
        else:
            ann += "\n"
    if pvalue is not None:
        ann += f"p = {pvalue:.2e}\n"
    if r2 is not None:
        ann += f"R² = {r2:.3f}"

    ax.text(0.98, 0.02, ann, transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.8"))

    ax.legend(frameon=False)
    plt.tight_layout()

    for ext in ("png", "pdf", "svg"):
        p = out_prefix.with_suffix("." + ext)
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def make_panel(out_prefix: Path, var_stats: List[dict], figsize_per_var=(8, 3), dpi=300):
    # var_stats: list of {var, rx, ry, stats}
    n = len(var_stats)
    if n == 0:
        return None
    fig_h = max(3, n * figsize_per_var[1])
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(figsize_per_var[0], fig_h), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, vs in zip(axes, var_stats):
        rx = vs["rx"]
        ry = vs["ry"]
        var = vs["var"]
        stats = vs["stats"]
        mask = np.isfinite(rx) & np.isfinite(ry)
        rxv = rx[mask]; ryv = ry[mask]
        ax.scatter(rxv, ryv, s=18, alpha=0.75)
        # OLS line
        if rxv.size > 1:
            X = sm.add_constant(rxv, has_constant="add")
            res = sm.OLS(ryv, X).fit()
            xs = np.linspace(rxv.min(), rxv.max(), 150)
            preds = res.predict(sm.add_constant(xs))
            ax.plot(xs, preds, color="#1f77b4", lw=2)
        # LOWESS
        if rxv.size >= 5:
            lo = lowess(ryv, rxv, frac=0.25, return_sorted=True)
            ax.plot(lo[:, 0], lo[:, 1], color="#ff7f0e", lw=1.4, linestyle="--")
        ax.axvline(0, color="0.6", linestyle="--")
        ax.set_ylabel(var, fontsize=11, rotation=0, labelpad=120, va="center")
    axes[-1].set_xlabel("Residualized predictor (after controls)")
    plt.suptitle("Added-variable (partial regression) panel", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf", "svg"):
        p = out_prefix.with_suffix("." + ext)
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_prefix.with_suffix(".png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vars", nargs="+", required=True, help="Variables to produce AV plots for (predictor names)")
    p.add_argument("--features", default=str(DEFAULT_FEATURES), help="Path to features CSV")
    p.add_argument("--target", default="gdp_growth_pct", help="Name of target column")
    p.add_argument("--entity_col", default="iso3", help="Entity / country column used for FE demeaning (optional)")
    p.add_argument("--time_col", default=None, help="Optional time column to exclude from predictors")
    p.add_argument("--demean_fe", action="store_true", help="If set, produce FE (within) AV plots using entity demeaning")
    p.add_argument("--outdir", default=str(OUTDIR), help="Output directory for figures")
    p.add_argument("--figsize", type=float, default=8.0, help="Base figure width (inches) for single plots")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    args = p.parse_args()

    features_path = Path(args.features)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(features_path)
    target = args.target
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found in features file.")

    var_stats = []

    for var in args.vars:
        print(f"[INFO] Processing variable: {var}")
        if var not in df.columns:
            print(f"[WARN] Variable '{var}' not found in data; skipping.")
            continue

        # choose other predictors
        other_preds = _select_other_predictors(df, target, var, args.entity_col, args.time_col, max_cols=200)
        print(f"       using {len(other_preds)} control predictors (examples): {other_preds[:6]}")

        # optionally demean within groups for FE
        if args.demean_fe and args.entity_col and args.entity_col in df.columns:
            # Build demeaned copies for target, var, and other_preds
            cols_to_demean = [target, var] + other_preds
            # ensure present
            cols_to_demean = [c for c in cols_to_demean if c in df.columns]
            dmat = df[cols_to_demean].copy()
            demeaned = demean_within(dmat, args.entity_col, cols_to_demean)
            # create a temporary dataframe for regressions (same index)
            reg_df = demeaned
            y = reg_df[target]
            x_var = reg_df[var]
            X_controls = reg_df[[c for c in other_preds if c in reg_df.columns]]
        else:
            # no FE demean
            y = df[target]
            x_var = df[var]
            X_controls = df[[c for c in other_preds if c in df.columns]]

        # compute residuals
        ry, res_y = fit_residuals(y, X_controls)
        rx, res_x = fit_residuals(x_var, X_controls)

        stats = compute_av_stats(rx, ry)
        print(f"       n={stats.get('n',0)} slope={stats.get('slope')} p={stats.get('pvalue')} R2={stats.get('r2')}")

        # save single plot
        out_prefix = outdir / f"av_{var}"
        plot_av_single(rx, ry, var, stats, out_prefix, figsize=(args.figsize, args.figsize * 0.75), dpi=args.dpi)

        # collect for panel
        var_stats.append({"var": var, "rx": rx, "ry": ry, "stats": stats})

    # combined panel
    if var_stats:
        panel_prefix = outdir / "av_panel"
        make_panel(panel_prefix, var_stats, figsize_per_var=(args.figsize, 3.0), dpi=args.dpi)
        print(f"[DONE] AV plots written to {outdir.resolve()}")
    else:
        print("[WARN] No variables processed; no panel created.")


if __name__ == "__main__":
    main()
