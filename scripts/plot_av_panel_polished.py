#!/usr/bin/env python3
"""
scripts/plot_av_panel_polished.py

Polished Added-Variable (partial regression) panel with z-scored residuals,
OLS fit + 95% CI ribbon, and LOWESS trend.

This is a drop-in corrected version that:
 - Never fails when 'controls' is empty (falls back to intercept-only residualization).
 - Always writes both the combined panel and individual plots.
 - Keeps the original visual style and layout from your working script.
 - Saves PNG / PDF / SVG outputs.

Usage:
    python scripts/plot_av_panel_polished.py --vars trade_exposure gov_index_zmean inflation_consumer_prices_pct
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})

OUTDIR = Path("reports/figs")
IND_DIR = OUTDIR / "individuals"
OUTDIR.mkdir(parents=True, exist_ok=True)
IND_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def residualize(y, X):
    """
    Return residuals of regressing y on X.
    If X is empty or None, return (y - mean(y), OLS_result_on_constant).
    """
    if X is None or len(X.columns) == 0:
        # intercept-only model: residuals are demeaned y
        mean_y = float(y.dropna().mean()) if not y.dropna().empty else 0.0
        resid = y - mean_y
        # build a trivial results-like object with needed attrs
        class _ResDummy:
            def __init__(self, resid, nobs):
                self.resid = resid
                self.params = np.array([mean_y, 0.0])  # const, slope placeholder
                self.pvalues = np.array([1.0, 1.0])
                self.nobs = nobs
        return resid, _ResDummy(resid, int(resid.dropna().shape[0]))
    # add constant and fit OLS with missing='drop' to handle any NA
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, Xc, missing="drop").fit()
    return model.resid, model


def fit_ols_line_with_ci(x, y, alpha=0.05):
    """OLS y ~ x with CI ribbon; works with numpy arrays or pandas Series."""
    X = sm.add_constant(x)
    res = sm.OLS(y, X, missing="drop").fit()

    # grid for prediction
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    Xpred = sm.add_constant(xx)

    pred = res.get_prediction(Xpred)
    mean = pred.predicted_mean
    ci_low, ci_high = pred.conf_int(alpha=alpha).T
    return xx, mean, ci_low, ci_high, res


def zscore(series):
    s = series.dropna()
    if s.empty:
        return series - series  # all NaN series but preserve index
    mu = s.mean()
    sd = s.std(ddof=0)
    return (series - mu) / (sd if sd != 0 else 1.0)


def make_pretty_name(col):
    mapping = {
        "trade_exposure": "Trade exposure (exports+imports / GDP)",
        "gov_index_zmean": "Governance index (z-mean)",
        "inflation_consumer_prices_pct": "Consumer inflation (pct)",
    }
    return mapping.get(col, col.replace("_", " "))


def save_fig_formats(fig, out_prefix: Path, dpi=300):
    for ext in ("png", "pdf", "svg"):
        outp = out_prefix.with_suffix("." + ext)
        fig.savefig(outp, dpi=dpi, bbox_inches="tight")
        print("Wrote", outp)


# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------

def plot_individual(df, pred, target_col="gdp_growth_pct", controls=None,
                    lowess_frac=0.5, save_prefix=IND_DIR / "av_{var}"):
    """
    Create an individual added-variable plot (z-scored residuals), OLS + CI, LOWESS.
    Saves to reports/figs/individuals/av_<var>.(png|pdf|svg)
    """
    sub = df[[target_col, pred] + (controls or [])].dropna(subset=[target_col, pred])
    if sub.empty:
        print("No data for", pred, "-> skipping individual plot")
        return False

    # residualize (controls may be empty list)
    ctrl_df = sub[controls] if controls else pd.DataFrame(index=sub.index)
    y_res, y_mod = residualize(sub[target_col], ctrl_df)
    x_res, x_mod = residualize(sub[pred], ctrl_df)

    y_z = zscore(y_res)
    x_z = zscore(x_res)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_z, y_z, s=36, edgecolor="k", linewidth=0.25, alpha=0.9)

    # OLS line + CI
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
    pretty = make_pretty_name(pred)
    ax.set_ylabel(pretty, fontsize=12)

    # stats box
    if ols_res is not None:
        try:
            slope = float(ols_res.params[1])
            pval = float(ols_res.pvalues[1])
            nobs = int(ols_res.nobs)
            txt = f"N={nobs}  slope={slope:.3f}  p={pval:.3g}"
        except Exception:
            txt = ""
    else:
        txt = ""
    if txt:
        ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha="right",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    plt.tight_layout()
    outp = Path(str(save_prefix).format(var=pred))
    save_fig_formats(fig, outp)
    plt.close(fig)
    print("Wrote individual:", outp.with_suffix(".png"))
    return True


def plot_panel(df, predictors, target_col="gdp_growth_pct",
               lowess_frac=0.5, figsize=(10, 10),
               save_prefix=OUTDIR / "av_panel_polished"):
    """
    Combined panel: one row per predictor sharing x-axis; same look as individual plots.
    Always attempts to produce output even when controls are missing.
    """
    # Determine controls: prefer numeric columns not in exclude
    exclude = set(predictors + [target_col, "iso3", "country", "year", "time"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    controls = [c for c in numeric_cols if c not in exclude]

    # If controls empty, fall back to intercept-only (empty DataFrame)
    if not controls:
        controls_df = None
        print("Warning: no controls found. Falling back to intercept-only residualization for panel.")
    else:
        controls_df = controls

    n = len(predictors)
    fig_h = max(3, n * 3.2)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    plt.suptitle("Added-variable (partial regression) panel", fontsize=18, y=0.96)
    palette = {"OLS": "#2c7bb6", "LOWESS": "#fdae61"}

    # collect all x-values for symmetric scaling
    all_xvals = []

    for ax, pred in zip(axes, predictors):
        sub = df[[target_col, pred] + (controls_df or [])].dropna(subset=[target_col, pred])
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for {pred}", ha="center", va="center")
            continue

        ctrl_df = sub[controls_df] if controls_df else pd.DataFrame(index=sub.index)
        y_res, y_mod = residualize(sub[target_col], ctrl_df)
        x_res, x_mod = residualize(sub[pred], ctrl_df)

        y_z = zscore(y_res)
        x_z = zscore(x_res)

        all_xvals.append(x_z.values)

        ax.scatter(x_z, y_z, s=28, edgecolor="k", linewidth=0.2, alpha=0.9)

        # OLS line + CI
        try:
            xx, mean, ci_low, ci_high, ols_res = fit_ols_line_with_ci(x_z.values, y_z.values)
            ax.plot(xx, mean, lw=2.0, color=palette["OLS"], label="OLS fit")
            ax.fill_between(xx, ci_low, ci_high, color=palette["OLS"], alpha=0.18)
        except Exception:
            ols_res = None

        # LOWESS
        try:
            lw_out = lowess(y_z.values, x_z.values, frac=lowess_frac, return_sorted=True)
            ax.plot(lw_out[:, 0], lw_out[:, 1], linestyle="--", color=palette["LOWESS"], lw=2, label="LOWESS")
        except Exception:
            pass

        ax.axvline(0.0, color="0.6", linestyle="--", linewidth=1.0)
        pretty = make_pretty_name(pred)
        # Align label to the right and rotate 0 to mimic polished panel
        ax.set_ylabel(pretty, fontsize=12, labelpad=12, rotation=0, ha="right")
        ax.grid(True, linewidth=0.5, alpha=0.6)

        # stats box
        if ols_res is not None:
            try:
                slope = float(ols_res.params[1])
                pval = float(ols_res.pvalues[1])
                nobs = int(ols_res.nobs)
                txt = f"N={nobs}  slope={slope:.3f}  p={pval:.3g}"
            except Exception:
                txt = ""
        else:
            txt = ""
        if txt:
            ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha="right",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    # global x-axis symmetric scaling
    if all_xvals:
        all_concat = np.concatenate(all_xvals)
        max_abs = np.nanmax(np.abs(all_concat)) if all_concat.size else 1.0
        lim = max(1.0, float(max_abs) * 1.05)
        for ax in axes:
            ax.set_xlim(-lim, lim)

    axes[-1].set_xlabel("Residual (z-score)", fontsize=13)

    # legend: combine handles from first axis
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.98, 0.55), fontsize=11, frameon=False)

    plt.subplots_adjust(left=0.14, right=0.88, top=0.93, bottom=0.12, hspace=0.45)

    foot = ("Note: Residuals z-scored (mean=0, sd=1). "
            "OLS line shown with 95% CI ribbon; LOWESS highlights nonlinearities.")
    fig.text(0.12, 0.02, foot, fontsize=10)

    # save combined panel
    save_fig_formats(fig, Path(save_prefix))
    plt.close(fig)
    print("Wrote panel:", Path(save_prefix).with_suffix(".png"))
    return True


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vars", nargs="+", required=True)
    p.add_argument("--features", default="data/processed/features_lean_imputed.csv")
    p.add_argument("--target", default="gdp_growth_pct")
    p.add_argument("--lowess-frac", type=float, default=0.5)
    p.add_argument("--outprefix", default=str(OUTDIR / "av_panel_polished"))
    args = p.parse_args()

    fpath = Path(args.features)
    if not fpath.exists():
        raise FileNotFoundError(f"Features file not found: {fpath}")

    df = pd.read_csv(fpath)

    # convert numerics safely: coerce non-numeric -> NaN (keeps behavior stable)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            # if conversion fails, leave column as-is (non-numeric)
            pass

    # create individual plots first (they always help for inspection)
    # Determine numeric controls used for individual plotting: same logic as in panel
    exclude = set(args.vars + [args.target, "iso3", "country", "year", "time"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    controls_cols = [c for c in numeric_cols if c not in exclude]
    # if controls_cols empty, pass None so residualize uses intercept-only
    controls_for_plots = controls_cols if controls_cols else None

    for v in args.vars:
        try:
            plot_individual(df, v, target_col=args.target, controls=controls_for_plots,
                            lowess_frac=args.lowess_frac,
                            save_prefix=IND_DIR / f"av_{v}")
        except Exception as e:
            print("Individual plot failed for", v, ":", e)

    # now create combined panel (this function will fall back gracefully)
    try:
        plot_panel(df, args.vars, target_col=args.target, lowess_frac=args.lowess_frac,
                   save_prefix=Path(args.outprefix))
    except Exception as e:
        print("Panel plotting failed:", e)


if __name__ == "__main__":
    main()
