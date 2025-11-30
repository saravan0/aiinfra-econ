#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

# --- logging
LOG = logging.getLogger("plot_rolling_validation")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# --- rcParams tuned for journal-quality figures
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # embed TrueType fonts in PDF
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Default figure colors
MODEL_COLOR = "#0072B2"
BENCH_COLOR = "#009E73"
CI_COLOR = "#0072B2"
MEAN_LINE = "#D55E00"
SHOCK_COLOR = "#E69F00"
SHOCK_BAND = "#F7EAD9"


def sha256_of_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def git_rev() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf8").strip()
    except Exception:
        return None


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args():
    p = argparse.ArgumentParser(
        description="Rolling/Expanding-window validation (year-based) for forecasting competence"
    )
    p.add_argument(
        "--features-csv",
        default="data/processed/features_lean_imputed.csv",
        help="CSV with columns: country, iso3, year, target and features",
    )
    p.add_argument(
        "--features",
        nargs="+",
        default=["gov_index_zmean", "trade_exposure", "inflation_consumer_prices_pct"],
    )
    p.add_argument("--target", default="gdp_growth_pct")
    p.add_argument(
        "--model",
        default="artifacts/en_model.joblib",
        help="saved sklearn pipeline to clone and refit",
    )
    p.add_argument("--outdir", default="reports/plot_rolling_validation")
    p.add_argument(
        "--window-type",
        choices=["expanding", "rolling"],
        default="expanding",
        help="expanding: train = all years <= t-1; rolling: train = last --window-size years",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="training window size (years) for rolling mode",
    )
    p.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1],
        help="list of test horizons to evaluate (e.g. 1 3)",
    )
    p.add_argument(
        "--min-train-years",
        type=int,
        default=5,
        help="minimum distinct years required to train",
    )
    p.add_argument(
        "--group-by",
        default="iso3",
        help="grouping column (for clustered bootstrap / persistence)",
    )
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument(
        "--bootstrap-rmse",
        action="store_true",
        help="compute clustered bootstrap CI for RMSE (slower)",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=500,
        help="number of bootstrap replicates for RMSE CI (if --bootstrap-rmse)",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="compute naive persistence benchmark and compare",
    )
    p.add_argument(
        "--dm-test",
        action="store_true",
        help="perform Diebold–Mariano test comparing model vs benchmark",
    )
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument(
        "--shock-years",
        nargs="+",
        type=int,
        default=[2009, 2020],
        help="shock years to mark on plots",
    )
    return p.parse_args()


# -------------------------
# Helper functions
# -------------------------
def build_splits_by_year(
    years_sorted: np.ndarray, window_type: str, window_size: int, horizon: int
):
    """
    Returns list of (train_years, test_year) pairs for given horizon
    """
    splits = []
    for idx, test_year in enumerate(years_sorted):
        test_idx = idx
        train_until_idx = test_idx - horizon
        if train_until_idx < 0:
            continue
        if window_type == "expanding":
            train_years = years_sorted[: train_until_idx + 1]
        else:  # rolling
            start_idx = max(0, train_until_idx - window_size + 1)
            train_years = years_sorted[start_idx : train_until_idx + 1]
        if len(train_years) >= 1:
            splits.append((list(train_years), int(test_year)))
    return splits


def persistence_forecast_one_year(
    df: pd.DataFrame, group_col: str, target_col: str, test_year: int
):
    """
    Returns persistence forecast array aligned to rows where df['year']==test_year.
    Forecast = previous-year target for same group (group_col). NaN if missing.
    """
    prev = df[df["year"] == (test_year - 1)][[group_col, target_col]].set_index(
        group_col
    )
    test_rows = df[df["year"] == test_year].copy()
    preds = []
    for idx in test_rows[group_col].values:
        try:
            preds.append(prev.loc[idx, target_col])
        except Exception:
            preds.append(np.nan)
    return np.array(preds, dtype=float)


def diebold_mariano_test(e_model, e_bench):
    """
    Simple Diebold-Mariano-like test using squared errors and Newey-West style variance (lag-1).
    Returns (DM_stat, pvalue)
    """
    d = (e_model**2) - (e_bench**2)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 3:
        return float("nan"), float("nan")
    mean_d = d.mean()
    # autocovariances
    gamma0 = np.mean((d - mean_d) ** 2)
    gamma1 = np.mean((d[1:] - mean_d) * (d[:-1] - mean_d)) if T > 1 else 0.0
    var_d = (gamma0 + 2 * gamma1) / T
    if var_d <= 0 or np.isnan(var_d):
        return float("nan"), float("nan")
    DM = mean_d / math.sqrt(var_d)
    pval = 2 * (1 - norm.cdf(abs(DM)))
    return float(DM), float(pval)


def _clustered_bootstrap_rmse(y_true, y_pred, cluster_ids, n_boot=500, random_state=0):
    """
    Clustered bootstrap for RMSE. cluster_ids aligned with y_true/y_pred.
    Returns (median, lower_2.5, upper_97.5) or (None, None, None) on failure.
    """
    try:
        rng = np.random.default_rng(random_state)
        clusters = np.array(cluster_ids)
        unique_clusters = np.unique(clusters[~pd.isna(clusters)])
        if unique_clusters.size == 0:
            # fallback: ordinary bootstrap over observations
            rmse_boot = []
            n = len(y_true)
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                mse = mean_squared_error(y_true[idx], y_pred[idx])
                rmse_boot.append(math.sqrt(mse))
            return (
                np.percentile(rmse_boot, 50),
                np.percentile(rmse_boot, 2.5),
                np.percentile(rmse_boot, 97.5),
            )
        cluster_to_idx = {c: np.where(clusters == c)[0] for c in unique_clusters}
        n_clusters = len(unique_clusters)
        rmse_boot = []
        for b in range(n_boot):
            chosen = rng.choice(unique_clusters, size=n_clusters, replace=True)
            idx = np.concatenate([cluster_to_idx[c] for c in chosen])
            if idx.size == 0:
                continue
            mse = mean_squared_error(y_true[idx], y_pred[idx])
            rmse_boot.append(math.sqrt(mse))
        if len(rmse_boot) == 0:
            return None, None, None
        return (
            float(np.percentile(rmse_boot, 50)),
            float(np.percentile(rmse_boot, 2.5)),
            float(np.percentile(rmse_boot, 97.5)),
        )
    except Exception as e:
        LOG.warning("Clustered bootstrap failed: %s", e)
        return None, None, None


# -------------------------
# Core validation routine
# -------------------------
def run_validation(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model_pipeline,
    out_base: Path,
    args,
) -> dict:
    """
    Runs validation across horizons and returns a dict of results per horizon:
    {h: {results_df, per_fold_preds_df, coef_drift_df, produced_files_list}}
    """
    results_all = {}
    features = list(features)
    # ensure year present
    if "year" not in df.columns:
        raise SystemExit("features file must contain 'year' column")
    years_sorted = np.array(sorted(df["year"].dropna().unique(), key=int))
    LOG.info("Unique years: %s", years_sorted.tolist())

    for horizon in args.horizons:
        LOG.info("Starting horizon=%d", horizon)
        splits = build_splits_by_year(
            years_sorted, args.window_type, args.window_size, horizon
        )
        LOG.info(
            "Found %d folds for horizon=%d (window=%s, size=%s)",
            len(splits),
            horizon,
            args.window_type,
            args.window_size,
        )

        records = []
        fold_preds = []
        coef_records = []

        for train_years, test_year in splits:
            train_mask = df["year"].isin(train_years)
            test_mask = df["year"] == test_year
            df_train = df.loc[train_mask, :].copy()
            df_test = df.loc[test_mask, :].copy()

            X_train = df_train[features].astype(float).values
            y_train = df_train[target].astype(float).values
            X_test = df_test[features].astype(float).values
            y_test = df_test[target].astype(float).values

            n_train = X_train.shape[0]
            n_test = X_test.shape[0]

            if (
                n_train < 10
                or len(set(train_years)) < args.min_train_years
                or n_test == 0
            ):
                LOG.info(
                    "Skipping test_year=%s (n_train=%d, years=%d, n_test=%d)",
                    test_year,
                    n_train,
                    len(set(train_years)),
                    n_test,
                )
                continue

            # clone and fit
            model = clone(model_pipeline)
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                LOG.warning("Model fit failed for test_year=%s: %s", test_year, e)
                continue

            # predict
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(math.sqrt(mse))

            # persistence benchmark if requested
            y_pred_bench = np.full_like(y_test, np.nan, dtype=float)
            if args.benchmark:
                bench = persistence_forecast_one_year(
                    df, args.group_by, target, int(test_year)
                )
                if bench.shape[0] == len(y_test):
                    y_pred_bench = bench
                else:
                    # fallback: fill with NaN aligned array
                    y_pred_bench = np.full(len(y_test), np.nan)

            # store predictions for this fold
            fold_df = pd.DataFrame(
                {
                    args.group_by: (
                        df_test[args.group_by].values
                        if args.group_by in df_test.columns
                        else [None] * len(y_test)
                    ),
                    "country": (
                        df_test["country"].values
                        if "country" in df_test.columns
                        else [None] * len(y_test)
                    ),
                    "year": df_test["year"].values,
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "y_pred_bench": y_pred_bench,
                }
            )
            fold_preds.append(fold_df)

            # record coefficients (if final estimator exposes coef_)
            try:
                final = model
                if hasattr(model, "named_steps"):
                    # sklearn Pipeline - take last step
                    final = list(model.named_steps.values())[-1]
                if hasattr(final, "coef_"):
                    coef_vals = np.array(final.coef_, dtype=float)
                    coef_records.append(
                        {
                            "test_year": int(test_year),
                            **{
                                f"coef_{f}": float(v)
                                for f, v in zip(features, coef_vals)
                            },
                        }
                    )
            except Exception:
                pass

            rec = {
                "test_year": int(test_year),
                "horizon": int(horizon),
                "train_years_start": int(min(train_years)),
                "train_years_end": int(max(train_years)),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "rmse": float(rmse),
                "mse": float(mse),
            }
            records.append(rec)
            LOG.info(
                "Fold h=%d test_year=%s | train=%s..%s | n_train=%d n_test=%d | RMSE=%.4f",
                horizon,
                test_year,
                min(train_years),
                max(train_years),
                n_train,
                n_test,
                rmse,
            )

        results_df = pd.DataFrame.from_records(records).sort_values(["test_year"])
        preds_all = (
            pd.concat(fold_preds, ignore_index=True) if fold_preds else pd.DataFrame()
        )
        coef_df = (
            pd.DataFrame.from_records(coef_records) if coef_records else pd.DataFrame()
        )

        results_all[horizon] = {
            "results_df": results_df,
            "preds_df": preds_all,
            "coef_df": coef_df,
        }

    return results_all


# -------------------------
# Plotting & I/O
# -------------------------
def plot_rmse_with_options(
    results_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    out_files_dir: Path,
    dpi: int,
    args,
    horizon: int,
):
    """
    Plot RMSE by test_year, optionally overlay persistence benchmark and CI from clustered bootstrap.
    Returns list of produced files (paths).
    """
    produced = []
    safe_mkdir(out_files_dir)
    if results_df.empty:
        LOG.warning("No results to plot for horizon=%s", horizon)
        return produced

    dfp = results_df.sort_values("test_year")
    years = dfp["test_year"].values
    model_rmse = dfp["rmse"].values

    # compute benchmark RMSE per year if preds_df has benchmark
    bench_rmse = []
    if args.benchmark and not preds_df.empty:
        for y in years:
            sub = preds_df[preds_df["year"] == y]
            if sub.shape[0] == 0 or sub["y_pred_bench"].isna().all():
                bench_rmse.append(np.nan)
            else:
                m = (
                    mean_squared_error(
                        sub["y_true"].values[~np.isnan(sub["y_pred_bench"].values)],
                        sub["y_pred_bench"].values[
                            ~np.isnan(sub["y_pred_bench"].values)
                        ],
                    )
                    if np.any(~np.isnan(sub["y_pred_bench"].values))
                    else np.nan
                )
                bench_rmse.append(math.sqrt(m) if not np.isnan(m) else np.nan)
        bench_rmse = np.array(bench_rmse)
    else:
        bench_rmse = None

    # clustered bootstrap CI per test year
    lower = np.full_like(model_rmse, np.nan, dtype=float)
    upper = np.full_like(model_rmse, np.nan, dtype=float)
    if args.bootstrap_rmse and not preds_df.empty:
        for i, y in enumerate(years):
            sub = preds_df[preds_df["year"] == y]
            if sub.shape[0] == 0:
                continue
            cluster_ids = (
                sub[args.group_by].values
                if args.group_by in sub.columns
                else np.arange(len(sub))
            )
            med, lo, hi = _clustered_bootstrap_rmse(
                sub["y_true"].values,
                sub["y_pred"].values,
                cluster_ids=cluster_ids,
                n_boot=args.n_boot,
                random_state=args.random_state,
            )
            lower[i] = lo if lo is not None else np.nan
            upper[i] = hi if hi is not None else np.nan

    # Prepare for plots
    shock_years = args.shock_years if hasattr(args, "shock_years") else [2009, 2020]
    shock_label_map = {2009: "Global financial crisis", 2020: "COVID-19 pandemic"}

    # errors
    errs = (
        (preds_df["y_true"].values - preds_df["y_pred"].values).astype(float)
        if not preds_df.empty
        else np.array([])
    )

    # rolling mean for middle panel (3-yr centered)
    try:
        roll_w = 3
        roll_series = (
            pd.Series(model_rmse, index=years)
            .rolling(window=roll_w, min_periods=1, center=True)
            .mean()
            .values
        )
    except Exception:
        roll_series = (
            pd.Series(model_rmse)
            .rolling(window=3, min_periods=1, center=True)
            .mean()
            .values
        )

    # ---------- Combined 3-panel figure (use constrained_layout to avoid tight_layout warnings) ----------
    fig = plt.figure(constrained_layout=True, figsize=(12.0, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.0, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Left: RMSE by year (with bench and CI + shock annotations)
    ax = ax0
    ax.plot(
        years,
        model_rmse,
        marker="o",
        lw=1.8,
        color=MODEL_COLOR,
        label="Model RMSE",
        zorder=3,
    )
    if bench_rmse is not None:
        ax.plot(
            years,
            bench_rmse,
            marker="s",
            lw=1.3,
            color=BENCH_COLOR,
            alpha=0.95,
            label="Persistence RMSE",
            zorder=2,
        )
    if args.bootstrap_rmse:
        ax.fill_between(
            years,
            lower,
            upper,
            color=CI_COLOR,
            alpha=0.18,
            label="95% CI (bootstrap)",
            zorder=1,
        )
    mean_val = float(np.nanmean(model_rmse))
    ax.axhline(
        mean_val,
        color=MEAN_LINE,
        lw=0.9,
        linestyle="--",
        label=f"Mean RMSE ({mean_val:.2f})",
        zorder=0,
    )

    # shock vertical lines and labels + subtle shading
    try:
        ymin = np.nanmin(model_rmse)
        ymax = np.nanmax(model_rmse)
    except Exception:
        ymin, ymax = 0.0, 1.0
    padding = max(0.6, 0.12 * (ymax - ymin)) if np.isfinite(ymax - ymin) else 0.6
    ax.set_ylim(ymin - padding, ymax + padding)
    for sy in shock_years:
        if sy >= years.min() and sy <= years.max():
            ax.axvline(
                sy, color=SHOCK_COLOR, linestyle="--", lw=1.1, alpha=0.95, zorder=0
            )
            ax.axvspan(sy - 0.25, sy + 0.25, color=SHOCK_BAND, alpha=0.95, zorder=0)
            label = shock_label_map.get(sy, f"Shock {sy}")
            ax.text(
                sy + 0.12,
                ax.get_ylim()[1] - 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                label,
                rotation=90,
                va="top",
                ha="left",
                fontsize=9,
                weight="semibold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            )

    ax.set_xlabel("Test year")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE by test year (h={horizon})", pad=8)
    ax.grid(alpha=0.14, linestyle=":")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

    # Middle: Rolling mean RMSE
    ax = ax1
    ax.plot(years, model_rmse, marker="o", lw=0.9, alpha=0.35, zorder=2)
    ax.plot(
        years,
        roll_series,
        marker="o",
        lw=2.2,
        color=MODEL_COLOR,
        zorder=3,
        markersize=5,
    )
    ax.set_title(f"{3}-yr rolling mean", pad=8)
    ax.set_xticks([])
    ax.grid(alpha=0.10, linestyle=":")

    # Right: Error distribution + summary box below
    ax = ax2
    if errs.size > 0:
        finite_errs = errs[np.isfinite(errs)]
        if finite_errs.size > 0:
            # Freedman–Diaconis for bin width with safe guards
            q75, q25 = (
                np.percentile(finite_errs, [75, 25])
                if finite_errs.size >= 2
                else (np.max(finite_errs), np.min(finite_errs))
            )
            iqr = max(q75 - q25, 1e-6)
            bin_width = (
                2 * iqr / (finite_errs.size ** (1 / 3)) if finite_errs.size > 1 else 1.0
            )
            nbins = int(
                max(
                    12,
                    min(
                        80,
                        (finite_errs.max() - finite_errs.min()) / max(bin_width, 1e-6),
                    ),
                )
            )
            nbins = max(12, min(nbins, 60))
            ax.hist(finite_errs, bins=nbins, alpha=0.95, edgecolor="none")
            ax.set_xlabel("Prediction error")
            ax.set_title("Error distribution", pad=8)
            ax.grid(alpha=0.08, linestyle=":")
        else:
            ax.text(0.5, 0.5, "No error data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, "No error data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    # summary text box placed beneath histogram (keeps figure uncluttered)
    mean_rmse = float(np.nanmean(model_rmse))
    median_rmse = float(np.nanmedian(model_rmse))
    latest_rmse = float(model_rmse[-1]) if len(model_rmse) > 0 else float("nan")
    n_folds = int(results_df.shape[0])
    boot_on = "on" if args.bootstrap_rmse else "off"
    bench_on = "yes" if (args.benchmark and bench_rmse is not None) else "no"
    summary_txt = (
        f"Mean RMSE: {mean_rmse:.3f}\n"
        f"Median RMSE: {median_rmse:.3f}\n"
        f"Latest RMSE: {latest_rmse:.3f}\n"
        f"Folds: {n_folds}\n\n"
        f"Shock years: {', '.join(map(str, args.shock_years))}\n\n"
        f"Bootstrap: {boot_on}\n"
        f"Bench: {bench_on}"
    )
    ax.text(
        0.5,
        -0.18,
        summary_txt,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="k", linewidth=0.6, alpha=0.95),
    )

    # save combined figure
    for ext in ("pdf", "svg", "png"):
        fp = out_files_dir / f"rmse_by_year_combined.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(fp, bbox_inches="tight")
        LOG.info("Wrote %s", fp)
        produced.append(str(fp.resolve()))
    plt.close(fig)

    # ---------- Single-panel RMSE per-horizon (kept for compatibility) ----------
    fig2, ax_single = plt.subplots(figsize=(9.0, 3.8), constrained_layout=True)
    ax_single.plot(
        years, model_rmse, marker="o", lw=1.6, color=MODEL_COLOR, label="Model RMSE"
    )
    if bench_rmse is not None:
        ax_single.plot(
            years,
            bench_rmse,
            marker="s",
            lw=1.2,
            color=BENCH_COLOR,
            alpha=0.9,
            label="Persistence RMSE",
        )
    if args.bootstrap_rmse:
        ax_single.fill_between(
            years, lower, upper, color=CI_COLOR, alpha=0.18, label="95% CI (bootstrap)"
        )
    ax_single.axhline(
        mean_val,
        color=MEAN_LINE,
        lw=0.9,
        linestyle="--",
        label=f"Mean RMSE ({mean_val:.2f})",
    )
    for sy in shock_years:
        if sy >= years.min() and sy <= years.max():
            ax_single.axvline(sy, color=SHOCK_COLOR, linestyle="--", lw=1.1, alpha=0.9)
            ax_single.text(
                sy + 0.12,
                ax_single.get_ylim()[1]
                - 0.06 * (ax_single.get_ylim()[1] - ax_single.get_ylim()[0]),
                shock_label_map.get(sy, f"Shock {sy}"),
                rotation=90,
                va="top",
                ha="left",
                fontsize=9,
            )
    ax_single.set_xlabel("Test year")
    ax_single.set_ylabel("RMSE")
    ax_single.set_title(f"RMSE by test year (h={horizon})")
    ax_single.grid(alpha=0.14, linestyle=":")
    ax_single.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False
    )

    for ext in ("pdf", "svg", "png"):
        fp2 = out_files_dir / f"rmse_by_year_h{horizon}.{ext}"
        if ext == "png":
            fig2.savefig(fp2, dpi=dpi, bbox_inches="tight")
        else:
            fig2.savefig(fp2, bbox_inches="tight")
        LOG.info("Wrote %s", fp2)
        produced.append(str(fp2.resolve()))
    plt.close(fig2)

    # write CSV table
    csv_fp = out_files_dir / f"rolling_validation_rmse_table_h{horizon}.csv"
    results_df.sort_values("test_year").to_csv(csv_fp, index=False)
    LOG.info("Wrote RMSE table -> %s", csv_fp)
    produced.append(str(csv_fp.resolve()))

    # save per-fold preds and coef drift if present
    if not preds_df.empty:
        preds_fp = out_files_dir / f"preds_by_fold_h{horizon}.csv"
        preds_df.to_csv(preds_fp, index=False)
        LOG.info("Wrote per-fold predictions -> %s", preds_fp)
        produced.append(str(preds_fp.resolve()))

    return produced


# -------------------------
# High-level orchestration
# -------------------------
def write_meta(meta_path: Path, produced_files: list, features_file: Path | None):
    meta = {
        "script": str(Path(__file__).resolve()),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "produced_files": produced_files,
        "features_file": str(features_file) if features_file else None,
        "features_sha256": (
            sha256_of_file(features_file)
            if features_file and features_file.exists()
            else None
        ),
        "git_commit": git_rev(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    LOG.info("Wrote meta -> %s", meta_path)


def main():
    args = parse_args()
    features_path = Path(args.features_csv)
    out_base = safe_mkdir(Path(args.outdir))
    out_files_dir = safe_mkdir(out_base / "files")

    LOG.info("Loading features: %s", features_path)
    df = pd.read_csv(features_path, low_memory=False)
    needed = [args.target] + args.features + ["year"]
    if args.group_by:
        needed.append(args.group_by)
    if "country" in df.columns:
        needed.append("country")
    missing = [c for c in needed if c not in df.columns]
    if missing:
        LOG.error("Missing columns in features CSV: %s", missing)
        raise SystemExit(1)

    # drop rows with NA in target/year/features (we will later check fold-specific NA)
    before = df.shape[0]
    df_sub = df.dropna(subset=[args.target, "year"] + args.features)
    LOG.info(
        "Dropped %d rows with missing target/year/features (rows: %d -> %d)",
        before - df_sub.shape[0],
        before,
        df_sub.shape[0],
    )
    if df_sub.shape[0] < 20:
        LOG.error(
            "Too few rows after dropping NA (%d). Need more data.", df_sub.shape[0]
        )
        raise SystemExit(1)

    # load model pipeline
    LOG.info("Loading model pipeline: %s", args.model)
    model_pipeline = joblib.load(args.model)

    # run validation
    results_all = run_validation(
        df_sub, args.features, args.target, model_pipeline, out_base, args
    )

    # collect produced files for manifest
    produced = []

    # for DM test aggregation
    dm_results = {}

    for horizon, info in results_all.items():
        results_df = info["results_df"]
        preds_df = info["preds_df"]
        coef_df = info["coef_df"]

        # optionally compute Diebold-Mariano across all folds for benchmark comparison
        if args.benchmark and args.dm_test and not preds_df.empty:
            valid = preds_df.dropna(subset=["y_true", "y_pred", "y_pred_bench"])
            if valid.shape[0] >= 3:
                e_model = valid["y_true"].values - valid["y_pred"].values
                e_bench = valid["y_true"].values - valid["y_pred_bench"].values
                DM, pval = diebold_mariano_test(e_model, e_bench)
                dm_results[horizon] = {"DM": DM, "pvalue": pval}
                LOG.info("Diebold-Mariano (h=%d): DM=%.4f p=%.4g", horizon, DM, pval)
            else:
                LOG.info(
                    "Diebold-Mariano not computed (insufficient aligned obs) for horizon=%d",
                    horizon,
                )

        # produce plots & files
        produced_h = plot_rmse_with_options(
            results_df,
            preds_df,
            out_files_dir,
            dpi=args.dpi,
            args=args,
            horizon=horizon,
        )
        produced.extend(produced_h)

        # save coefficient drift if available
        if not coef_df.empty:
            coef_fp = out_files_dir / f"coef_drift_h{horizon}.csv"
            coef_df.to_csv(coef_fp, index=False)
            produced.append(str(coef_fp.resolve()))
            LOG.info("Wrote coefficient drift -> %s", coef_fp)

    # write DM results if any
    if dm_results:
        dm_path = out_files_dir / "diebold_mariano_summary.json"
        dm_path.write_text(json.dumps(dm_results, indent=2), encoding="utf8")
        produced.append(str(dm_path.resolve()))
        LOG.info("Wrote Diebold-Mariano summary -> %s", dm_path)

    # meta & manifest
    meta_path = out_base / "meta.json"
    write_meta(meta_path, produced, features_file=features_path)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "produced_files": produced,
        "features_file": str(features_path),
        "features_sha256": sha256_of_file(features_path),
        "git_commit": git_rev(),
    }
    manifest_path = out_base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    LOG.info("Wrote manifest -> %s", manifest_path)

    LOG.info("Done. Outputs under %s", out_base.resolve())
    print("Wrote files to:", out_base)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
