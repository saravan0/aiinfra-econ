"""
Generate nonlinear diagnostics using LOWESS and GAM-based smoothing.

Produces:
 - results/plot_lowess_nonlinearity/<feature>/ (per-feature figures and summaries)
 - manifest JSON files recording provenance and diagnostic metadata

Design notes:
 - Estimates nonlinear response curves using local smoothing and spline-based models.
 - Supports uncertainty estimation via resampling-based procedures.
 - Records provenance information for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

# Try pygam
try:
    from pygam import LinearGAM, s

    HAS_PYGAM = True
except Exception:
    HAS_PYGAM = False

LOG = logging.getLogger("plot_lowess_nonlinearity")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# publication rc
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
    }
)

PALETTE = ["#0072B2", "#009E73", "#D55E00"]
DEFAULT_FEATURES = [
    "gov_index_zmean",
    "trade_exposure",
    "inflation_consumer_prices_pct",
]


def write_metadata(manifest_path: Path, produced: List[str], features_file: Path):
    """
    Write top-level manifest/metadata for the run.
    Kept simple and compatible with earlier script versions.
    """
    meta = {
        "script": str(Path(__file__).resolve()),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "produced_files": produced,
        "features_file": str(features_file),
        "features_sha256": (
            sha256_of_file(features_file)
            if features_file and features_file.exists()
            else None
        ),
        "git_commit": git_rev(),
    }
    manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    LOG.info("Wrote manifest -> %s", manifest_path)


def sha256_of_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def git_rev() -> Optional[str]:
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


def linreg_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return slope, intercept


def compute_lowess(
    x: np.ndarray, y: np.ndarray, frac: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:

    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size == 0:
        return np.array([]), np.array([])

    ux, inv = np.unique(x, return_inverse=True)
    if ux.size < x.size:

        ys_u = np.zeros_like(ux)
        counts = np.zeros_like(ux, dtype=int)
        for i, idx in enumerate(inv):
            ys_u[idx] += y[i]
            counts[idx] += 1
        ys_u = ys_u / counts
        x_agg = ux
        y_agg = ys_u
    else:
        x_agg = x
        y_agg = y

    if x_agg.size < 5:
        jitter = np.random.RandomState(0).normal(scale=1e-8, size=x_agg.shape)
        x_agg = x_agg + jitter

    try:
        res = lowess(endog=y_agg, exog=x_agg, frac=frac, return_sorted=True)
        xs = res[:, 0]
        ys = res[:, 1]
    except Exception as e:

        order = np.argsort(x_agg)
        xs = x_agg[order]

        window = max(1, int(np.ceil(len(xs) * frac)))
        ys = (
            pd.Series(y_agg[order])
            .rolling(window=window, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )

    return xs, ys


def fit_gam_or_spline(
    x: np.ndarray, y: np.ndarray, n_splines=20, lam=0.6, max_fit_n: Optional[int] = None
):
    if HAS_PYGAM:
        LOG.info("Fitting pygam LinearGAM (n_splines=%d, lam=%g)", n_splines, lam)
        X = x.reshape(-1, 1)
        # optionally subsample for faster gam fitting
        if max_fit_n is not None and len(X) > max_fit_n:
            idx = np.linspace(0, len(X) - 1, max_fit_n).astype(int)
            Xg, yg = X[idx], y[idx]
        else:
            Xg, yg = X, y
        gam = LinearGAM(s(0, n_splines=n_splines, lam=lam)).gridsearch(
            Xg, yg, progress=False
        )
        return ("pygam", gam)
    LOG.info("pygam unavailable — using UnivariateSpline fallback")
    s_factor = max(1e-8, len(x) * np.var(y) * 0.05)
    spline = UnivariateSpline(x, y, s=s_factor)
    return ("spline", spline)


def gam_predict_and_ci(
    model_tuple, x_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kind, model = model_tuple
    if kind == "pygam":
        XX = x_grid.reshape(-1, 1)
        y_pred = model.predict(XX)
        try:
            ci = model.prediction_intervals(XX, width=0.95)
            lower, upper = ci[:, 0], ci[:, 1]
        except Exception:
            lower = y_pred - 1.96 * np.std(y_pred)
            upper = y_pred + 1.96 * np.std(y_pred)
        return y_pred, lower, upper
    else:
        y_pred = model(x_grid)
        # crude CI approximation
        sigma = np.std(y_pred - model(x_grid)) if len(x_grid) else 0.0
        lower = y_pred - 1.96 * sigma
        upper = y_pred + 1.96 * sigma
        return y_pred, lower, upper


def gam_derivative(model_tuple, x_grid: np.ndarray) -> np.ndarray:
    kind, model = model_tuple
    if kind == "pygam":
        try:
            grad = model.gradient(x_grid.reshape(-1, 1))
            if grad.ndim == 2:
                return grad[:, 0]
            return grad
        except Exception:
            y = model.predict(x_grid.reshape(-1, 1))
            return np.gradient(y, x_grid)
    else:
        return model.derivative()(x_grid)


def binned_stats(x: np.ndarray, y: np.ndarray, n_bins: int = 20):
    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) <= 1:
        return np.array([]), np.array([]), np.array([])
    idx = np.digitize(x, bins) - 1
    means_x, means_y, se_y = [], [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() == 0:
            continue
        means_x.append(np.mean(x[sel]))
        means_y.append(np.mean(y[sel]))
        se_y.append(np.std(y[sel]) / np.sqrt(sel.sum()))
    return np.array(means_x), np.array(means_y), np.array(se_y)


def detect_turning_points(x_grid: np.ndarray, deriv: np.ndarray) -> List[Dict]:
    sign = np.sign(deriv)
    zeros = np.where(np.diff(sign) != 0)[0]
    pts = []
    for z in zeros:
        x0, x1 = x_grid[z], x_grid[z + 1]
        y0, y1 = deriv[z], deriv[z + 1]
        x_cross = (x0 + x1) / 2 if (y1 - y0) == 0 else x0 - y0 * (x1 - x0) / (y1 - y0)
        pts.append({"x": float(x_cross), "index": int(z)})
    return pts


# --- Bootstrap LOWESS with optional clustering
def bootstrap_lowess_ci(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    frac: float,
    n_boot: int = 500,
    random_state: int = 0,
    cluster_ids: Optional[np.ndarray] = None,
    cache_path: Optional[Path] = None,
    cluster_boot_n: Optional[int] = None,
):
    """
    Robust LOWESS bootstrap:
    - returns (median, lower, upper, turn_locations) aligned with x_grid
    - respects cache_path if provided
    - is resilient to degenerate bootstrap replicates (duplicate x, too few uniques)
    """
    # load cache if present
    if cache_path and cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            LOG.info("Loaded cached bootstrap from %s", cache_path)
            return data["median"], data["lower"], data["upper"], data.get("turns", None)
        except Exception:
            LOG.warning("Failed to load bootstrap cache, recomputing")

    rng = np.random.default_rng(random_state)
    n = len(x)
    boot_preds = np.full((n_boot, len(x_grid)), np.nan)
    turn_locations: List[float] = []

    def safe_lowess_interp(xb, yb):
        """Compute LOWESS on (xb,yb) robustly, return interp on x_grid (may be all-nan)."""
        try:
            xs_bs, ys_bs = compute_lowess(xb, yb, frac=frac)
            if xs_bs.size == 0:
                return np.full(len(x_grid), np.nan), xs_bs, ys_bs
            # ensure xs_bs is strictly increasing for np.interp
            # if not, aggregate duplicates (shouldn't happen because compute_lowess aggregates), else jitter
            if not np.all(np.diff(xs_bs) > 0):
                # tiny jitter to restore strict monotonicity
                xs_bs = xs_bs + np.linspace(0, 1e-12, xs_bs.size)
            interp_vals = np.interp(x_grid, xs_bs, ys_bs, left=np.nan, right=np.nan)
            return interp_vals, xs_bs, ys_bs
        except Exception:
            return np.full(len(x_grid), np.nan), np.array([]), np.array([])

    if cluster_ids is not None:
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
        for b in range(n_boot):
            chosen = rng.choice(unique_clusters, size=n_clusters, replace=True)
            idx = np.concatenate([cluster_to_idx[c] for c in chosen])
            xb, yb = x[idx], y[idx]
            interp_vals, xs_bs, ys_bs = safe_lowess_interp(xb, yb)
            boot_preds[b, :] = interp_vals

            # turning points: only if xs_bs valid and strictly increasing
            try:
                if xs_bs.size >= 3 and np.all(np.diff(xs_bs) > 0):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        deriv_bs = np.gradient(ys_bs, xs_bs)
                    deriv_grid = np.interp(
                        x_grid, xs_bs, deriv_bs, left=np.nan, right=np.nan
                    )
                    zeros = np.where(np.diff(np.sign(deriv_grid)) != 0)[0]
                    if zeros.size:
                        z = zeros[0]
                        x_cross = 0.5 * (x_grid[z] + x_grid[z + 1])
                        turn_locations.append(float(x_cross))
            except Exception:
                # ignore degenerate replicate
                pass
    else:
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            xb, yb = x[idx], y[idx]
            interp_vals, xs_bs, ys_bs = safe_lowess_interp(xb, yb)
            boot_preds[b, :] = interp_vals

            try:
                if xs_bs.size >= 3 and np.all(np.diff(xs_bs) > 0):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        deriv_bs = np.gradient(ys_bs, xs_bs)
                    deriv_grid = np.interp(
                        x_grid, xs_bs, deriv_bs, left=np.nan, right=np.nan
                    )
                    zeros = np.where(np.diff(np.sign(deriv_grid)) != 0)[0]
                    if zeros.size:
                        z = zeros[0]
                        x_cross = 0.5 * (x_grid[z] + x_grid[z + 1])
                        turn_locations.append(float(x_cross))
            except Exception:
                pass

    # percentiles (allow nans; nanpercentile will ignore by default only if all-nan then raises - guard)
    try:
        median = np.nanpercentile(boot_preds, 50, axis=0)
        lower = np.nanpercentile(boot_preds, 2.5, axis=0)
        upper = np.nanpercentile(boot_preds, 97.5, axis=0)
    except Exception:
        # if something odd happened (e.g., all rows nan), return arrays of nan
        median = np.full(len(x_grid), np.nan)
        lower = np.full(len(x_grid), np.nan)
        upper = np.full(len(x_grid), np.nan)

    # cache results
    if cache_path:
        try:
            np.savez_compressed(
                cache_path,
                median=median,
                lower=lower,
                upper=upper,
                turns=np.array(turn_locations),
            )
            LOG.info("Saved bootstrap cache -> %s", cache_path)
        except Exception:
            LOG.warning("Failed to write bootstrap cache to %s", cache_path)

    return median, lower, upper, turn_locations


def plot_feature_elite(
    df: pd.DataFrame,
    feature: str,
    target: str,
    out_dir: Path,
    frac: float,
    dpi: int,
    n_bins: int,
    bootstrap: int,
    n_boot: int,
    cluster_by: Optional[str],
    cluster_boot: Optional[int],
    frac_grid: Optional[List[float]],
    max_fit_n: Optional[int],
):
    """
    Updated plotting function — improved reviewer-level styling.
    Drop-in replacement for the previous implementation.
    """
    LOG.info("Plotting feature %s (nrows=%d)", feature, df.shape[0])
    sub = (
        df[[feature, target, cluster_by]]
        if cluster_by and cluster_by in df.columns
        else df[[feature, target]]
    )
    sub = sub.dropna()
    n = sub.shape[0]
    if n < 20:
        LOG.warning("Too few rows (%d) for %s; skipping", n, feature)
        return []

    x = sub[feature].astype(float).values
    y = sub[target].astype(float).values
    clusters = (
        sub[cluster_by].values if (cluster_by and cluster_by in df.columns) else None
    )

    # output grid for prediction / derivative
    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 400)

    # compute lowess + gam + derivative (same logic)
    lowess_x, lowess_y = compute_lowess(x, y, frac=frac)

    envelope = {}
    if frac_grid:
        for f in frac_grid:
            xs_f, ys_f = compute_lowess(x, y, frac=f)
            envelope[f] = {"xs": xs_f.tolist(), "ys": ys_f.tolist()}

    model_tuple = fit_gam_or_spline(x, y, n_splines=20, lam=0.6, max_fit_n=max_fit_n)
    gam_y, gam_lower, gam_upper = gam_predict_and_ci(model_tuple, x_grid)
    deriv = gam_derivative(model_tuple, x_grid)
    try:
        deriv_smooth = UnivariateSpline(
            x_grid, deriv, s=len(x_grid) * np.var(deriv) * 0.01
        )(x_grid)
    except Exception:
        deriv_smooth = deriv

    bx, by, bse = binned_stats(x, y, n_bins=n_bins)
    slope, intercept = linreg_line(x, y)
    lin_y = slope * x_grid + intercept
    tps = detect_turning_points(x_grid, deriv)

    # bootstrap call (unchanged) — expects bootstrap_lowess_ci to exist
    boot_median = boot_lo = boot_hi = None
    boot_turns = []
    if bootstrap:
        cache_dir = out_dir / "cache"
        safe_mkdir(cache_dir)
        cache_file = cache_dir / f"lowess_boot_{feature}_frac{frac}_n{n_boot}.npz"
        LOG.info(
            "Computing bootstrap LOWESS (n_boot=%d, cluster_by=%s)...",
            n_boot,
            cluster_by,
        )
        start = time.time()
        cluster_ids = clusters if (cluster_by and cluster_by in df.columns) else None
        boot_median, boot_lo, boot_hi, boot_turns = bootstrap_lowess_ci(
            x,
            y,
            x_grid=x_grid,
            frac=frac,
            n_boot=n_boot,
            random_state=0,
            cluster_ids=cluster_ids,
            cache_path=cache_file,
        )
        if boot_turns is None:
            boot_turns = []
        else:
            try:
                boot_turns = [
                    float(v) for v in np.asarray(boot_turns).ravel() if np.isfinite(v)
                ]
            except Exception:
                boot_turns = list(boot_turns) if hasattr(boot_turns, "__iter__") else []
        LOG.info(
            "Bootstrap done in %.1f s — found %d turning-point samples",
            time.time() - start,
            len(boot_turns),
        )

    # -------------------------
    # Styling tweaks (elite)
    # -------------------------
    # refined color palette (print-friendly + colorblind safe)
    COLOR_LOWESS_MED = "#004488"  # darker blue
    COLOR_LOWESS = "#5DA5D1"  # lighter blue
    COLOR_GAM = "#228833"  # green
    COLOR_OLS = "#6E6E6E"  # grey/black
    DERIV_COLOR = "#D55E00"  # orange (same as before but bolder)

    # scatter jitter for near-integer/discrete features (very small)
    def _apply_small_jitter_if_discrete(arr: np.ndarray):
        # If >40% of points are near-integers, apply tiny jitter
        frac_near_int = np.mean(np.isclose(arr, np.round(arr), atol=1e-6))
        if frac_near_int > 0.4:
            jitter = np.random.RandomState(0).normal(
                scale=(np.ptp(arr) * 1e-4 + 1e-8), size=arr.shape
            )
            return arr + jitter
        return arr

    x_plot = _apply_small_jitter_if_discrete(x)

    # Combined figure: improved layout and legend outside
    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 3.6), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]}
    )
    ax0, ax1, ax2 = axes

    # Panel A: scatter + LOWESS (boot median) + GAM + OLS + CI + frac envelope
    ax0.scatter(x_plot, y, s=6, alpha=0.15, linewidths=0, rasterized=True, zorder=1)
    if bootstrap and boot_median is not None:
        ax0.fill_between(
            x_grid,
            boot_lo,
            boot_hi,
            color=COLOR_LOWESS,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )
        ax0.plot(
            x_grid,
            boot_median,
            lw=1.6,
            color=COLOR_LOWESS_MED,
            label="LOWESS median (boot)",
            zorder=5,
        )
    if len(lowess_x) > 0:
        ax0.plot(
            lowess_x,
            lowess_y,
            lw=1.8,
            color=COLOR_LOWESS,
            label=f"LOWESS (frac={frac})",
            zorder=6,
        )
    ax0.plot(x_grid, gam_y, lw=2.0, color=COLOR_GAM, label="GAM/spline", zorder=7)
    ax0.fill_between(
        x_grid, gam_lower, gam_upper, color=COLOR_GAM, alpha=0.12, linewidth=0, zorder=3
    )
    ax0.plot(
        x_grid,
        lin_y,
        lw=0.9,
        linestyle="--",
        color=COLOR_OLS,
        label="Linear OLS",
        zorder=4,
    )

    # frac envelope overlay (light)
    if frac_grid:
        for f, data in envelope.items():
            ax0.plot(data["xs"], data["ys"], lw=0.8, alpha=0.45, zorder=2)

    ax0.set_xlabel(feature.replace("_", " "))
    ax0.set_ylabel(target)
    ax0.grid(alpha=0.12, linestyle=":")
    # legend outside plot (right)
    ax0.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)

    # annotate turning points (GAM-derived)
    ytop = ax0.get_ylim()[1]
    ypos = ytop * 0.92  # place TP text at 92% of top
    for i, tp in enumerate(tps):
        try:
            txt = f"TP: {tp['x']:.3g}"
            ax0.axvline(
                tp["x"], color="#D55E00", linestyle=":", lw=1.0, alpha=0.85, zorder=3
            )
            # small triangular marker at the top of the axis (slightly below top to avoid clipping)
            ax0.plot(tp["x"], ypos, marker="v", markersize=4, color="#D55E00", zorder=8)
            ax0.text(
                tp["x"],
                ypos * (0.98 - 0.02 * i),
                txt,
                va="top",
                ha="center",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
        except Exception:
            pass

    # Panel B: binned scatter + errorbars + LOWESS overlay (clean styling)
    if len(bx) > 0:
        ax1.errorbar(
            bx,
            by,
            yerr=bse,
            fmt="o",
            ms=5,
            capsize=3,
            markeredgecolor="k",
            alpha=0.95,
            zorder=6,
        )
    if len(lowess_x) > 0:
        ax1.plot(lowess_x, lowess_y, lw=1.6, color=COLOR_LOWESS_MED, zorder=5)
    ax1.set_xlabel(feature.replace("_", " "))
    ax1.set_title("Binned means (quantile bins)", fontsize=9)
    ax1.grid(alpha=0.12, linestyle=":")

    # Panel C: derivative plot — thinner baseline and smaller annotation
    ax2.plot(x_grid, deriv_smooth, lw=2.0, color=DERIV_COLOR, zorder=5)
    ax2.axhline(0, color="#222222", lw=0.6, linestyle="--", alpha=0.6)
    idx_max = int(np.nanargmax(np.abs(deriv_smooth)))
    # smaller marker + text
    ax2.scatter(
        [x_grid[idx_max]], [deriv_smooth[idx_max]], s=18, color=DERIV_COLOR, zorder=6
    )
    ax2.text(
        x_grid[idx_max],
        deriv_smooth[idx_max],
        f" slope={deriv_smooth[idx_max]:.3g}",
        fontsize=7,
        va="bottom",
    )
    if len(boot_turns) > 0:
        ax2.scatter(
            boot_turns,
            np.full(len(boot_turns), ax2.get_ylim()[0] * 0.98),
            marker="|",
            color="#333333",
            s=8,
            alpha=0.6,
            zorder=3,
        )
    ax2.set_xlabel(feature.replace("_", " "))
    ax2.set_title("Marginal effect (derivative)", fontsize=9)
    ax2.grid(alpha=0.12, linestyle=":")

    plt.tight_layout(rect=(0, 0, 0.94, 1.0))  # leave space for legend on right

    out_files = []
    # save combined figure
    for ext in ("pdf", "svg", "png"):
        fp = out_dir / f"lowess_gam_{feature}.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(fp, bbox_inches="tight")
        LOG.info("Wrote %s", fp)
        out_files.append(str(fp.resolve()))
    plt.close(fig)

    # --- separate binned scatter figure (same style)
    if len(bx) > 0:
        fig_b, axb = plt.subplots(figsize=(5.5, 3.6))
        axb.errorbar(
            bx, by, yerr=bse, fmt="o", ms=6, capsize=3, markeredgecolor="k", alpha=0.95
        )
        if len(lowess_x) > 0:
            axb.plot(lowess_x, lowess_y, lw=1.6, color=COLOR_LOWESS_MED)
        axb.set_xlabel(feature.replace("_", " "))
        axb.set_ylabel(target)
        axb.grid(alpha=0.12, linestyle=":")
        plt.tight_layout()
        for ext in ("pdf", "svg", "png"):
            fpb = out_dir / f"lowess_binned_{feature}.{ext}"
            if ext == "png":
                fig_b.savefig(fpb, dpi=dpi, bbox_inches="tight")
            else:
                fig_b.savefig(fpb, bbox_inches="tight")
            LOG.info("Wrote %s", fpb)
            out_files.append(str(fpb.resolve()))
        plt.close(fig_b)

    # --- separate derivative figure (same style)
    fig_d, axd = plt.subplots(figsize=(5.5, 3.6))
    axd.plot(x_grid, deriv_smooth, lw=2.0, color=DERIV_COLOR)
    axd.axhline(0, color="#222222", lw=0.6, linestyle="--", alpha=0.6)
    axd.scatter([x_grid[idx_max]], [deriv_smooth[idx_max]], s=18, color=DERIV_COLOR)
    if len(boot_turns) > 0:
        axd.scatter(
            boot_turns,
            np.full(len(boot_turns), axd.get_ylim()[0] * 0.98),
            marker="|",
            color="#333333",
            s=8,
            alpha=0.6,
        )
    axd.set_xlabel(feature.replace("_", " "))
    axd.set_ylabel("dY/dX")
    axd.grid(alpha=0.12, linestyle=":")
    plt.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fpd = out_dir / f"lowess_derivative_{feature}.{ext}"
        if ext == "png":
            fig_d.savefig(fpd, dpi=dpi, bbox_inches="tight")
        else:
            fig_d.savefig(fpd, bbox_inches="tight")
        LOG.info("Wrote %s", fpd)
        out_files.append(str(fpd.resolve()))
    plt.close(fig_d)

    # metadata JSON (unchanged)
    meta = {
        "feature": feature,
        "n_points": int(n),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "frac": float(frac),
        "turning_points_gam": tps,
        "bootstrap": (
            {
                "performed": bool(bootstrap),
                "n_boot": int(n_boot) if bootstrap else 0,
                "cluster_by": cluster_by,
                "turns_sample": boot_turns[:200] if boot_turns else [],
            }
            if bootstrap
            else None
        ),
    }
    meta_path = out_dir / f"lowess_gam_{feature}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    LOG.info("Wrote %s", meta_path)
    out_files.append(str(meta_path.resolve()))

    return out_files


def parse_args():
    p = argparse.ArgumentParser(
        description="Elite LOWESS diagnostics with bootstrap + cluster + frac-grid"
    )
    p.add_argument("--features-csv", default="data/processed/features_lean_imputed.csv")
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--target", default="gdp_growth_pct")
    p.add_argument("--frac", type=float, default=0.3)
    p.add_argument(
        "--frac-grid",
        nargs="+",
        type=float,
        default=None,
        help="additional LOWESS fracs to draw as sensitivity envelope (space-separated)",
    )
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--outdir", default="reports/plot_lowess_nonlinearity")
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument(
        "--bootstrap", action="store_true", help="compute LOWESS bootstrap CIs"
    )
    p.add_argument(
        "--n-boot", type=int, default=500, help="number of bootstrap replicates"
    )
    p.add_argument(
        "--cluster-by",
        type=str,
        default=None,
        help="column name to use for block-bootstrap (e.g., iso3)",
    )
    p.add_argument(
        "--cluster-boot",
        type=int,
        default=None,
        help="cluster bootstrap replicates (overrides --n-boot for clusters)",
    )
    p.add_argument(
        "--max-fit-n",
        type=int,
        default=None,
        help="subsample size for GAM fitting (if dataset is huge)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    LOG.info("Starting plot_lowess_nonlinearity; pygam available: %s", HAS_PYGAM)
    features_path = Path(args.features_csv)
    df = pd.read_csv(features_path, low_memory=False)

    missing = [c for c in ([args.target] + args.features) if c not in df.columns]
    if missing:
        LOG.error("Missing cols: %s", missing)
        raise SystemExit(1)

    base = safe_mkdir(Path(args.outdir))
    produced = []

    frac_grid = args.frac_grid if args.frac_grid else None
    if frac_grid:
        frac_grid = [float(f) for f in frac_grid]

    for feature in args.features:
        out_feat = safe_mkdir(base / feature)
        files = plot_feature_elite(
            df=df,
            feature=feature,
            target=args.target,
            out_dir=out_feat,
            frac=args.frac,
            dpi=args.dpi,
            n_bins=args.n_bins,
            bootstrap=args.bootstrap,
            n_boot=(args.cluster_boot if args.cluster_boot else args.n_boot),
            cluster_by=args.cluster_by,
            cluster_boot=args.cluster_boot,
            frac_grid=frac_grid,
            max_fit_n=args.max_fit_n,
        )
        produced.extend(files)

    manifest_path = base / "manifest.json"
    write_metadata(manifest_path, produced, features_path)
    LOG.info("Wrote manifest %s", manifest_path)
    LOG.info("Done. Outputs under %s", base.resolve())


if __name__ == "__main__":
    main()
