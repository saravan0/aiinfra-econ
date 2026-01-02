"""
Generate SHAP-based interpretability diagnostics for ElasticNet models.

Produces:
 - reports/plot_shap_elasticnet/files/shap_summary.(pdf|svg|png)
 - reports/plot_shap_elasticnet/files/shap_dependence_<feature>.(pdf|svg|png)
 - reports/plot_shap_elasticnet/files/shap_feature_importance.csv
 - reports/plot_shap_elasticnet/meta.json
 - reports/plot_shap_elasticnet/manifest.json

Design notes:
 - Computes feature attribution values using linear SHAP methods.
 - Generates summary, dependence, and importance diagnostics.
 - Records provenance information for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# logging
LOG = logging.getLogger("plot_shap_elasticnet")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# figure rc (journal-quality)
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["#0072B2", "#009E73", "#D55E00", "#000000"], linestyle=["-", "-", "-", "--"]
)

# color palette
BAR_COLOR = "#004488"
POINT_COLOR = "#D55E00"
LOWESS_COLOR = "#0072B2"

HIST_CAP_PCT = {
    "gov_index_zmean": None,
    "trade_exposure": 99.0,
    "inflation_consumer_prices_pct": 99.5,
}


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


def extract_coeffs_from_model(model) -> Tuple[np.ndarray, float, Optional[List[str]]]:
    """
    Get coef_, intercept_, feature_names (optional) from a fitted ElasticNet or pipeline.
    Returns (coef_array, intercept, feature_names_or_None)
    """
    # sklearn Pipeline case
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            # try to extract final estimator
            if hasattr(model, "named_steps") and "elasticnet" in model.named_steps:
                est = model.named_steps.get("elasticnet") or model.named_steps.get(
                    "elasticnetcv"
                )
            else:
                # try last step
                est = model.steps[-1][1]
        else:
            est = model
        coef = getattr(est, "coef_", None)
        intercept = float(getattr(est, "intercept_", 0.0))
        return np.asarray(coef, dtype=float), intercept, None
    except Exception:
        # fallback: try direct attributes
        coef = getattr(model, "coef_", None)
        intercept = float(getattr(model, "intercept_", 0.0))
        return np.asarray(coef, dtype=float), intercept, None


def plot_summary_bar(mean_abs_shap: pd.Series, out_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    mean_abs_shap.sort_values(ascending=True).plot(kind="barh", ax=ax, color=BAR_COLOR)
    ax.set_xlabel("Mean |SHAP| (scaled features)")
    for i, (name, val) in enumerate(mean_abs_shap.sort_values(ascending=True).items()):
        ax.text(
            val + (0.01 * max(0.001, mean_abs_shap.max())),
            i,
            f"{val:.3f}",
            va="center",
            fontsize=8,
            color="#222222",
        )
    ax.set_title("SHAP feature importance — ElasticNet (scaled X)")
    ax.grid(axis="x", alpha=0.12, linestyle=":")
    plt.tight_layout()
    out_files = []
    for ext in ("pdf", "svg", "png"):
        fp = out_dir / f"shap_summary.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(fp, bbox_inches="tight")
        LOG.info("Wrote %s", fp)
        out_files.append(str(fp.resolve()))
    plt.close(fig)
    return out_files


def plot_dependence(
    feature_vals: np.ndarray,
    shap_vals: np.ndarray,
    feature_name: str,
    out_dir: Path,
    dpi: int,
    add_lowess: bool = True,
):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    # scatter SHAP vs raw feature
    ax.scatter(
        feature_vals, shap_vals, s=8, alpha=0.45, rasterized=True, color=POINT_COLOR
    )
    ax.set_xlabel(feature_name.replace("_", " "))
    ax.set_ylabel("SHAP value (contribution to predicted gdp_growth_pct)")
    ax.set_title(f"SHAP dependence — {feature_name}")
    ax.grid(alpha=0.12, linestyle=":")
    # optional LOWESS overlay for readability (not used for inference)
    if add_lowess:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            xs, ys = lowess(shap_vals, feature_vals, frac=0.3, return_sorted=True).T
            ax.plot(xs, ys, lw=1.8, color=LOWESS_COLOR)
        except Exception:
            pass
    plt.tight_layout()
    out_files = []
    for ext in ("pdf", "svg", "png"):
        fp = out_dir / f"shap_dependence_{feature_name}.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(fp, bbox_inches="tight")
        LOG.info("Wrote %s", fp)
        out_files.append(str(fp.resolve()))
    plt.close(fig)
    return out_files


def plot_shap_3panel(
    feature_raw: np.ndarray,
    feature_scaled: np.ndarray,
    shap_vals: np.ndarray,
    coef_val: float,
    feature_name: str,
    out_dir: Path,
    dpi: int,
    add_lowess: bool = True,
) -> List[str]:

    from scipy.stats import gaussian_kde
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

    # style / colors (reuse constants)
    BAR_COLOR_LOCAL = BAR_COLOR
    POINT_COLOR_LOCAL = POINT_COLOR
    LOWESS_COLOR_LOCAL = LOWESS_COLOR
    MARGINAL_COLOR = "#228833"

    # ensure out_dir exists
    safe_mkdir(out_dir)

    # panel layout
    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 3.6), gridspec_kw={"width_ratios": [1.2, 0.9, 1.0]}
    )
    ax0, ax1, ax2 = axes

    # Panel A: SHAP vs raw feature
    ax0.scatter(
        feature_raw,
        shap_vals,
        s=8,
        alpha=0.45,
        rasterized=True,
        color=POINT_COLOR_LOCAL,
        zorder=2,
    )
    ax0.set_xlabel(feature_name.replace("_", " "))
    ax0.set_ylabel("SHAP value (contribution)")
    ax0.set_title(f"SHAP dependence — {feature_name}")
    ax0.grid(alpha=0.12, linestyle=":")
    if add_lowess:
        try:
            xs_lo, ys_lo = _lowess(
                shap_vals, feature_raw, frac=0.3, return_sorted=True
            ).T
            ax0.plot(xs_lo, ys_lo, lw=1.6, color=LOWESS_COLOR_LOCAL, zorder=4)
        except Exception:
            pass

    # add a small summary annotation: mean |SHAP|
    mean_abs = float(np.mean(np.abs(shap_vals)))
    ax0.text(
        0.98,
        0.02,
        f"mean|SHAP| = {mean_abs:.3g}",
        transform=ax0.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Panel B: histogram / density of raw feature (with rug)
    ax1.hist(
        feature_raw,
        bins=30,
        density=False,
        alpha=0.9,
        edgecolor="#333333",
        color="#CCCCCC",
    )
    # density overlay
    try:
        kde = gaussian_kde(feature_raw)
        x_kde = np.linspace(np.nanmin(feature_raw), np.nanmax(feature_raw), 200)
        binwidth = (np.nanmax(feature_raw) - np.nanmin(feature_raw)) / 30.0
        kde_vals = kde(x_kde) * len(feature_raw) * binwidth
        ax1.plot(x_kde, kde_vals, lw=1.2, color=LOWESS_COLOR_LOCAL)
    except Exception:
        pass

    cap_pct = HIST_CAP_PCT.get(feature_name, None)
    if cap_pct is not None:
        try:
            cap_val = float(np.nanpercentile(feature_raw, cap_pct))
            ax1.set_xlim(left=np.nanmin(feature_raw), right=cap_val)
            ax1.axvline(
                cap_val, color="#777777", linestyle=":", linewidth=0.8, alpha=0.6
            )
        except Exception:
            pass

    # Panel C: marginal effect (coef * scaled feature)
    marg_vals = coef_val * feature_scaled
    # plot marginal as a line across sorted x (use sorted scaled for smooth line)
    order = np.argsort(feature_scaled)
    xs_sorted = feature_scaled[order]
    marg_sorted = marg_vals[order]
    ax2.plot(xs_sorted, marg_sorted, lw=1.8, color=MARGINAL_COLOR)
    ax2.axhline(0, color="#222222", lw=0.6, linestyle="--", alpha=0.6)
    try:
        rug_y = ax2.get_ylim()[0] + 0.02 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])
        ax2.scatter(
            feature_scaled,
            np.full_like(feature_scaled, rug_y),
            marker="|",
            s=12,
            color="#444444",
            alpha=0.35,
            linewidths=0.8,
            zorder=3,
        )
    except Exception:
        pass
    ax2.set_xlabel(f"{feature_name} (scaled)")
    ax2.set_title("Marginal effect (coef × scaled X)")
    ax2.grid(alpha=0.12, linestyle=":")

    # small annotation of coef (value) on panel C
    ax2.text(
        0.98,
        0.02,
        f"coef = {coef_val:.4g}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.tight_layout(rect=(0, 0, 0.94, 1.0))

    out_files = []
    # Save combined figure (vector-first + high-dpi png)
    for ext in ("pdf", "svg", "png"):
        fp = out_dir / f"shap_3panel_{feature_name}.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(fp, bbox_inches="tight")
        LOG.info("Wrote %s", fp)
        out_files.append(str(fp.resolve()))
    plt.close(fig)

    # metadata for this feature
    meta = {
        "feature": feature_name,
        "n_points": int(len(feature_raw)),
        "coef": float(coef_val),
        "mean_abs_shap": mean_abs,
    }
    meta["hist_display_cap_pct"] = cap_pct
    meta["hist_display_cap_val"] = (
        float(cap_val) if (cap_pct is not None and "cap_val" in locals()) else None
    )

    meta_path = out_dir / f"shap_3panel_{feature_name}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    LOG.info("Wrote %s", meta_path)
    out_files.append(str(meta_path.resolve()))

    return out_files


def _update_feature_meta_with_singleplots(meta_path: Path, single_info: dict):
    """
    Update existing per-feature JSON (shap_3panel_<feature>.json) by adding keys for single-panel files.
    single_info is a dict like {"dependence": fname, "distribution": fname, "marginal": fname, "notes": "..."}
    """
    try:
        meta = json.loads(meta_path.read_text(encoding="utf8"))
    except Exception:
        meta = {}
    meta.setdefault("single_panel_plots", {}).update(single_info)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    LOG.info("Updated meta -> %s", meta_path)


def plot_shap_dependence_single(
    feature_raw: np.ndarray,
    shap_vals: np.ndarray,
    feature_name: str,
    out_dir: Path,
    dpi: int,
    add_lowess: bool = True,
) -> str:
    """
    Standalone dependence plot (SHAP vs raw feature). Returns written filename.
    """
    safe_mkdir(out_dir)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.scatter(
        feature_raw, shap_vals, s=10, alpha=0.45, rasterized=True, color=POINT_COLOR
    )
    ax.set_xlabel(feature_name.replace("_", " "))
    ax.set_ylabel("SHAP value (contribution)")
    ax.set_title(f"SHAP dependence — {feature_name}")
    ax.grid(alpha=0.12, linestyle=":")
    if add_lowess:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            xs, ys = lowess(shap_vals, feature_raw, frac=0.3, return_sorted=True).T
            ax.plot(xs, ys, lw=1.6, color=LOWESS_COLOR)
        except Exception:
            pass
    # small summary annotation
    ax.text(
        0.98,
        0.02,
        f"mean|SHAP| = {np.mean(np.abs(shap_vals)):.3g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    out_fp = out_dir / f"dependence_{feature_name}.svg"
    fig.savefig(out_fp, bbox_inches="tight")
    fig.savefig(str(out_fp.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_fp.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote dependence single -> %s", out_fp)
    return str(out_fp.resolve())


def plot_shap_distribution_single(
    feature_raw: np.ndarray,
    feature_name: str,
    out_dir: Path,
    dpi: int,
    cap_pct: Optional[float] = None,
) -> str:
    """
    Standalone distribution plot (histogram + KDE). cap_pct is optional display cap percentile (e.g. 99.5).
    """
    from scipy.stats import gaussian_kde

    safe_mkdir(out_dir)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(feature_raw, bins=30, alpha=0.95, edgecolor="#333333", color="#CCCCCC")
    try:
        kde = gaussian_kde(feature_raw)
        x_kde = np.linspace(np.nanmin(feature_raw), np.nanmax(feature_raw), 200)
        binwidth = (np.nanmax(feature_raw) - np.nanmin(feature_raw)) / 30.0
        kde_vals = kde(x_kde) * len(feature_raw) * binwidth
        ax.plot(x_kde, kde_vals, lw=1.2, color=LOWESS_COLOR)
    except Exception:
        pass

    cap_val = None
    if cap_pct is not None:
        try:
            cap_val = float(np.nanpercentile(feature_raw, cap_pct))
            ax.set_xlim(left=np.nanmin(feature_raw), right=cap_val)
            ax.axvline(
                cap_val, color="#777777", linestyle=":", linewidth=0.8, alpha=0.6
            )
        except Exception:
            cap_val = None

    ax.set_xlabel(feature_name.replace("_", " "))
    ax.set_title("Distribution")
    ax.grid(alpha=0.12, linestyle=":")
    plt.tight_layout()
    out_fp = out_dir / f"distribution_{feature_name}.svg"
    fig.savefig(out_fp, bbox_inches="tight")
    fig.savefig(str(out_fp.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_fp.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote distribution single -> %s", out_fp)
    return str(out_fp.resolve()), cap_val


def plot_shap_marginal_single(
    feature_scaled: np.ndarray,
    coef_val: float,
    feature_name: str,
    out_dir: Path,
    dpi: int,
) -> str:
    """
    Standalone marginal effect plot (coef × scaled X).
    """
    safe_mkdir(out_dir)
    order = np.argsort(feature_scaled)
    xs_sorted = feature_scaled[order]
    marg_sorted = (coef_val * feature_scaled)[order]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs_sorted, marg_sorted, lw=1.8, color="#228833")
    ax.axhline(0, color="#222222", lw=0.6, linestyle="--", alpha=0.6)
    # faint rug
    try:
        rug_y = ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.scatter(
            feature_scaled,
            np.full_like(feature_scaled, rug_y),
            marker="|",
            s=12,
            color="#444444",
            alpha=0.35,
            linewidths=0.8,
            zorder=3,
        )
    except Exception:
        pass
    ax.set_xlabel(f"{feature_name} (scaled)")
    ax.set_title("Marginal effect (coef × scaled X)")
    ax.text(
        0.98,
        0.02,
        f"coef = {coef_val:.4g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.grid(alpha=0.12, linestyle=":")
    plt.tight_layout()
    out_fp = out_dir / f"marginal_{feature_name}.svg"
    fig.savefig(out_fp, bbox_inches="tight")
    fig.savefig(str(out_fp.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_fp.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote marginal single -> %s", out_fp)
    return str(out_fp.resolve())


def write_manifest(manifest_path: Path, produced: List[str], features_file: Path):
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


def parse_args():
    p = argparse.ArgumentParser(
        description="SHAP diagnostics for ElasticNet (scaled X)"
    )
    p.add_argument("--features-csv", default="data/processed/features_lean_imputed.csv")
    p.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="feature columns to include (order preserved)",
    )
    p.add_argument("--target", default="gdp_growth_pct")
    p.add_argument(
        "--three-panel",
        action="store_true",
        help="generate 3-panel SHAP diagnostic per feature",
    )
    p.add_argument(
        "--single-panels",
        action="store_true",
        help="also emit single-panel dependence/distribution/marginal plots per top-K feature (for app)",
    )
    p.add_argument("--scaler", default="artifacts/scaler.joblib")
    p.add_argument("--model", default="artifacts/en_model.joblib")
    p.add_argument("--outdir", default="reports/plot_shap_elasticnet")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--top-k", type=int, default=3, help="top-K dependence plots")
    return p.parse_args()


def main():
    args = parse_args()
    LOG.info("Starting SHAP diagnostics (scaled X).")
    features_path = Path(args.features_csv)
    out_base = safe_mkdir(Path(args.outdir))
    out_files_dir = safe_mkdir(out_base / "files")

    # load data
    df = pd.read_csv(features_path, low_memory=False)
    missing = [c for c in ([args.target] + args.features) if c not in df.columns]
    if missing:
        LOG.error("Missing columns in features CSV: %s", missing)
        raise SystemExit(1)
    df_sub = df[[args.target] + args.features].dropna()
    X_raw = df_sub[args.features].astype(float).values
    y = df_sub[args.target].astype(float).values
    feature_names = list(args.features)

    # load scaler and model
    scaler_path = Path(args.scaler)
    model_path = Path(args.model)
    if not scaler_path.exists():
        LOG.error("Scaler not found: %s", scaler_path)
        raise SystemExit(1)
    if not model_path.exists():
        LOG.error("Model not found: %s", model_path)
        raise SystemExit(1)

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # standardized X (scaled space used for SHAP)
    Xs = scaler.transform(X_raw)
    LOG.info("Scaled X shape: %s", Xs.shape)

    # extract coefficients
    coef, intercept, _ = extract_coeffs_from_model(model)
    if coef.size != Xs.shape[1]:
        LOG.warning(
            "coef length (%d) != n_features (%d). Attempting to align by trimming/padding.",
            coef.size,
            Xs.shape[1],
        )
        # attempt safe alignment
        if coef.size > Xs.shape[1]:
            coef = coef[: Xs.shape[1]]
        else:
            coef = np.pad(coef, (0, Xs.shape[1] - coef.size), constant_values=0.0)

    # compute SHAP values (exact for linear models)
    # shap_i_j = x_scaled_i_j * coef_j
    shap_values = Xs * coef.reshape(1, -1)
    # model prediction = intercept + sum(shap_values, axis=1)
    preds_from_shap = intercept + shap_values.sum(axis=1)
    # quick sanity check vs model.predict on scaled/raw as available
    try:
        # If model is a pipeline, calling predict may apply scaler twice; so prefer final estimator if available
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            # prefer to use final estimator directly if possible (model.named_steps)
            if "elasticnet" in model.named_steps:
                pred_direct = model.named_steps["elasticnet"].predict(Xs)
            else:
                # fallback to pipeline prediction on raw X
                pred_direct = model.predict(X_raw)
        else:
            pred_direct = model.predict(Xs)
        # allow small numerical differences
        max_diff = float(np.max(np.abs(preds_from_shap - pred_direct)))
        LOG.info(
            "Sanity check: max difference between SHAP-sum preds and model.predict = %g",
            max_diff,
        )
    except Exception:
        LOG.info("Sanity check skipped (model.predict not available or incompatible).")

    # feature importance: mean absolute SHAP
    mean_abs = pd.Series(np.mean(np.abs(shap_values), axis=0), index=feature_names)
    mean_abs_sorted = mean_abs.sort_values(ascending=False)
    # save importance table
    fi_csv = out_files_dir / "shap_feature_importance.csv"
    mean_abs_sorted.rename("mean_abs_shap").to_csv(fi_csv, index=True)
    LOG.info("Wrote %s", fi_csv)

    produced = [str(fi_csv.resolve())]

    # 1) summary bar
    produced += plot_summary_bar(mean_abs_sorted, out_files_dir, dpi=args.dpi)

    # 2) top-k plots (either simple dependence or full 3-panel)
    topk = list(mean_abs_sorted.index[: args.top_k])
    for feat in topk:
        i = feature_names.index(feat)
        feat_vals_raw = X_raw[:, i]
        feat_vals_scaled = Xs[:, i]
        feat_shap = shap_values[:, i]
        coef_val = float(coef[i]) if i < len(coef) else float(coef[0])
        feature_folder = safe_mkdir(out_base / feat)

        # 1) Create 3-panel figure if requested (paper)
        if args.three_panel:
            produced_files = plot_shap_3panel(
                feature_raw=feat_vals_raw,
                feature_scaled=feat_vals_scaled,
                shap_vals=feat_shap,
                coef_val=coef_val,
                feature_name=feat,
                out_dir=feature_folder,
                dpi=args.dpi,
                add_lowess=True,
            )
            produced.extend(produced_files)

        # 2) Create single-panel files for app if requested, and update existing feature meta
        if getattr(args, "single_panels", False):
            dep_fp = plot_shap_dependence_single(
                feat_vals_raw,
                feat_shap,
                feat,
                feature_folder,
                dpi=args.dpi,
                add_lowess=True,
            )
            dist_res = plot_shap_distribution_single(
                feat_vals_raw,
                feat,
                feature_folder,
                dpi=args.dpi,
                cap_pct=HIST_CAP_PCT.get(feat),
            )
            if isinstance(dist_res, tuple):
                dist_fp, cap_val = dist_res
            else:
                dist_fp, cap_val = dist_res, None
            marg_fp = plot_shap_marginal_single(
                feat_vals_scaled, coef_val, feat, feature_folder, dpi=args.dpi
            )

            produced.extend([dep_fp, dist_fp, marg_fp])

            meta_path = feature_folder / f"shap_3panel_{feat}.json"
            single_info = {
                "dependence": Path(dep_fp).name,
                "distribution": Path(dist_fp).name,
                "marginal": Path(marg_fp).name,
                "distribution_display_cap_pct": HIST_CAP_PCT.get(feat),
                "distribution_display_cap_val": (
                    float(cap_val) if cap_val is not None else None
                ),
            }
            try:
                _update_feature_meta_with_singleplots(meta_path, single_info)
            except Exception as e:
                LOG.warning("Failed to update meta for %s: %s", feat, e)

    meta = {
        "script": str(Path(__file__).resolve()),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "produced_files": produced,
        "features_file": str(features_path),
        "features_sha256": sha256_of_file(features_path),
        "scaler_file": str(scaler_path),
        "scaler_sha256": sha256_of_file(scaler_path),
        "model_file": str(model_path),
        "model_sha256": sha256_of_file(model_path),
        "model_git_commit": git_rev(),
        "n_rows": int(Xs.shape[0]),
        "n_features": int(Xs.shape[1]),
        "coef": {name: float(c) for name, c in zip(feature_names, coef.tolist())},
        "intercept": float(intercept),
    }
    meta_path = out_files_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    produced.append(str(meta_path.resolve()))
    LOG.info("Wrote %s", meta_path)

    # manifest at base
    manifest_path = out_base / "manifest.json"
    write_manifest(manifest_path, produced, features_path)

    LOG.info("Done. Outputs under %s", out_files_dir.resolve())
    print("Wrote files to:", out_files_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
