"""
ElasticNet coefficient path & CV diagnostics

Publication-ready figures:
  reports/plot_elasticnet_paths/files/en_path.(pdf|svg|png)
  reports/plot_elasticnet_paths/files/en_cv_mse.(pdf|svg|png)
  reports/plot_elasticnet_paths/files/en_cv_selected_coefs.csv
  reports/plot_elasticnet_paths/meta.json

Features of this version (journal-ready):
 - Vector-first output (PDF + SVG) with embedded TrueType fonts (pdf.fonttype=42)
 - High-resolution PNG export for raster needs (dpi default 600)
 - Clean typography and smaller, publication-appropriate title
 - Colorblind-safe palette and clear line weights
 - Explicit annotation for selected alpha (vertical line + boxed label)
 - Exports metadata (sha256 of features file, git commit)
 - Robust I/O and helpful INFO logs
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import enet_path, ElasticNetCV
from sklearn.preprocessing import StandardScaler

# --- logging
LOG = logging.getLogger("plot_elasticnet_paths")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# --- rcParams tuned for journal-quality figures
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,   # embed TrueType fonts in PDF
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,    # reduced header size (journal style)
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# colorblind-friendly palette
PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]

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
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf8").strip()
    except Exception:
        return None

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_coef_path(alphas: np.ndarray, coefs: np.ndarray, feature_names: List[str],
                   out_files_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(8.0, 4.5))  # aspect tuned for journals
    for i, name in enumerate(feature_names):
        ax.plot(alphas, coefs[i, :], label=name, linewidth=2.0, color=PALETTE[i % len(PALETTE)])
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\alpha$ (log scale — higher = stronger penalty)")
    ax.set_ylabel("Coefficient (standardized)")
    ax.set_title(f"ElasticNet coefficient path (l1_ratio={args.l1_ratio})", pad=8)
    ax.grid(alpha=0.22, linestyle=":")
    # legend outside plot area for publication figures
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    # outputs
    for ext in ("pdf", "svg", "png"):
        fp = out_files_dir / f"en_path.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(fp, bbox_inches="tight")
        LOG.info("Wrote %s", fp)
    plt.close(fig)

def plot_cv_mse(alphas_cv: np.ndarray, mse_mean: np.ndarray, mse_std: np.ndarray,
                best_alpha: float, out_files_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(alphas_cv, mse_mean, lw=2.2, color=PALETTE[0], label="Mean CV MSE")
    ax.fill_between(alphas_cv, mse_mean - mse_std, mse_mean + mse_std, color=PALETTE[0], alpha=0.18)
    # vertical line for chosen alpha
    if best_alpha is not None:
        ax.axvline(best_alpha, color="#D55E00", linestyle="--", linewidth=1.6)
        # boxed annotation placed to the right of the line
        text = f"α = {best_alpha:.3g}"
        ax.text(best_alpha * 1.05, np.nanmax(mse_mean), text,
                va="top", ha="left", rotation=0, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#333333", boxstyle="round,pad=0.3", alpha=0.95))
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\alpha$ (log scale)")
    ax.set_ylabel("Cross-validated MSE")
    ax.set_title("ElasticNetCV — cross-validated MSE", pad=8)
    ax.grid(alpha=0.22, linestyle=":")
    ax.legend(loc="lower left", frameon=False)
    plt.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fp = out_files_dir / f"en_cv_mse.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(fp, bbox_inches="tight")
        LOG.info("Wrote %s", fp)
    plt.close(fig)

def write_meta(meta_path: Path, produced_files: list, features_file: Path | None):
    meta = {
        "script": str(Path(__file__).resolve()),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "produced_files": produced_files,
        "features_file": str(features_file) if features_file else None,
        "features_sha256": sha256_of_file(features_file) if features_file and features_file.exists() else None,
        "git_commit": git_rev(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    LOG.info("Wrote meta -> %s", meta_path)

def parse_args():
    p = argparse.ArgumentParser(description="Journal-ready ElasticNet coefficient path + CV diagnostics")
    p.add_argument("--features", default="data/processed/features_lean_imputed.csv",
                   help="features CSV (must contain features and target)")
    p.add_argument("--features-list", nargs="+", required=True,
                   help="list of feature column names to include in EN path (order preserved)")
    p.add_argument("--target", default="gdp_growth_pct")
    p.add_argument("--l1-ratio", type=float, default=0.5)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--n-alphas", type=int, default=200)
    p.add_argument("--outdir", default="reports/plot_elasticnet_paths")
    p.add_argument("--dpi", type=int, default=600, help="PNG dpi (raster export)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    LOG.info("Loading features: %s", args.features)
    features_path = Path(args.features)
    df = pd.read_csv(features_path, low_memory=False)

    # keep only requested columns + target
    cols = [args.target] + list(args.features_list)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        LOG.error("Missing columns in features CSV: %s", missing)
        raise SystemExit(1)
    df = df[cols].dropna()
    if df.shape[0] < 10:
        LOG.error("Too few rows after dropping NA (%d). Need more data.", df.shape[0])
        raise SystemExit(1)

    X = df[list(args.features_list)].astype(float).values
    y = df[args.target].astype(float).values

    out_base = Path(args.outdir)
    out_files = safe_mkdir(out_base / "files")

    # standardize X for plotting coefficient path
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 1) coefficient path (enet_path)
    LOG.info("Computing ElasticNet coefficient path...")
    alphas, coefs, _ = enet_path(Xs, y, l1_ratio=args.l1_ratio, n_alphas=args.n_alphas)

    # save path figure
    plot_coef_path(alphas, coefs, list(args.features_list), out_files, dpi=args.dpi)

    # 2) ElasticNetCV for CV-MSE and chosen alpha
    LOG.info("Running ElasticNetCV...")
    en_cv = ElasticNetCV(cv=args.cv, l1_ratio=args.l1_ratio, n_jobs=-1).fit(Xs, y)

    alphas_cv = en_cv.alphas_
    mse_mean = en_cv.mse_path_.mean(axis=1)
    mse_std = en_cv.mse_path_.std(axis=1)
    best_alpha = float(en_cv.alpha_)

    plot_cv_mse(alphas_cv, mse_mean, mse_std, best_alpha, out_files, dpi=args.dpi)

    # save chosen coefficients (returned in original scale standardized for Xs)
    coef_df = pd.DataFrame({
        "feature": list(args.features_list),
        "coef_at_best_alpha": en_cv.coef_
    })
    coef_csv = out_files / "en_cv_selected_coefs.csv"
    coef_df.to_csv(coef_csv, index=False)
    LOG.info("Wrote %s", coef_csv)

    # metadata
    produced = [str(p) for p in sorted(out_files.glob("*"))]
    meta_path = out_base / "meta.json"
    write_meta(meta_path, produced, features_file=features_path)

    LOG.info("Done — journal-ready EN figures in: %s", out_files.resolve())
    print("Wrote files to:", out_files)
    print("Meta:", meta_path)
