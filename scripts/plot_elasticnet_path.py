#!/usr/bin/env python3
# scripts/plot_elasticnet_path.py
"""
ElasticNet coefficient paths + CV-MSE diagnostic (phd-polished version).

Produces:
 - en_path_polished.png/.pdf
 - en_cv_mse_polished.png/.pdf
 - en_cv_selected_coefs.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import enet_path, ElasticNetCV
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------
# Global styling (Nature/Science style)
# --------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

COLORS = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00"]  # clean pro palette


# --------------------------------------------------------
# Main Function
# --------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_lean_imputed.csv")
    p.add_argument("--features-list", nargs="+", required=True)
    p.add_argument("--target", default="gdp_growth_pct")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--l1-ratio", type=float, default=0.5)
    p.add_argument("--n-alphas", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.features, low_memory=False)
    cols = [args.target] + args.features_list
    df = df[cols].dropna()

    if df.shape[0] < 10:
        raise RuntimeError("Not enough data rows to compute EN path.")

    X = df[args.features_list].astype(float).values
    y = df[args.target].astype(float).values

    outdir = Path("reports/figs/additional/en_polished")
    outdir.mkdir(parents=True, exist_ok=True)

    # Standardize for consistent coefficient scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # --------------------------------------------------------
    # 1. ElasticNet coefficient path (α-path)
    # --------------------------------------------------------
    alphas, coefs, _ = enet_path(
        Xs, y,
        l1_ratio=args.l1_ratio,
        n_alphas=args.n_alphas
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, name in enumerate(args.features_list):
        ax.plot(
            alphas, coefs[i, :],
            label=name,
            linewidth=2.2,
            color=COLORS[i % len(COLORS)]
        )

    ax.set_xscale("log")
    ax.invert_xaxis()

    ax.set_xlabel("α (log scale — higher = stronger penalty)")
    ax.set_ylabel("Coefficient (standardized)")
    ax.set_title(f"ElasticNet coefficient path (l1_ratio={args.l1_ratio})")

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(outdir / "en_path_polished.png", bbox_inches="tight")
    fig.savefig(outdir / "en_path_polished.pdf", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 2. CV-MSE path (ElasticNetCV)
    # --------------------------------------------------------
    en_cv = ElasticNetCV(
        cv=args.cv,
        l1_ratio=args.l1_ratio,
        n_jobs=-1
    ).fit(Xs, y)

    alphas_cv = en_cv.alphas_
    mse_mean = en_cv.mse_path_.mean(axis=1)
    mse_std = en_cv.mse_path_.std(axis=1)
    best_alpha = en_cv.alpha_

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        alphas_cv, mse_mean,
        lw=2.4,
        color="#1f78b4",
        label="Mean CV MSE"
    )
    ax.fill_between(
        alphas_cv,
        mse_mean - mse_std,
        mse_mean + mse_std,
        color="#1f78b4",
        alpha=0.20
    )

    ax.axvline(best_alpha, color="#e31a1c", linestyle="--", linewidth=1.8,
               label=f"Chosen α = {best_alpha:.4g}")

    ax.set_xscale("log")
    ax.invert_xaxis()

    ax.set_xlabel("α (log scale)")
    ax.set_ylabel("CV Mean Squared Error")
    ax.set_title("ElasticNetCV — Cross-validated MSE path")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(outdir / "en_cv_mse_polished.png", bbox_inches="tight")
    fig.savefig(outdir / "en_cv_mse_polished.pdf", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # Save chosen coefficients
    # --------------------------------------------------------
    coef_df = pd.DataFrame({
        "feature": args.features_list,
        "coef_at_best_alpha": en_cv.coef_
    })

    coef_df.to_csv(outdir / "en_cv_selected_coefs.csv", index=False)

    print("Done — polished EN plots written to:", outdir.resolve())


if __name__ == "__main__":
    main()
