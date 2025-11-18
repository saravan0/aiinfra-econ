#!/usr/bin/env python3
"""
scripts/plot_placebo_test.py

Placebo predictor test (panel FE). For a given predictor variable:
 - compute the observed FE standardized effect (b * sd_x / sd_y)
 - generate a null distribution by applying a placebo transformation and refitting the FE model repeatedly
 - compute empirical p-value and plot histogram with observed statistic

Placebo strategies:
  - 'permute'         : global permutation of the predictor across all rows
  - 'within_entity'   : shuffle predictor within each entity (iso3) separately (preserves cross-entity distribution)
  - 'time_shift'      : circularly shift predictor within each entity by a random lag (preserves temporal autocorrelation structure)

Outputs:
  - reports/figs/additional/placebo_<var>_hist.png/.pdf
  - reports/figs/additional/placebo_<var>_summary.json
  - reports/figs/additional/placebo_<var>_null_draws.csv
"""

from pathlib import Path
import argparse
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from tqdm import trange

# --- styling ---
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14
})

OUTDIR = Path("reports/figs/additional")
OUTDIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Utilities
# -------------------------
def safe_to_numeric(df):
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    return df


def fit_fe_and_standardized_effect(df, predictor, target="gdp_growth_pct", entity_col="iso3", drop_first=True):
    """
    Robust FE fit that coerces numeric types before calling statsmodels.
    Returns standardized effect (coef * sd_x / sd_y) and fit stats or None on failure.
    """
    # copy relevant columns and coerce to numeric safely
    sub = df[[target, predictor, entity_col]].copy()

    # coerce predictor / target to numeric (turn non-numeric -> NaN)
    sub[predictor] = pd.to_numeric(sub[predictor], errors="coerce")
    sub[target] = pd.to_numeric(sub[target], errors="coerce")

    # drop rows where any of these are missing now
    sub = sub.dropna(subset=[target, predictor, entity_col])
    if sub.shape[0] < 10:
        # not enough rows to fit reliably
        return None

    # One-hot encode FE dummies; ensure resulting columns are numeric
    fe = pd.get_dummies(sub[entity_col].astype(str), prefix="FE", drop_first=drop_first).astype(float)

    # Build design with the predictor as "Xpred" (float)
    X_pred = sub[[predictor]].astype(float).rename(columns={predictor: "Xpred"})
    X = pd.concat([X_pred, fe], axis=1)

    # Align y and design (drop rows with any NA)
    y = sub[target].astype(float)
    Xc = sm.add_constant(X, has_constant="add")
    mask = Xc.notna().all(axis=1) & y.notna()
    Xc = Xc.loc[mask]
    y = y.loc[mask]
    if len(y) < 10:
        return None

    # final safety: convert to numpy arrays of float
    try:
        Xc_mat = Xc.astype(float).to_numpy()
        y_vec = y.astype(float).to_numpy()
    except Exception:
        return None

    # fit OLS and extract coefficient
    res = sm.OLS(y_vec, Xc_mat).fit()

    # find index/location of Xpred in Xc columns (we used Xc.columns earlier)
    # But because we passed numpy matrices to sm.OLS, we must retrieve coef position:
    # Xc.columns exists — use it to locate "Xpred" position
    try:
        pred_idx = list(Xc.columns).index("Xpred")
    except ValueError:
        # fallback: assume predictor is first (after const)
        try:
            pred_idx = list(Xc.columns).index("Xpred")
        except Exception:
            return None

    coef = float(res.params[pred_idx])
    # try to get standard error and pvalue if available (res.bse may be None for numpy input)
    se = None
    pval = None
    try:
        se = float(res.bse[pred_idx])
        pval = float(res.pvalues[pred_idx])
    except Exception:
        se = None
        pval = None

    sd_y = float(y.std(ddof=0))
    sd_x = float(X_pred["Xpred"].std(ddof=0))
    std_effect = coef * (sd_x / (sd_y if sd_y != 0 else 1.0))

    out = {
        "coef": coef,
        "std_err": se,
        "pvalue": pval,
        "n_obs": int(len(y)),
        "sd_x": sd_x,
        "sd_y": sd_y,
        "standardized_effect": std_effect,
    }
    return out



# -------------------------
# Placebo generators
# -------------------------
def placebo_permute(df, predictor, seed=None):
    """Global permutation of predictor values across rows."""
    rng = np.random.RandomState(seed)
    new = df.copy()
    perm = rng.permutation(len(df))
    new[predictor] = df[predictor].values[perm]
    return new


def placebo_within_entity(df, predictor, entity_col="iso3", seed=None):
    """Shuffle predictor values within each entity (iso3)."""
    rng = random.Random(seed)
    new = df.copy()
    groups = new.groupby(entity_col).groups
    for g, idxs in groups.items():
        vals = new.loc[idxs, predictor].values
        # shuffle in-place with RNG
        vals = list(vals)
        rng.shuffle(vals)
        new.loc[idxs, predictor] = vals
    return new


def placebo_time_shift(df, predictor, entity_col="iso3", seed=None):
    """Circularly shift predictor within each entity by a random lag (0..n-1)."""
    rng = random.Random(seed)
    new = df.copy()
    for g, group in new.groupby(entity_col):
        idx = group.index.to_numpy()
        n = len(idx)
        if n <= 1:
            continue
        lag = rng.randint(1, n - 1)
        vals = group[predictor].values
        shifted = np.roll(vals, lag)
        new.loc[idx, predictor] = shifted
    return new


# -------------------------
# Runner
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_lean_imputed.csv")
    p.add_argument("--var", required=True, help="Predictor variable to test (e.g. trade_exposure)")
    p.add_argument("--n", type=int, default=500, help="Number of placebo draws (default 500)")
    p.add_argument("--strategy", choices=["permute", "within_entity", "time_shift"], default="within_entity")
    p.add_argument("--target", default="gdp_growth_pct")
    p.add_argument("--entity-col", default="iso3")
    p.add_argument("--seed", type=int, default=20251117)
    p.add_argument("--drop-first-fe", action="store_true", help="Drop first FE dummy (default True behavior kept)")
    args = p.parse_args()

    df = pd.read_csv(args.features, low_memory=False)
    df = safe_to_numeric(df)

    # compute observed
    obs = fit_fe_and_standardized_effect(df, args.var, target=args.target, entity_col=args.entity_col, drop_first=not args.drop_first_fe)
    if obs is None:
        raise RuntimeError("Could not compute observed effect — check variable presence and non-missing rows.")

    observed = obs["standardized_effect"]

    # prepare placebo generator
    gen = placebo_within_entity if args.strategy == "within_entity" else (placebo_time_shift if args.strategy == "time_shift" else placebo_permute)

    # run draws
    rng_seed = int(args.seed)
    null_draws = []
    n = int(args.n)
    print(f"Running placebo test: var={args.var} strategy={args.strategy} draws={n}")

    for i in trange(n):
        seed_i = rng_seed + i
        place_df = gen(df, args.var, entity_col=args.entity_col, seed=seed_i)
        res = fit_fe_and_standardized_effect(place_df, args.var, target=args.target, entity_col=args.entity_col, drop_first=not args.drop_first_fe)
        if res is None:
            null_draws.append(np.nan)
        else:
            null_draws.append(res["standardized_effect"])

    null_arr = np.array(null_draws, dtype=float)
    # remove NaNs
    null_clean = null_arr[~np.isnan(null_arr)]
    draws_used = len(null_clean)

    # empirical p-values (two-sided)
    if draws_used == 0:
        raise RuntimeError("All placebo draws failed (NaNs). Increase data or check predictor.")
    p_two = np.mean(np.abs(null_clean) >= abs(observed))
    p_one_pos = np.mean(null_clean >= observed)
    p_one_neg = np.mean(null_clean <= observed)

    summary = {
        "variable": args.var,
        "strategy": args.strategy,
        "n_requested": n,
        "n_used": draws_used,
        "observed_standardized_effect": observed,
        "empirical_p_two_sided": float(p_two),
        "empirical_p_one_sided_pos": float(p_one_pos),
        "empirical_p_one_sided_neg": float(p_one_neg),
        "observed_details": obs,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    # save null draws CSV
    df_null = pd.DataFrame({
        "draw_index": np.arange(len(null_arr)),
        "standardized_effect": null_arr
    })
    csv_out = OUTDIR / f"placebo_{args.var}_null_draws.csv"
    df_null.to_csv(csv_out, index=False)

    # save summary JSON
    json_out = OUTDIR / f"placebo_{args.var}_summary.json"
    json_out.write_text(json.dumps(summary, indent=2))

    # ----------------- plotting -----------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    # histogram of null
    ax.hist(null_clean, bins=40, density=False, alpha=0.85, color="#7f8c8d", edgecolor="k")
    # mark observed
    ax.axvline(observed, color="#e31a1c", lw=2.2, label=f"Observed = {observed:.3f}")
    # mark central 95% of null
    lo95, hi95 = np.percentile(null_clean, [2.5, 97.5])
    ax.axvline(lo95, color="#1f78b4", linestyle="--", lw=1.4, label="2.5% / 97.5% null")
    ax.axvline(hi95, color="#1f78b4", linestyle="--", lw=1.4)
    ax.set_xlabel("Standardized effect (σ target)")
    ax.set_ylabel("Count")
    ax.set_title(f"Placebo null distribution — {args.var} ({args.strategy})")
    ax.legend(frameon=False, fontsize=11)
    foot = f"n_draws={n}, used={draws_used}, two-sided p={p_two:.3g}"
    fig.text(0.01, 0.01, foot, fontsize=9)
    plt.tight_layout()
    png_out = OUTDIR / f"placebo_{args.var}_hist.png"
    pdf_out = OUTDIR / f"placebo_{args.var}_hist.pdf"
    fig.savefig(png_out, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", csv_out, json_out, png_out)
    print("Summary p(two-sided):", p_two)


if __name__ == "__main__":
    main()
