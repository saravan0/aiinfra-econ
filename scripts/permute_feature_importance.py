#!/usr/bin/env python3
# scripts/permute_feature_importance.py
"""
Permutation importance for ElasticNet model.

Usage:
    python scripts/permute_feature_importance.py \
        --features data/processed/features_lean_imputed.csv \
        --features-list gov_index_zmean trade_exposure inflation_consumer_prices_pct \
        --n-repeats 50 --cv 5

Outputs -> reports/figs/additional/perm_importance.png and CSV
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def perm_score(X_df, y, model_cls, col, random_state=None):
    Xp = X_df.copy()
    rng = np.random.RandomState(random_state)
    Xp[col] = rng.permutation(Xp[col].values)
    m = model_cls.fit(Xp.values, y)
    return m.score(Xp.values, y)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_lean_imputed.csv")
    p.add_argument("--features-list", nargs="+", required=True)
    p.add_argument("--target", default="gdp_growth_pct")
    p.add_argument("--n-repeats", type=int, default=50)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    F = Path(args.features)
    if not F.exists():
        raise FileNotFoundError(F)
    df = pd.read_csv(F, low_memory=False)
    cols = [args.target] + args.features_list
    sub = df[cols].dropna()
    X_df = sub[args.features_list].astype(float)
    y = sub[args.target].astype(float).values

    outdir = Path("reports/figs/additional/perm_importance")
    outdir.mkdir(parents=True, exist_ok=True)

    base_model = ElasticNetCV(cv=args.cv, l1_ratio=0.5, n_jobs=-1)
    base_model.fit(X_df.values, y)
    base_score = base_model.score(X_df.values, y)

    # repeat permutations per feature in parallel
    def feature_perm(col):
        res = Parallel(n_jobs=args.n_jobs)(
            delayed(perm_score)(X_df, y, ElasticNetCV(cv=args.cv, l1_ratio=0.5, n_jobs=1), col, i)
            for i in range(args.n_repeats)
        )
        res = np.array(res)
        # importance defined as drop in R^2 (positive => important)
        drop = base_score - res
        return drop

    drops = {}
    for col in args.features_list:
        drops[col] = feature_perm(col)
        pd.Series(drops[col]).to_csv(outdir / f"perm_{col}.csv", index=False)

    # summarize
    summary = {c: {"mean_drop": float(np.nanmean(drops[c])), "std_drop": float(np.nanstd(drops[c])), "n": len(drops[c])} for c in drops}
    pd.DataFrame(summary).T.to_csv(outdir / "perm_importance_summary.csv")
    # bar plot
    means = {c: summary[c]["mean_drop"] for c in summary}
    s = pd.Series(means).sort_values()
    plt.figure(figsize=(6,4))
    s.plot.barh()
    plt.xlabel("Mean drop in R^2 (permutation)")
    plt.title("Permutation importance (ElasticNet)")
    plt.tight_layout()
    plt.savefig(outdir / "perm_importance.png", dpi=300)
    plt.close()
    print("Wrote outputs to", outdir.resolve())

if __name__ == "__main__":
    main()
