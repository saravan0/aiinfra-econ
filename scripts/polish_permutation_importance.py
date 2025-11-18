#!/usr/bin/env python3
"""
scripts/polish_permutation_importance.py

Cross-validated permutation importance polishing script.

Example:
    python scripts/polish_permutation_importance.py \
      --model artifacts/en_model.joblib \
      --features-csv data/processed/features_lean_imputed.csv \
      --features-list gov_index_zmean trade_exposure inflation_consumer_prices_pct \
      --repeats 500 --cv 5 --top-k 10 \
      --outdir reports/figs/polished/permutation

Notes:
 - This performs per-fold permutation_importance on the test fold and aggregates across folds.
 - Scoring defaults to 'neg_mean_squared_error' (we report ΔMSE = permuted_mse - baseline_mse).
"""

from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import load as joblib_load
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from typing import List, Dict, Any

# Aesthetics
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def safe_load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib_load(path)

def load_feature_names_from_artifact_or_list(artifact_model, features_arg: List[str], feature_names_path: Path):
    # prefer explicit list
    if features_arg:
        return list(features_arg)
    # try feature_names.json
    if feature_names_path and feature_names_path.exists():
        try:
            return json.loads(feature_names_path.read_text(encoding="utf8"))
        except Exception:
            pass
    # try to infer from pipeline (feature_names_in_)
    if hasattr(artifact_model, "feature_names_in_"):
        try:
            return list(artifact_model.feature_names_in_)
        except Exception:
            pass
    # nothing found
    raise ValueError("Could not determine feature names. Provide --features-list or artifacts/feature_names.json")

def build_Xy_from_csv(csv_path: Path, feature_names: List[str], target_col: str):
    if not csv_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    # ensure numeric where possible
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Feature names not found in CSV: {missing}")
    X = df[feature_names].copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in CSV")
    y = df[target_col].copy()
    # drop rows with NA in X or y
    full_mask = X.notna().all(axis=1) & y.notna()
    X_clean = X.loc[full_mask].reset_index(drop=True)
    y_clean = y.loc[full_mask].reset_index(drop=True)
    return X_clean, y_clean

def cv_permutation_importance(estimator, X: pd.DataFrame, y: pd.Series,
                              cv: int = 5, repeats: int = 200, scoring: str = "neg_mean_squared_error",
                              random_state: int = 42, verbose: bool = True):
    """
    Performs CV per-fold permutation importance.
    Returns:
      results: dict mapping feature -> list of importance values (one per fold*repeat)
      baseline_scores: list of baseline scores per fold
    Importance definition used here: Delta(MSE) = permuted_mse - baseline_mse
    (so positive = feature helpful; larger positive => bigger harm when permuted)
    """
    rng = np.random.RandomState(random_state)
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    n_features = X.shape[1]
    feature_names = list(X.columns)
    aggregated = {f: [] for f in feature_names}
    baseline_scores = []
    fold_idx = 0

    iter_splits = list(kf.split(X))
    if verbose:
        print(f"Running CV permutation importance: folds={cv}, repeats={repeats}, features={n_features}")

    for train_idx, test_idx in (tqdm(iter_splits, desc="folds") if verbose else iter_splits):
        fold_idx += 1
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # fit estimator fresh on train
        try:
            est = estimator
            # if estimator is a pipeline or object reused, clone by loading a fresh instance if possible
            # but simplest: fit a fresh one only if estimator has a fit method and is not already fitted
            est_fold = est
            # try fit; if fails because estimator already fitted with data shape mismatch, attempt to clone via joblib load
            est_fold.fit(X_train, y_train)
        except Exception as e:
            # fallback: try to deep-copy via joblib_load of original path if provided (not available here)
            # re-raise with helpful message
            raise RuntimeError(f"Failed to fit estimator on fold {fold_idx}: {e}")

        # baseline predictions & baseline mse
        y_pred = est_fold.predict(X_test)
        baseline_mse = mean_squared_error(y_test, y_pred)
        baseline_r2 = r2_score(y_test, y_pred)
        baseline_scores.append({"mse": baseline_mse, "r2": baseline_r2})

        # Use sklearn's permutation_importance on test set for efficiency
        # It computes decrease in scoring; for neg_mean_squared_error we convert accordingly.
        # We'll compute raw permuted MSE difference ourselves by using scoring=None path: use permutation_importance with scoring that returns neg_mse.
        try:
            perm = permutation_importance(est_fold, X_test, y_test,
                                          n_repeats=repeats, random_state=rng, scoring=scoring, n_jobs=1)
            # permutation_importance returns importances relative to scoring (higher = better). For neg_mean_squared_error,
            # higher neg_mse (less negative) means better; we want DeltaMSE = permuted_mse - baseline_mse.
            # sklearn's permutation_importance returns importances as differences in scoring: result.importances_mean is baseline_score - permuted_score (for scorers where higher is better).
            # For neg_mean_squared_error: importance_sklearn = neg_mse_baseline - neg_mse_permuted = -(mse_baseline - mse_permuted) = (mse_permuted - mse_baseline) * -1
            # That sign is confusing; simpler: compute permuted MSEs directly from perm.importances (which are in scoring units), but we'll compute per-repeat MSEs by re-evaluating predictions with shuffled columns (safer and explicit).
        except Exception:
            perm = None

        # Explicit per-repeat loop (clear and control over metric sign). This is heavier but precise.
        for feat_i, feat in enumerate(feature_names):
            # perform 'repeats' permuted test evaluations
            for r in range(repeats):
                X_test_perm = X_test.copy(deep=True)
                # shuffle the column in-place
                X_test_perm.iloc[:, feat_i] = X_test_perm.iloc[:, feat_i].sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)
                yp = est_fold.predict(X_test_perm)
                permuted_mse = mean_squared_error(y_test, yp)
                delta_mse = permuted_mse - baseline_mse  # positive => feature is helpful (permuting hurts)
                aggregated[feat].append(delta_mse)

    return aggregated, baseline_scores

def summarize_aggregated(aggregated: Dict[str, List[float]]):
    rows = []
    for feat, vals in aggregated.items():
        arr = np.asarray(vals, dtype=float)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr, ddof=1))
        n = int(np.sum(~np.isnan(arr)))
        se = std / math.sqrt(n) if n > 0 else float("nan")
        ci_low = mean - 1.96 * se if n > 0 else float("nan")
        ci_high = mean + 1.96 * se if n > 0 else float("nan")
        # p-value: proportion of permuted deltas <= 0 (if delta>0 beneficial)
        p_emp = float(np.mean(arr <= 0)) if n > 0 else float("nan")
        rows.append({"feature": feat, "mean_delta_mse": mean, "std": std, "n": n,
                     "se": se, "ci_low": ci_low, "ci_high": ci_high, "p_emp_le_zero": p_emp})
    df = pd.DataFrame(rows).sort_values("mean_delta_mse", ascending=False).reset_index(drop=True)
    return df

def plot_bar_topk(df_summary: pd.DataFrame, top_k: int, out_prefix: Path):
    df = df_summary.head(top_k).copy().sort_values("mean_delta_mse")
    fig, ax = plt.subplots(figsize=(9, max(3, top_k * 0.45)))
    ax.barh(df["feature"], df["mean_delta_mse"], xerr=[df["mean_delta_mse"] - df["ci_low"], df["ci_high"] - df["mean_delta_mse"]],
            align="center", capsize=4)
    ax.set_xlabel("Mean ΔMSE (permuted_mse - baseline_mse)")
    ax.set_title(f"Permutation importance (top {top_k})")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        outp = out_prefix.with_suffix("." + ext)
        fig.savefig(outp, bbox_inches="tight", dpi=600)
        print("Wrote", outp)
    plt.close(fig)

def plot_violin_top5(aggregated: Dict[str, List[float]], top5: List[str], out_prefix: Path):
    # build long table
    records = []
    for feat in top5:
        vals = aggregated.get(feat, [])
        for v in vals:
            records.append({"feature": feat, "delta_mse": v})
    df = pd.DataFrame.from_records(records)
    if df.empty:
        print("No data for violin plot.")
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.violinplot(x="delta_mse", y="feature", data=df, inner=None, scale="width", cut=0, orient="h")
    sns.stripplot(x="delta_mse", y="feature", data=df, size=2, color="k", alpha=0.3, orient="h")
    ax.set_xlabel("ΔMSE (permuted - baseline)")
    ax.set_title("Distribution of permutation ΔMSE (top features)")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        outp = out_prefix.with_suffix("." + ext)
        fig.savefig(outp, bbox_inches="tight", dpi=600)
        print("Wrote", outp)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to trained model artifact (joblib/pickle) - should implement fit/predict.")
    p.add_argument("--feature-names", default="artifacts/feature_names.json", help="Optional JSON file listing feature names in order.")
    p.add_argument("--features-csv", required=True, help="Features CSV (processed) containing features and target.")
    p.add_argument("--features-list", nargs="+", help="Explicit list of feature names to evaluate (overrides feature-names file).")
    p.add_argument("--target", default="gdp_growth_pct", help="Target column name in features CSV.")
    p.add_argument("--repeats", type=int, default=500, help="Permutation repeats per fold (recommended 200-1000).")
    p.add_argument("--cv", type=int, default=5, help="CV folds.")
    p.add_argument("--top-k", type=int, default=10, help="Top-K features to show in bar plot.")
    p.add_argument("--outdir", default="reports/figs/polished/permutation", help="Output directory.")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_prefix = outdir / "perm_importance_top{}".format(args.top_k)

    # load model
    model_path = Path(args.model)
    model = safe_load_model(model_path)

    # feature names
    feature_names_path = Path(args.feature_names) if args.feature_names else None
    try:
        feat_names = load_feature_names_from_artifact_or_list(model, args.features_list or [], feature_names_path)
    except Exception as e:
        raise RuntimeError(f"Could not obtain feature names: {e}")

    # load X,y
    X, y = build_Xy_from_csv(Path(args.features_csv), feat_names, args.target)
    print(f"Loaded X ({X.shape[0]} rows, {X.shape[1]} cols) and y ({len(y)})")

    # run CV permutation importances
    aggregated, baseline_scores = cv_permutation_importance(model, X, y,
                                                            cv=args.cv, repeats=args.repeats,
                                                            scoring="neg_mean_squared_error",
                                                            random_state=args.random_state,
                                                            verbose=True)

    # summarize
    summary_df = summarize_aggregated(aggregated)
    csv_out = outdir / "perm_importance_summary.csv"
    summary_df.to_csv(csv_out, index=False)
    print("Wrote", csv_out)

    # save full distributions as a wide CSV (features in cols)
    wide = pd.DataFrame({k: pd.Series(v) for k, v in aggregated.items()})
    wide_out = outdir / "perm_importance_distributions.csv"
    wide.to_csv(wide_out, index=False)
    print("Wrote", wide_out)

    # plots
    plot_bar_topk(summary_df, args.top_k, outdir / f"perm_importance_top{args.top_k}")
    top5 = summary_df.head(5)["feature"].tolist()
    plot_violin_top5(aggregated, top5, outdir / "perm_importance_top5_violin")

    # write a small manifest/caption for the figure
    caption = (
        f"Permutation importance computed with {args.repeats} repeats per fold, {args.cv}-fold CV. "
        f"Importance = mean ΔMSE (permuted_mse - baseline_mse) across folds; positive values indicate features that reduce MSE when unpermuted "
        f"(i.e., make predictions better)."
    )
    (outdir / "perm_importance_caption.txt").write_text(caption, encoding="utf-8")

    print("Wrote caption.")

    # lightweight audit
    manifest = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "repeats": int(args.repeats),
        "cv_folds": int(args.cv),
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z"
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote manifest.json")

    print("Done. Outputs in", outdir.resolve())

if __name__ == "__main__":
    main()
