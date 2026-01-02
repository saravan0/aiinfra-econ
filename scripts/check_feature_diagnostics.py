"""
Feature-level diagnostic checks and correlation analysis.

Produces:
 - reports/correlation_matrix.csv
 - reports/correlation_heatmap.png
 - reports/feature_diagnostics.csv
 - reports/feature_diagnostics.md

Design notes:
 - Computes pairwise correlations and summary diagnostics for modeled features.
 - Generates tabular and visual outputs for inspection and reporting.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = logging.getLogger("check_feature_diagnostics")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

DEFAULT_FEATURES = Path("data/processed/features_lean_imputed.csv")
TRAIN_INDEX = Path("artifacts/train_index.csv")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_features(path: Path, apply_train_index: bool = True):
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    LOG.info("Loaded features: %s rows, %s cols", df.shape[0], df.shape[1])
    if apply_train_index and TRAIN_INDEX.exists():
        try:
            idx = pd.read_csv(TRAIN_INDEX, header=None).iloc[:, 0].astype(int).to_list()
            df = df.reset_index(drop=True).loc[idx].reset_index(drop=True)
            LOG.info("Applied train_index subset of length %d to features (to ensure SD/sample parity).", len(idx))
        except Exception as e:
            LOG.warning("Could not apply train_index: %s — proceeding with full features.", e)
    return df

def numeric_only(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).copy()

def corr_matrix_and_heatmap(df_num: pd.DataFrame, out_csv: Path, out_png: Path, figsize=(10,10), annotate_thresh=0.7):
    cm = df_num.corr()
    cm.to_csv(out_csv, index=True)
    LOG.info("Wrote correlation matrix -> %s", out_csv)

    # heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm.values, aspect='auto', interpolation='nearest')
    ax.set_xticks(np.arange(len(cm.columns)))
    ax.set_yticks(np.arange(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(cm.index, fontsize=6)
    plt.colorbar(im, ax=ax)
    # annotate large correlations for readability
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            val = cm.iat[i,j]
            if abs(val) >= annotate_thresh and i != j:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    LOG.info("Wrote correlation heatmap -> %s", out_png)
    return cm

def per_feature_summary(df: pd.DataFrame):
    rows = []
    df_num = numeric_only(df)
    for col in df_num.columns:
        ser = df_num[col]
        n = int(ser.notna().sum())
        total = int(len(ser))
        pct_missing = float(100*(total - n)/total) if total>0 else np.nan
        sd = float(ser.dropna().std(ddof=0)) if n>0 else np.nan
        mean = float(ser.dropna().mean()) if n>0 else np.nan
        uniq = int(ser.dropna().nunique())
        zeros = int((ser.dropna() == 0).sum()) if n>0 else 0
        skew = float(ser.dropna().skew()) if n>2 else np.nan
        kurt = float(ser.dropna().kurt()) if n>4 else np.nan
        rows.append({
            "feature": col,
            "n_nonmissing": n,
            "n_total": total,
            "pct_missing": pct_missing,
            "mean": mean,
            "sd": sd,
            "unique_vals": uniq,
            "n_zeros": zeros,
            "skew": skew,
            "kurtosis": kurt
        })
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)

def detect_flags(df_summary: pd.DataFrame, corr_mat: pd.DataFrame, corr_threshold: float, missing_threshold: float, zero_frac_threshold: float):
    flags = {}
    # high missing
    flags["high_missing"] = df_summary.loc[df_summary["pct_missing"] >= missing_threshold, "feature"].tolist()
    # low variance (sd == 0 or very small)
    flags["near_zero_variance"] = df_summary.loc[df_summary["sd"].fillna(0) <= 1e-12, "feature"].tolist()
    # many zeros
    flags["many_zeros"] = df_summary.loc[(df_summary["n_zeros"] / df_summary["n_nonmissing"].replace({0:np.nan})) >= zero_frac_threshold, "feature"].tolist()
    # high pairwise correlation groups
    high_pairs = []
    cols = list(corr_mat.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            v = corr_mat.iat[i,j]
            if abs(v) >= corr_threshold:
                high_pairs.append({"f1": cols[i], "f2": cols[j], "corr": float(v)})
    flags["high_corr_pairs"] = high_pairs
    # duplicated columns (exact equal across non-missing)
    dups = []
    numeric = corr_mat.columns.tolist()
    checked = set()
    for i, a in enumerate(numeric):
        for b in numeric[i+1:]:
            if a in checked or b in checked:
                continue
            try:
                equal_mask = df[a].equals(df[b])
                if equal_mask:
                    dups.append({"a":a, "b":b})
            except Exception:
                pass
    flags["duplicates"] = dups
    return flags

def write_markdown_report(out_md: Path, summary_df: pd.DataFrame, flags: dict, corr_threshold: float, missing_threshold: float, zero_frac_threshold: float):
    lines = []
    lines.append(f"# Feature diagnostics — Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append("## Summary (per-feature)")
    lines.append(f"- Total features (numeric): {len(summary_df)}")
    lines.append(f"- Correlation threshold (flag): {corr_threshold}")
    lines.append(f"- Missingness threshold (flag %): {missing_threshold}")
    lines.append(f"- Zero-fraction threshold (flag): {zero_frac_threshold}")
    lines.append("")
    lines.append("## High-missing features")
    lines.append("")
    if flags.get("high_missing"):
        for f in flags["high_missing"]:
            pct = float(summary_df.loc[summary_df["feature"]==f, "pct_missing"].iloc[0])
            lines.append(f"- {f} — pct_missing={pct:.2f}%")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Near-zero variance features")
    if flags.get("near_zero_variance"):
        for f in flags["near_zero_variance"]:
            sd = summary_df.loc[summary_df["feature"]==f, "sd"].iloc[0]
            lines.append(f"- {f} — sd={sd}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Many-zeros features")
    if flags.get("many_zeros"):
        for f in flags["many_zeros"]:
            zf = summary_df.loc[summary_df["feature"]==f, "n_zeros"].iloc[0]
            lines.append(f"- {f} — n_zeros={int(zf)}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## High-correlation pairs (abs(corr) >= {:.2f})".format(corr_threshold))
    if flags.get("high_corr_pairs"):
        for p in flags["high_corr_pairs"]:
            lines.append(f"- {p['f1']} <-> {p['f2']}: corr={p['corr']:.3f}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Exact duplicate columns")
    if flags.get("duplicates"):
        for d in flags["duplicates"]:
            lines.append(f"- {d['a']} == {d['b']}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Top 20 country observation counts (panel completeness)")
    # compute this if iso3, year exist in summary context; safe fallback below
    try:
        lines.append("")
    except Exception:
        pass
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf8")
    LOG.info("Wrote diagnostic markdown -> %s", out_md)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(DEFAULT_FEATURES), help="features CSV")
    parser.add_argument("--outdir", default=str(REPORT_DIR), help="reports output dir")
    parser.add_argument("--corr-threshold", type=float, default=0.8, help="abs(corr) threshold to flag high correlation")
    parser.add_argument("--missing-threshold", type=float, default=50.0, help="percent missing threshold to flag (0-100)")
    parser.add_argument("--zero-frac-threshold", type=float, default=0.9, help="fraction of non-missing that are zero to flag (0-1)")
    args = parser.parse_args()

    features_path = Path(args.features)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_features(features_path, apply_train_index=True)
    df_num = numeric_only(df)
    if df_num.shape[1] == 0:
        LOG.error("No numeric columns found in features.")
        return

    corr_csv = outdir / "correlation_matrix.csv"
    heatmap_png = outdir / "correlation_heatmap.png"
    corr_mat = corr_matrix_and_heatmap(df_num, corr_csv, heatmap_png, figsize=(12,12), annotate_thresh=0.85)

    summary_df = per_feature_summary(df)
    summary_csv = outdir / "feature_diagnostics.csv"
    summary_df.to_csv(summary_csv, index=False)
    LOG.info("Wrote per-feature diagnostics -> %s", summary_csv)

    flags = detect_flags(summary_df, corr_mat, corr_threshold=args.corr_threshold, missing_threshold=args.missing_threshold, zero_frac_threshold=args.zero_frac_threshold)
    md_path = outdir / "feature_diagnostics.md"
    write_markdown_report(md_path, summary_df, flags, corr_threshold=args.corr_threshold, missing_threshold=args.missing_threshold, zero_frac_threshold=args.zero_frac_threshold)

    # Also print brief console summary
    LOG.info("=== Quick summary ===")
    LOG.info("High-missing features: %s", flags.get("high_missing")[:10])
    LOG.info("Near-zero variance: %s", flags.get("near_zero_variance")[:10])
    LOG.info("High-correlation pairs (sample): %s", flags.get("high_corr_pairs")[:10])
    LOG.info("Reports written to %s", outdir.resolve())

if __name__ == "__main__":
    main()
