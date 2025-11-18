# src/model/predictor_coverage.py
"""
Predictor coverage analysis for the One Piece.

Outputs -> reports/
 - predictor_coverage.csv        (per-predictor: n_nonnull, pct_nonnull)
 - coverage_by_year.csv          (rows: year x predictor -> pct_nonnull)
 - coverage_by_country_top.csv   (top 50 countries coverage summary)
 - predictor_coverage.md         (short textual summary + remediation hints)
 - reports/predictor_coverage_heatmap.png (year x predictor heatmap)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
FEATURES = ROOT / "data" / "processed" / "features_lean.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---- CONFIG: change these if you want different predictor sets ----
DEFAULT_PREDICTORS = [
    "gov_index_zmean",
    "trade_exposure",
    "inflation_consumer_prices_pct",
    "fdi_inflow_usd_ln_safe",
    "imports_usd_ln_safe",
    "exports_usd_ln_safe",
    "gdp_usd_ln_safe",
    "gdp_per_capita_usd_ln_safe",
    "total_reserves_usd_ln_safe",
    "current_account_balance_usd_ln_safe",
]

IDENTIFIERS = ["country", "iso3", "year"]

def load_df(path: Path, nhead: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Features file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if nhead:
        return df.head(nhead).reset_index(drop=True)
    return df.reset_index(drop=True)

def predictor_coverage(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for p in predictors:
        present = p in df.columns
        if not present:
            rows.append({"predictor": p, "present": False, "n_nonnull": 0, "pct_nonnull": 0.0})
            continue
        nn = df[p].notna().sum()
        rows.append({"predictor": p, "present": True, "n_nonnull": int(nn), "pct_nonnull": float(nn/total)})
    return pd.DataFrame(rows)

def coverage_by_year(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    if "year" not in df.columns:
        raise SystemExit("No 'year' column in features table.")
    years = sorted(df["year"].dropna().unique())
    recs = []
    for y in years:
        sub = df[df["year"] == y]
        total = len(sub)
        for p in predictors:
            if p not in df.columns:
                recs.append({"year": int(y), "predictor": p, "n_nonnull": 0, "pct_nonnull": 0.0})
            else:
                nn = int(sub[p].notna().sum())
                recs.append({"year": int(y), "predictor": p, "n_nonnull": nn, "pct_nonnull": float(nn/total) if total>0 else 0.0})
    return pd.DataFrame(recs)

def coverage_by_country_top(df: pd.DataFrame, predictors: List[str], top_n: int = 50) -> pd.DataFrame:
    # compute average fraction of predictors present per country, then return top_n countries with most coverage
    countries = df["iso3"].dropna().unique()
    out = []
    for c in countries:
        sub = df[df["iso3"] == c]
        total = len(sub)
        if total == 0:
            continue
        present_counts = {p: int(sub[p].notna().sum()) if p in sub.columns else 0 for p in predictors}
        avg_pct = np.mean([present_counts[p] / total for p in predictors])
        out.append({"iso3": c, "n_obs": total, "avg_pct_predictors": float(avg_pct), **{f"{p}_n": present_counts[p] for p in predictors}})
    outdf = pd.DataFrame(out).sort_values("avg_pct_predictors", ascending=False).head(top_n)
    return outdf

def heatmap_year_predictor(df_year: pd.DataFrame, predictors: List[str], outpath: Path):
    # pivot: rows=year, cols=predictor, values=pct_nonnull
    pivot = df_year.pivot(index="year", columns="predictor", values="pct_nonnull").fillna(0.0)
    plt.figure(figsize=(max(6, len(predictors)*0.6), max(4, min(20, len(pivot)/2))))
    sns.heatmap(pivot, vmin=0.0, vmax=1.0, cmap="viridis", cbar_kws={"format":"%0.0f%%"}, annot=False)
    plt.title("Predictor coverage by year (fraction non-missing)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def write_summary_md(coverage_df: pd.DataFrame, by_year_df: pd.DataFrame, by_country_df: pd.DataFrame, out_path: Path):
    total_rows = len(df)
    lines = []
    lines.append("# Predictor coverage summary\n")
    lines.append(f"- features file: `{FEATURES}`")
    lines.append(f"- total rows (observations): **{total_rows:,}**\n")
    lines.append("## Per-predictor coverage\n")
    for _, r in coverage_df.sort_values("pct_nonnull", ascending=False).iterrows():
        pct = float(r["pct_nonnull"]) * 100
        lines.append(f"- `{r['predictor']}` — present={r['present']}, n_nonnull={r['n_nonnull']:,}, pct={pct:.1f}%")
    lines.append("\n## Year coverage (top / bottom years)\n")
    # compute average coverage per year across predictors
    year_avg = by_year_df.groupby("year")["pct_nonnull"].mean().reset_index().sort_values("pct_nonnull", ascending=False)
    top = year_avg.head(3)
    bottom = year_avg.tail(3)
    lines.append("Top years by avg predictor coverage:")
    for _, r in top.iterrows():
        lines.append(f"- {int(r['year'])}: avg_coverage={r['pct_nonnull']*100:.1f}%")
    lines.append("\nBottom years by avg predictor coverage:")
    for _, r in bottom.iterrows():
        lines.append(f"- {int(r['year'])}: avg_coverage={r['pct_nonnull']*100:.1f}%")
    lines.append("\n## Country-level (top sample)\n")
    for _, r in by_country_df.head(10).iterrows():
        lines.append(f"- {r['iso3']}: n_obs={int(r['n_obs'])}, avg_predictor_pct={r['avg_pct_predictors']*100:.1f}%")
    lines.append("\n## Quick recommendations")
    lines.append("- If any baseline predictor has <30% coverage → consider (i) imputing conservatively, (ii) dropping that predictor from baseline, OR (iii) expanding panel (add years/countries).")
    lines.append("- If many monetary predictors have tiny coverage but share the same pattern, consider creating a `monetary_available` flag and using group-wise imputation.")
    lines.append("- If coverage collapses after filtering by sample_filter, run the same analysis with `max_rows=None` and without the filter to see raw coverage.\n")
    out_path.write_text("\n".join(lines), encoding="utf8")

if __name__ == "__main__":
    # load
    df = load_df(FEATURES)
    predictors = DEFAULT_PREDICTORS
    coverage = predictor_coverage(df, predictors)
    coverage.to_csv(REPORTS / "predictor_coverage.csv", index=False)

    by_year = coverage_by_year(df, predictors)
    by_year.to_csv(REPORTS / "predictor_coverage_by_year.csv", index=False)

    by_country = coverage_by_country_top(df, predictors, top_n=50)
    by_country.to_csv(REPORTS / "predictor_coverage_by_country_top50.csv", index=False)

    # heatmap (year x predictor)
    heatmap_year_predictor(by_year, predictors, REPORTS / "predictor_coverage_heatmap.png")

    # markdown summary
    write_summary_md(coverage, by_year, by_country, REPORTS / "predictor_coverage.md")

    print("Wrote predictor coverage artifacts to", REPORTS)
