"""
plot_comparative_model_effects — Comparative forest plot of model effects

Forest plot that visualizes standardized effect sizes (in SD units)
estimated across three estimation strategies:
- FE (within-country fixed effects)
- OLS
- ElasticNet

Inputs:
  - Baseline snapshot:
        reports/generate_baseline_snapshot/generate_baseline_snapshot.json
  - Comparison table:
        reports/generate_baseline_snapshot/model_comparison_table.csv

Outputs:
  reports/plot_comparative_model_effects/files/plot_comparative_model_effects.(png|pdf|svg)
  reports/plot_comparative_model_effects/plot_comparative_model_effects_meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LOG = logging.getLogger("plot_comparative_model_effects")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(".")
SNAP = ROOT / "reports" / "generate_baseline_snapshot" / "generate_baseline_snapshot.json"
MODEL_TABLE = ROOT / "reports" / "generate_baseline_snapshot" / "model_comparison_table.csv"

OUT_BASE = ROOT / "reports" / "plot_comparative_model_effects"
OUT_FILES = OUT_BASE / "files"
OUT_FILES.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
sns.set_style("white")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_snapshot() -> List[Dict[str, Any]]:
    """Load the new baseline snapshot (list of dicts)."""
    if not SNAP.exists():
        LOG.error("Snapshot not found: %s", SNAP)
        return []
    try:
        return json.loads(SNAP.read_text(encoding="utf8"))
    except Exception as e:
        LOG.error("Failed to load snapshot: %s", e)
        return []

def load_model_table() -> pd.DataFrame:
    if not MODEL_TABLE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(MODEL_TABLE)
    except Exception:
        return pd.DataFrame()

# ----------------------------------------------------------------------------
# Extract models from new snapshot structure
# ----------------------------------------------------------------------------
def extract_from_entry(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract FE, OLS, ElasticNet from baseline snapshot entry."""
    out = {}
    std = entry.get("standardized_summary") or {}

    summary = std.get("summary") if isinstance(std, dict) else None
    if not isinstance(summary, dict):
        return out

    for model in ("FE", "OLS", "ElasticNet"):
        mv = summary.get(model)
        if not isinstance(mv, dict):
            continue
        out[model] = {
            "standardized": mv.get("standardized"),
            "coef": mv.get("coef"),
            "std_err": mv.get("std_err"),
            "ci": mv.get("ci"),
            "n": mv.get("n_obs"),
        }
    return out

# ----------------------------------------------------------------------------
# Build plotting frame
# ----------------------------------------------------------------------------
def build_plot_frame(vars_list: List[str], snapshot: List[Dict[str, Any]], mtable: pd.DataFrame) -> pd.DataFrame:
    rows = []

    idx_map = {entry.get("variable"): entry for entry in snapshot}

    for var in vars_list:
        entry = idx_map.get(var)

        if entry:
            models = extract_from_entry(entry)
            for mname, mv in models.items():
                rows.append({
                    "variable": var,
                    "model": mname,
                    "standardized": mv.get("standardized"),
                    "coef": mv.get("coef"),
                    "std_err": mv.get("std_err"),
                    "ci": mv.get("ci"),
                    "n": mv.get("n"),
                })
            continue

        # fallback to model table
        if not mtable.empty:
            sub = mtable[mtable["variable"] == var]
            for _, r in sub.iterrows():
                rows.append({"variable": var, "model": "FE",
                             "standardized": r.get("fe_std_effect"),
                             "coef": r.get("fe_coef"),
                             "std_err": None})
                rows.append({"variable": var, "model": "OLS",
                             "standardized": r.get("ols_std_effect"),
                             "coef": r.get("ols_coef"),
                             "std_err": None})
                rows.append({"variable": var, "model": "ElasticNet",
                             "standardized": r.get("en_std_effect"),
                             "coef": r.get("en_coef"),
                             "std_err": None})

    df = pd.DataFrame(rows)
    return df

# ----------------------------------------------------------------------------
# CI helper
# ----------------------------------------------------------------------------
def ci_from_coef_se(coef, se, z=1.96):
    try:
        return (coef - z * se, coef + z * se)
    except Exception:
        return (None, None)

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_polished(df: pd.DataFrame, out_prefix: Path, dpi_png: int = 600):
    if df.empty:
        raise RuntimeError("No data to plot.")

    df = df.copy()
    use_std = not df["standardized"].isna().all()
    xcol = "standardized" if use_std else "coef"
    title = "Standardized effects (σ of target) by model" if use_std else "Raw coefficients"

    def low_high(row):
        ci = row.get("ci")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            return ci
        if pd.notna(row.get("std_err")) and pd.notna(row.get(xcol)):
            return ci_from_coef_se(row[xcol], row["std_err"])
        return (None, None)

    df[["low", "high"]] = df.apply(lambda r: pd.Series(low_high(r)), axis=1)

    # figure
    variables = df["variable"].unique()
    fig_h = max(2.6, len(variables) * 1.1)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    palette = {"FE": "#2b8cbe", "OLS": "#f03b20", "ElasticNet": "#7b3294"}

    y = 0
    y_positions = []
    y_labels = []

    for var in variables:
        sub = df[df["variable"] == var].copy()
        sub["model_rank"] = sub["model"].map({"FE": 0, "OLS": 1, "ElasticNet": 2})
        sub = sub.sort_values("model_rank")

        for _, r in sub.iterrows():
            if pd.isna(r[xcol]):
                y -= 1
                continue

            val = r[xcol]
            low = r["low"]
            high = r["high"]
            col = palette.get(r["model"], "#444")

            if low is None or high is None:
                ax.plot([val], [y], "o", color=col)
            else:
                ax.hlines(y, low, high, color=col, linewidth=2)
                ax.plot([val], [y], "o", color=col)

            y_positions.append(y)
            y_labels.append(f"{var} — {r['model']}")
            y -= 1

        y -= 0.3

    ax.axvline(0, linestyle="--", color="0.6")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    caption = (
        "Note: FE = within-country fixed effects. ElasticNet = penalized regression. "
        "Standardized effects are in SD units of the target."
    )
    fig.text(0.01, 0.02, caption, fontsize=9)

    out_png = OUT_FILES / f"{out_prefix.name}.png"
    out_pdf = OUT_FILES / f"{out_prefix.name}.pdf"
    out_svg = OUT_FILES / f"{out_prefix.name}.svg"

    fig.savefig(out_png, dpi=dpi_png)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)

    return out_png

# ----------------------------------------------------------------------------
# Meta
# ----------------------------------------------------------------------------
def make_meta(produced_files: List[str], args, features_file=None):
    meta = {
        "script": str(Path(__file__).resolve()),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "args": vars(args),
        "produced_files": produced_files,
        "features_file": features_file,
    }
    if features_file:
        p = Path(features_file)
        if p.exists():
            meta["features_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return meta

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vars", nargs="+",
                        default=["trade_exposure", "gov_index_zmean", "inflation_consumer_prices_pct"])
    parser.add_argument("--outprefix", default="plot_comparative_model_effects")
    parser.add_argument("--features", default=None)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    LOG.info("Starting comparative model-effects plot. Vars=%s", args.vars)

    snap = load_snapshot()
    mtable = load_model_table()

    df = build_plot_frame(args.vars, snap, mtable)
    if df.empty:
        LOG.error("No data found — check snapshot or variables.")
        return

    out_png = plot_polished(df, OUT_FILES / args.outprefix, dpi_png=args.dpi)

    produced = [
        str(OUT_FILES / f"{args.outprefix}.png"),
        str(OUT_FILES / f"{args.outprefix}.pdf"),
        str(OUT_FILES / f"{args.outprefix}.svg"),
    ]
    meta = make_meta(produced, args, features_file=args.features)
    meta_path = OUT_BASE / f"{args.outprefix}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")

    print("Saved plot:", out_png)
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()
