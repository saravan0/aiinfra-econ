#!/usr/bin/env python3
"""
scripts/plot_coef_forest_nature.py

Publication-quality Nature-style coefficient forest plot (phd polish).

Outputs:
 - reports/figs/coef_forest_nature.png/pdf/svg

Behavior highlights:
 - dynamic figure sizing by rows
 - legend placed outside plot (no clipping)
 - horizontal CI lines centered on point estimate
 - missing CI -> dot only
 - robust snapshot parsing (list or dict)
"""
from pathlib import Path
import json
import argparse
import math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 12})

ROOT = Path(".")
SNAP_PATH = ROOT / "reports" / "stage1_snapshot.json"
MODEL_TABLE = ROOT / "reports" / "model_comparison_table.csv"
OUTDIR = ROOT / "reports" / "figs"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_snapshot() -> Any:
    if not SNAP_PATH.exists():
        raise FileNotFoundError(f"Missing snapshot: {SNAP_PATH}")
    raw = json.loads(SNAP_PATH.read_text(encoding="utf8"))
    return raw


def load_model_table() -> pd.DataFrame:
    if MODEL_TABLE.exists():
        return pd.read_csv(MODEL_TABLE)
    return pd.DataFrame(columns=["variable", "fe_coef", "fe_std_effect", "ols_coef", "ols_std_effect", "en_coef", "en_std_effect"])


def _find_snapshot_entry(snapshot: Any, var: str) -> Optional[Dict[str, Any]]:
    """Return snapshot element for var. Snapshot can be dict or list-of-dicts."""
    if isinstance(snapshot, dict):
        return snapshot.get(var)
    if isinstance(snapshot, list):
        for el in snapshot:
            if not isinstance(el, dict):
                continue
            # normalized keys: 'variable' or 'var'
            if el.get("variable") == var or el.get("var") == var:
                return el
            # fallback: find in nested standardized_summary
            if "standardized_summary" in el and isinstance(el["standardized_summary"], dict):
                if el["standardized_summary"].get("var") == var:
                    return el
    return None


def extract_models_from_entry(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return mapping {model_name: {coef, std_err, ci, standardized, n}}"""
    out: Dict[str, Dict[str, Any]] = {}
    # many snapshot formats — handle robustly
    # primary candidate: entry["standardized_summary"]["summary"]
    if not isinstance(entry, dict):
        return out

    # try normalized place first
    ss = entry.get("standardized_summary") or entry.get("standardized_effects") or entry.get("summary")
    if isinstance(ss, dict):
        # standardized_summary.summary often contains FE/OLS/ElasticNet dicts
        inner = ss.get("summary") if "summary" in ss else ss
        if isinstance(inner, dict):
            for mname, mv in inner.items():
                if not isinstance(mv, dict):
                    continue
                out[mname] = {
                    "coef": mv.get("coef"),
                    "std_err": mv.get("std_err") or mv.get("se") or mv.get("bse"),
                    "ci": mv.get("conf_int") or mv.get("ci"),
                    "standardized": mv.get("standardized") or mv.get("std_effect"),
                    "n": mv.get("n_obs") or mv.get("n"),
                }
            return out

    # fallback: direct keys fe_json / ols_json / elasticnet_json
    for k in ("fe_json", "ols_json", "elasticnet_json", "en_json"):
        if k in entry and isinstance(entry[k], dict):
            mn = k.replace("_json", "").upper()
            mv = entry[k]
            out[mn] = {
                "coef": mv.get("coef"),
                "std_err": mv.get("std_err"),
                "ci": mv.get("conf_int") or mv.get("ci"),
                "standardized": mv.get("standardized_effect"),
                "n": mv.get("n_obs") or mv.get("n"),
            }
    # last resort: if entry itself looks like per-model dict
    for cand in ("FE", "OLS", "ElasticNet", "ElasticNetCV", "WithinDemean_Aligned"):
        if cand in entry and isinstance(entry[cand], dict):
            mv = entry[cand]
            out[cand] = {
                "coef": mv.get("coef"),
                "std_err": mv.get("std_err"),
                "ci": mv.get("conf_int"),
                "standardized": mv.get("standardized"),
                "n": mv.get("n_obs") or mv.get("n"),
            }
    return out


def compute_ci_from_coef_se(coef: float, se: float, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    try:
        c = float(coef)
        s = float(se)
        return (c - z * s, c + z * s)
    except Exception:
        return (None, None)


def build_plot_table(vars_list: List[str], snapshot: Any, mtable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in vars_list:
        entry = _find_snapshot_entry(snapshot, var)
        if entry:
            per_model = extract_models_from_entry(entry)
            # prefer FE/OLS/ElasticNet keys (normalize names)
            for mn in ("FE", "OLS", "ElasticNet", "ElasticNetCV", "EN", "WithinDemean_Aligned"):
                if mn in per_model:
                    mv = per_model[mn]
                    rows.append({
                        "variable": var,
                        "model": "ElasticNet" if mn.lower().startswith("elastic") else ("FE" if "FE" in mn or "within" in mn.lower() else "OLS"),
                        "standardized": mv.get("standardized"),
                        "coef": mv.get("coef"),
                        "se": mv.get("std_err"),
                        "ci": mv.get("ci"),
                        "n": mv.get("n"),
                    })
        # fallback to model table (raw)
        if var and var in mtable.get("variable", mtable.get("term", pd.Series())).astype(str).values:
            # model_comparison_table uses columns variable, fe_coef, ols_coef, en_coef
            subset = mtable[mtable["variable"].astype(str).str.contains(var, case=False, na=False)]
            for _, r in subset.iterrows():
                rows.append({
                    "variable": var,
                    "model": "FE",
                    "standardized": r.get("fe_std_effect"),
                    "coef": r.get("fe_coef"),
                    "se": None,
                    "ci": None,
                    "n": None,
                })
                rows.append({
                    "variable": var,
                    "model": "OLS",
                    "standardized": r.get("ols_std_effect"),
                    "coef": r.get("ols_coef"),
                    "se": None,
                    "ci": None,
                    "n": None,
                })
                rows.append({
                    "variable": var,
                    "model": "ElasticNet",
                    "standardized": r.get("en_std_effect"),
                    "coef": r.get("en_coef"),
                    "se": None,
                    "ci": None,
                    "n": None,
                })
    df = pd.DataFrame(rows)
    # normalize model names
    df["model"] = df["model"].replace({None: "OLS"})
    return df


def plot_nature_forest(df: pd.DataFrame, out_prefix: Path) -> Path:
    # reorder and clean
    model_order = ["FE", "OLS", "ElasticNet"]
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values(["variable", "model"], ascending=[True, True]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("No rows to plot.")

    # Prepare rows: group by variable; each model is a row
    plot_rows = []
    for _, row in df.iterrows():
        coef = row["coef"] if pd.notna(row["coef"]) else None
        se = row["se"] if pd.notna(row["se"]) else None
        ci = row["ci"]
        low, high = (None, None)
        if ci and isinstance(ci, (list, tuple)) and len(ci) == 2:
            low, high = float(ci[0]), float(ci[1])
        elif coef is not None and se is not None:
            low, high = compute_ci_from_coef_se(coef, se)
        plot_rows.append({
            "variable": row["variable"],
            "model": row["model"],
            "coef": coef,
            "se": se,
            "low": low,
            "high": high,
            "std": row.get("standardized"),
            "n": row.get("n")
        })

    pdf = pd.DataFrame(plot_rows)

    # choose x values: prefer standardized if present (consistent scale across vars)
    use_std = pdf["std"].notna().any()
    xcol = "std" if use_std else "coef"
    title = "Standardized effects (σ of target) by model" if use_std else "Coefficients by model"

    # figure sizing: height ~ rows * 0.55 + margins; width generous to accommodate legend on right
    nrows = len(pdf)
    fig_h = max(4, nrows * 0.45 + 1.5)
    fig_w = 12
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # y positions
    labels = []
    y_positions = []
    y = (nrows - 1)  # top to bottom
    spacing = 1.0
    marker_map = {"FE": {"marker": "o", "color": "#1f77b4"}, "OLS": {"marker": "o", "color": "#ff5733"}, "ElasticNet": {"marker": "o", "color": "#6a3d9a"}}

    # plot rows grouped by variable but keep ordering same as pdf
    for idx, r in pdf.iterrows():
        val = r.get(xcol)
        low = r.get("low")
        high = r.get("high")
        model = r.get("model")
        # if value missing, skip (can't plot)
        if val is None:
            y_positions.append(y)
            labels.append(f"{r['variable']} — {model}")
            y -= spacing
            continue

        # CI calculation; ensure low/high numeric
        err_low = 0.0
        err_high = 0.0
        if low is not None and high is not None:
            # if low > high (safety), swap
            if low > high:
                low, high = high, low
            err_low = float(val) - float(low)
            err_high = float(high) - float(val)
            # enforce non-negative (numerical jitter can make tiny negative numbers)
            err_low = max(err_low, 0.0)
            err_high = max(err_high, 0.0)
        else:
            err_low = err_high = None  # so we can draw dot-only

        color = marker_map.get(model, {}).get("color", "#333333")
        marker = marker_map.get(model, {}).get("marker", "o")

        # draw horizontal CI line if present
        if err_low is not None and err_high is not None and (err_low > 0 or err_high > 0):
            # draw a solid horizontal line (thicker for visibility)
            ax.hlines(y, xmin=val - err_low, xmax=val + err_high, colors=color, linewidth=3, alpha=0.85)
            # endpoint triangles for truncated CI (if beyond axis later)
            ax.plot([val - err_low], [y], marker="<", color=color, markersize=6, clip_on=False)
            ax.plot([val + err_high], [y], marker=">", color=color, markersize=6, clip_on=False)

        # draw point estimate marker centered on the CI
        ax.plot([val], [y], marker=marker, color="white", markeredgecolor=color, markeredgewidth=1.6, markersize=8, zorder=5)
        ax.plot([val], [y], marker=".", color=color, markersize=6, zorder=6)

        labels.append(f"{r['variable']}  ")
        y_positions.append(y)
        y -= spacing

    # aesthetics
    ax.axvline(0, color="0.6", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    # show only variable names on left: we put larger, bold group labels
    # create left yticklabels with variable names grouped: for readability use every third row (since each var has ~3 models)
    # We'll replace ticks with empty space and draw group labels manually on the left
    ax.set_yticklabels(["" for _ in y_positions])
    # compute mid-y for each variable group and write bold label
    grouped = pdf.groupby("variable")
    for var, group in grouped:
        ys = []
        for _, rr in group.iterrows():
            # find the index in pdf to get y position
            # since we used sequential y_positions, we can find them by index order
            pass
    # place variable labels manually aligned to left
    # find unique variables in original order
    unique_vars = list(df["variable"].unique())
    # compute group mid positions by scanning y_positions in blocks of len(model_order) (approx)
    # simpler: compute mid positions by finding all y_positions where label startswith var
    for var in unique_vars:
        inds = [i for i, lab in enumerate(labels) if lab.strip().startswith(var)]
        if not inds:
            continue
        ys_for_var = [y_positions[i] for i in inds]
        mid = sum(ys_for_var) / len(ys_for_var)
        ax.text(-0.03, mid, var, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=12, fontweight="bold")

    # x label and title
    ax.set_xlabel("Standardized effect (σ of target)" if use_std else "Coefficient (raw units)")
    ax.set_title(title, fontsize=16)

    # set x-limits with some padding
    xvals = pdf[xcol].dropna().astype(float).tolist()
    # include CI endpoints
    for lo, hi in zip(pdf["low"], pdf["high"]):
        if lo is not None and hi is not None:
            xvals.extend([float(lo), float(hi)])
    if xvals:
        xmin = min(xvals)
        xmax = max(xvals)
        pad = max(0.1 * (xmax - xmin) if xmax > xmin else 0.6, 0.1)
        ax.set_xlim(xmin - pad, xmax + pad)

    # legend outside
    import matplotlib.lines as mlines
    legend_handles = []
    for m in model_order:
        color = marker_map.get(m, {}).get("color", "#333333")
        handle = mlines.Line2D([], [], color=color, marker="o", linestyle="None", markersize=8, label=m)
        legend_handles.append(handle)
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    # draw note at bottom with enough margin
    note = "Note: FE = within-country fixed-effects; ElasticNet = penalized ML model (coef shrinkage). Standardized effects convert coefficient into SD units of the target for comparability."
    fig.subplots_adjust(bottom=0.15, right=0.78)
    fig.text(0.02, 0.02, note, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 0.95, 1])
    # save
    for ext in ("png", "pdf", "svg"):
        outp = out_prefix.with_suffix("." + ext)
        fig.savefig(outp, bbox_inches="tight")
        print("Wrote", outp)
    plt.close(fig)
    return out_prefix.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vars", nargs="+", default=["trade_exposure", "gov_index_zmean", "inflation_consumer_prices_pct"])
    args = parser.parse_args()
    snapshot = load_snapshot()
    mtable = load_model_table()
    df = build_plot_table(args.vars, snapshot, mtable)
    if df.empty:
        print("No model rows found for requested vars. Exiting.")
        return
    out = plot_nature_forest(df, OUTDIR / "coef_forest_nature")
    print("Saved Nature-style forest at:", out)


if __name__ == "__main__":
    main()
