#!/usr/bin/env python3
"""
Polished coefficient forest plot for Stage 1 (publication-ready).

Produces:
 - reports/figs/coef_forest_polished.png  (300 dpi raster)
 - reports/figs/coef_forest_polished.pdf  (vector)
 - reports/figs/coef_forest_polished.svg

Run:
    python scripts/plot_coef_forest_polished.py --vars trade_exposure gov_index_zmean inflation_consumer_prices_pct
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- styling ---
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

ROOT = Path(".")
SNAP = ROOT / "reports" / "stage1_snapshot.json"
MODEL_TABLE = ROOT / "reports" / "model_comparison_table.csv"
OUTDIR = ROOT / "reports" / "figs"
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_snapshot() -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if not SNAP.exists():
        raise FileNotFoundError(f"Missing snapshot: {SNAP}")
    return json.loads(SNAP.read_text(encoding="utf8"))

def load_model_table() -> pd.DataFrame:
    if MODEL_TABLE.exists():
        return pd.read_csv(MODEL_TABLE)
    return pd.DataFrame()

# simple helper for mapping snapshot shapes (keeps logic compact)
def _find_in_snapshot_list(snap_list: List[Dict[str,Any]], varname: str):
    for it in snap_list:
        if not isinstance(it, dict):
            continue
        if it.get("variable") == varname or it.get("var") == varname or it.get("name") == varname:
            return it
        # allow for nested form: element contains a key equal to varname
        if varname in it:
            return it[varname]
    return {}

def extract_models_from_snapshot(snap: Union[Dict[str,Any], List[Dict[str,Any]]], varname: str) -> Dict[str,Dict]:
    """
    Return dict with keys FE, OLS, ElasticNet where available.
    Each inner dict: standardized, coef, std_err, ci, n
    """
    out = {}
    v = None
    if isinstance(snap, dict):
        v = snap.get(varname) or next((val for k,val in snap.items() if isinstance(k,str) and varname.lower() in k.lower()), None)
    if v is None and isinstance(snap, list):
        v = _find_in_snapshot_list(snap, varname)
    if not v or not isinstance(v, dict):
        return out

    # candidate places: standardized_summary.summary, per_model, models, or direct mapping
    candidates = []
    ss = v.get("standardized_summary") or v.get("standardized_effects") or v.get("standardized")
    if isinstance(ss, dict):
        if isinstance(ss.get("summary"), dict):
            candidates.append(ss["summary"])
        else:
            candidates.append(ss)
    for key in ("summary","per_model","models","by_model"):
        if key in v and isinstance(v[key], dict):
            candidates.append(v[key])
    # if top-level keys FE/OLS exist
    top_keys = {k:v[k] for k in v.keys() if isinstance(k,str) and k.upper() in ("FE","OLS","ELASTICNET","ELASTICNETCV","EN","WITHINDEMEAN_ALIGNED")}
    if top_keys:
        candidates.append(top_keys)
    # fallback: treat v as model dict
    if not candidates:
        candidates.append(v)

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        for mname, mv in cand.items():
            if not isinstance(mv, dict):
                continue
            name = mname
            if name.upper() in ("WITHINDEMEAN_ALIGNED","WITHIN","FIXED_EFFECTS"):
                name = "FE"
            elif name.upper() in ("ELASTICNETCV","ELASTICNET","EN"):
                name = "ElasticNet"
            elif name.upper() in ("OLS",):
                name = "OLS"
            std = mv.get("standardized_effect") or mv.get("standardized") or mv.get("std_effect") or mv.get("std")
            coef = mv.get("coef") or mv.get("b") or mv.get("estimate")
            se = mv.get("std_err") or mv.get("se")
            ci = mv.get("conf_int") or mv.get("ci")
            n = mv.get("n_obs") or mv.get("n")
            out[name] = {"standardized": std, "coef": coef, "std_err": se, "ci": ci, "n": n}
    return out

def build_plot_frame(vars_list:List[str], snap, mtable:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for var in vars_list:
        m = extract_models_from_snapshot(snap, var)
        for nm in ("FE","OLS","ElasticNet"):
            e = m.get(nm)
            if e:
                rows.append({"variable":var,"model":nm,"std":e.get("standardized"),"coef":e.get("coef"),"se":e.get("std_err"),"ci":e.get("ci"),"n":e.get("n")})
        # fallback: model_table wide format
        if not mtable.empty:
            wide = {"variable","fe_coef","fe_std_effect","ols_coef","ols_std_effect","en_coef","en_std_effect"}
            if wide.issubset(set(mtable.columns)):
                sub = mtable[mtable["variable"].astype(str).str.contains(var,case=False,na=False)]
                for _,r in sub.iterrows():
                    rows.append({"variable":var,"model":"FE","std":r.get("fe_std_effect"),"coef":r.get("fe_coef"),"se":None,"ci":None,"n":None})
                    rows.append({"variable":var,"model":"OLS","std":r.get("ols_std_effect"),"coef":r.get("ols_coef"),"se":None,"ci":None,"n":None})
                    rows.append({"variable":var,"model":"ElasticNet","std":r.get("en_std_effect"),"coef":r.get("en_coef"),"se":None,"ci":None,"n":None})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["model"] = df["model"].replace({"ElasticNetCV":"ElasticNet","EN":"ElasticNet","WithinDemean_Aligned":"FE"})
    return df

def ci_from_coef_se(coef,se,z=1.96):
    try:
        c=float(coef); s=float(se)
        return (c - z*s, c + z*s)
    except Exception:
        return (None,None)

def plot_polished(df:pd.DataFrame, out_prefix:Path):
    # prepare rows
    df = df.copy()
    df = df.sort_values(["variable","model"])
    variables = df["variable"].unique().tolist()
    if len(variables)==0:
        raise RuntimeError("No variables to plot.")
    plot_rows=[]
    for v in variables:
        sub = df[df["variable"]==v]
        for _,r in sub.iterrows():
            std=r.get("std"); coef=r.get("coef"); se=r.get("se"); ci=r.get("ci")
            if isinstance(ci,(list,tuple)) and len(ci)==2:
                low,high = ci
            elif (se is not None) and (coef is not None):
                low,high = ci_from_coef_se(coef,se)
            else:
                low,high = (None,None)
            plot_rows.append({"variable":v,"model":r.get("model"),"std":std,"coef":coef,"low":low,"high":high,"n":r.get("n")})
    pdf = pd.DataFrame(plot_rows)
    use_std = not pdf["std"].isna().all()
    xcol = "std" if use_std else "coef"
    title = "Standardized effects (σ of target) by model" if use_std else "Raw coefficients by model"

    # figure geometry
    nvars = len(variables)
    fig_h = max(3.0, nvars * 1.1)
    fig_w = 9.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    palette = {"FE":"#2b8cbe","OLS":"#f03b20","ElasticNet":"#7b3294"}
    y = 0.0
    spacing = 1.0
    y_positions=[]
    y_labels=[]
    for v in variables:
        grp = pdf[pdf["variable"]==v].sort_values("model")
        # small header line for group
        for _,r in grp.iterrows():
            val = r.get(xcol)
            if val is None or (isinstance(val,float) and np.isnan(val)):
                y -= spacing
                continue
            try:
                valf=float(val)
            except Exception:
                y -= spacing
                continue
            low = r.get("low"); high = r.get("high")
            # numeric cast
            try:
                if low is not None: low=float(low)
                if high is not None: high=float(high)
            except Exception:
                low=None; high=None
            err_low = max(0.0, valf - low) if low is not None else 0.0
            err_high = max(0.0, high - valf) if high is not None else 0.0
            col = palette.get(r.get("model"), "#444444")
            # draw thin CI line with alpha and thicker cap
            if err_low==0.0 and err_high==0.0:
                ax.plot([valf],[y], marker='o', markersize=6, color=col)
            else:
                ax.hlines(y, xmin=(low if low is not None else valf), xmax=(high if high is not None else valf), color=col, alpha=0.8, linewidth=2)
                ax.plot([valf],[y], marker='o', markersize=6, color=col)
            y_positions.append(y)
            y_labels.append(f"{v} — {r.get('model')}")
            y -= spacing
        # slightly larger gap between variables
        y -= spacing * 0.25

    # aesthetics
    ax.axvline(0, color="0.6", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Standardized effect (σ of target)" if use_std else "Coefficient (raw units)")
    ax.set_title(title)
    ax.grid(axis='x', linestyle=':', linewidth=0.6, alpha=0.6)
    # tight layout and caption space
    plt.tight_layout(rect=[0,0.06,1,1])
    caption = ("Note: FE = within-country fixed-effects; ElasticNet = penalized ML model (coef shrinkage). "
               "Standardized effects convert coefficient into SD units of the target for comparability.")
    fig.text(0.01, 0.02, caption, fontsize=9, ha="left", wrap=True)
    for ext in ("png","pdf","svg"):
        outp = out_prefix.with_suffix("."+ext)
        if ext=="png":
            fig.savefig(outp, bbox_inches="tight", dpi=300)
        else:
            fig.savefig(outp, bbox_inches="tight")
        print("Wrote", outp)
    plt.close(fig)
    return out_prefix.with_suffix(".png")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vars", nargs="+", default=["trade_exposure","gov_index_zmean","inflation_consumer_prices_pct"])
    args = p.parse_args()
    snap = load_snapshot()
    mtable = load_model_table()
    df = build_plot_frame(args.vars, snap, mtable)
    if df.empty:
        print("No model rows found for requested vars. Exiting.")
        return
    # compute ci where possible
    df["ci"] = df.apply(lambda r: r["ci"] if (r["ci"] and isinstance(r["ci"], (list,tuple)) and len(r["ci"])==2) else (ci_from_coef_se(r["coef"], r["se"]) if (pd.notna(r["coef"]) and pd.notna(r["se"])) else None), axis=1)
    outp = plot_polished(df, OUTDIR / "coef_forest_polished")
    print("\nSaved polished figure at:", outp)

if __name__ == "__main__":
    main()
