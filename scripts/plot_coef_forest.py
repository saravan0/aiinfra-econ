#!/usr/bin/env python3
"""
scripts/plot_coef_forest.py

Robust forest plot for standardized effects (or raw coefs) across FE / OLS / ElasticNet.

This version:
 - Handles stage1_snapshot.json as list or dict (your current snapshot is a list).
 - Handles model_comparison_table.csv with either wide-summary columns or term rows.
 - Defensively avoids negative xerr values and None plotting.
Outputs:
 - reports/figs/coef_forest.png/.pdf/.svg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 300, "savefig.dpi": 300})

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


def _find_in_snapshot_list(snap_list: List[Dict[str, Any]], varname: str) -> Dict[str, Any]:
    for it in snap_list:
        if not isinstance(it, dict):
            continue
        if it.get("variable") == varname or it.get("var") == varname or it.get("name") == varname:
            return it
        if varname in it:
            return it[varname]
    return {}


def extract_from_snapshot(snap: Union[Dict[str, Any], List[Dict[str, Any]]], varname: str) -> Dict[str, Any]:
    """
    Return mapping: { "FE": {...}, "OLS": {...}, "ElasticNet": {...} }
    Each inner dict may contain: standardized, coef, std_err, ci, n
    """
    out: Dict[str, Any] = {}
    v = None

    # snapshot as dict keyed by var
    if isinstance(snap, dict):
        if varname in snap:
            v = snap[varname]
        else:
            for k, val in snap.items():
                if isinstance(k, str) and varname.lower() in k.lower():
                    v = val
                    break

    # snapshot is list -> search for element
    if v is None and isinstance(snap, list):
        v = _find_in_snapshot_list(snap, varname)

    if not v or not isinstance(v, dict):
        return out

    # Candidate locations for model summaries
    candidates = []
    ss = v.get("standardized_summary") or v.get("standardized_effects") or v.get("standardized")
    if isinstance(ss, dict):
        if "summary" in ss and isinstance(ss["summary"], dict):
            candidates.append(ss["summary"])
        else:
            candidates.append(ss)

    for key in ("summary", "per_model", "models", "by_model"):
        if key in v and isinstance(v[key], dict):
            candidates.append(v[key])

    top_model_map = {k: v[k] for k in v.keys() if k.upper() in ("FE", "OLS", "ELASTICNET", "ELASTICNETCV", "EN", "WITHINDEMEAN_ALIGNED") and isinstance(v[k], dict)}
    if top_model_map:
        candidates.append(top_model_map)

    if not candidates:
        possible_keys = {"coef", "std_err", "pvalue", "standardized_effect", "n_obs", "conf_int", "sd_var", "sd_target"}
        if any(k in v for k in possible_keys):
            candidates.append({"FE": v})

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        for mname, mv in cand.items():
            if not isinstance(mv, dict):
                continue
            mnorm = mname
            if mnorm.upper() in ("WITHINDEMEAN_ALIGNED", "FE", "WITHIN", "FIXED_EFFECTS"):
                mnorm = "FE"
            elif mnorm.upper() in ("ELASTICNETCV", "ELASTICNET", "EN"):
                mnorm = "ElasticNet"
            elif mnorm.upper() in ("OLS", "OLS_BASELINE"):
                mnorm = "OLS"

            std = mv.get("standardized_effect") or mv.get("standardized") or mv.get("std") or mv.get("std_effect")
            coef = mv.get("coef") or mv.get("b") or mv.get("estimate")
            se = mv.get("std_err") or mv.get("se")
            ci = mv.get("conf_int") or mv.get("ci")
            nobs = mv.get("n_obs") or mv.get("n")
            out[mnorm] = {"standardized": std, "coef": coef, "std_err": se, "ci": ci, "n": nobs}

    return out


def build_table(vars_list: List[str], snap: Union[Dict[str, Any], List[Dict[str, Any]]], mtable: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for var in vars_list:
        s = extract_from_snapshot(snap, var)
        for model_name in ("FE", "OLS", "ElasticNet"):
            entry = s.get(model_name)
            if entry:
                rows.append(
                    {
                        "variable": var,
                        "model": model_name,
                        "std": entry.get("standardized"),
                        "coef": entry.get("coef"),
                        "se": entry.get("std_err"),
                        "ci": entry.get("ci"),
                        "n": entry.get("n"),
                    }
                )

        if not mtable.empty:
            wide_cols = {"variable", "fe_coef", "fe_std_effect", "ols_coef", "ols_std_effect", "en_coef", "en_std_effect"}
            if wide_cols.issubset(set(mtable.columns)):
                sub = mtable[mtable["variable"].astype(str).str.contains(var, case=False, na=False)]
                for _, r in sub.iterrows():
                    rows.append({"variable": var, "model": "FE", "std": r.get("fe_std_effect"), "coef": r.get("fe_coef"), "se": None, "ci": None, "n": None})
                    rows.append({"variable": var, "model": "OLS", "std": r.get("ols_std_effect"), "coef": r.get("ols_coef"), "se": None, "ci": None, "n": None})
                    rows.append({"variable": var, "model": "ElasticNet", "std": r.get("en_std_effect"), "coef": r.get("en_coef"), "se": None, "ci": None, "n": None})
            else:
                if {"term", "model", "coef"}.issubset(set(mtable.columns)):
                    mask = mtable["term"].astype(str).str.contains(var, case=False, na=False)
                    sub = mtable.loc[mask]
                    for _, r in sub.iterrows():
                        rows.append(
                            {
                                "variable": var,
                                "model": r.get("model", "unknown"),
                                "std": r.get("std") if "std" in r else None,
                                "coef": r.get("coef"),
                                "se": r.get("std_err") if "std_err" in r else r.get("se"),
                                "ci": r.get("ci") if "ci" in r else None,
                                "n": r.get("n_obs") if "n_obs" in r else r.get("n"),
                            }
                        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["model"] = df["model"].replace({"ElasticNetCV": "ElasticNet", "EN": "ElasticNet", "WithinDemean_Aligned": "FE"})
    return df


def compute_ci_from_coef_se(coef, se, z=1.96) -> Tuple[Union[float, None], Union[float, None]]:
    try:
        c = float(coef)
        s = float(se)
        return (c - z * s, c + z * s)
    except Exception:
        return (None, None)


def plot_forest(df: pd.DataFrame, out_prefix: Path) -> Path:
    models_order = ["FE", "OLS", "ElasticNet"]
    df["model"] = pd.Categorical(df["model"], categories=models_order, ordered=True)
    df = df.sort_values(["variable", "model"])
    variables = df["variable"].unique().tolist()
    if not variables:
        raise RuntimeError("No variables to plot.")

    plot_rows: List[Dict[str, Any]] = []
    for v in variables:
        sub = df[df["variable"] == v]
        for _, r in sub.iterrows():
            std = r.get("std")
            coef = r.get("coef")
            se = r.get("se")
            ci = r.get("ci")
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                low, high = ci
            elif (se is not None) and (coef is not None):
                low, high = compute_ci_from_coef_se(coef, se)
            else:
                low, high = (None, None)
            plot_rows.append({"variable": v, "model": r.get("model"), "std": std, "coef": coef, "low": low, "high": high, "n": r.get("n")})

    pdf = pd.DataFrame(plot_rows)
    use_std = not pdf["std"].isna().all()
    xcol = "std" if use_std else "coef"
    title = "Standardized effects (σ of target) by model" if use_std else "Raw coefficients by model"
    fig_h = max(3, len(variables) * 0.9)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    y_positions: List[float] = []
    labels: List[str] = []
    y = 0.0
    spacing = 1.0

    for v in variables:
        grp = pdf[pdf["variable"] == v].sort_values("model")
        for _, r in grp.iterrows():
            val = r.get(xcol)
            if val is None or (isinstance(val, float) and (np.isnan(val))):
                # skip plotting points without value
                y -= spacing
                continue
            low = r.get("low")
            high = r.get("high")

            # ensure numeric and non-negative errorbars
            try:
                if low is not None:
                    low = float(low)
                if high is not None:
                    high = float(high)
                valf = float(val)
            except Exception:
                y -= spacing
                continue

            err_low = max(0.0, valf - low) if (low is not None) else 0.0
            err_high = max(0.0, high - valf) if (high is not None) else 0.0

            # If both err_low and err_high are zero and no CI, plot dot only
            if err_low == 0.0 and err_high == 0.0:
                ax.plot([valf], [y], "o", markersize=6)
            else:
                ax.errorbar([valf], [y], xerr=[[err_low], [err_high]], fmt="o", capsize=4, markersize=6)

            labels.append(f"{v} — {r.get('model')}")
            y_positions.append(y)
            y -= spacing
        y -= spacing * 0.25

    ax.axvline(0, color="0.6", linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Standardized effect (σ) " if use_std else "Coefficient (raw units)")
    ax.set_title(title)
    plt.tight_layout()

    for ext in ("png", "pdf", "svg"):
        outp = out_prefix.with_suffix("." + ext)
        fig.savefig(outp, bbox_inches="tight")
        print("Wrote", outp)
    plt.close(fig)
    return out_prefix.with_suffix(".png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vars", nargs="+", default=["trade_exposure", "gov_index_zmean", "inflation_consumer_prices_pct"])
    args = p.parse_args()

    snap = load_snapshot()
    mtable = load_model_table()
    df = build_table(args.vars, snap, mtable)
    if df.empty:
        print("No model rows found for requested vars. Exiting.")
        return

    df["ci"] = df.apply(
        lambda r: r["ci"]
        if (r["ci"] and isinstance(r["ci"], (list, tuple)) and len(r["ci"]) == 2)
        else (compute_ci_from_coef_se(r["coef"], r["se"]) if (pd.notna(r["coef"]) and pd.notna(r["se"])) else None),
        axis=1,
    )

    outp = plot_forest(df, OUTDIR / "coef_forest")
    print("\nSUMMARY (first rows):")
    print(df.head(40).to_string(index=False))
    print("\nSaved figure at:", outp)


if __name__ == "__main__":
    main()
