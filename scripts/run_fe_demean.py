"""
Run within-entity fixed-effects (demeaned) diagnostics.

Produces:
 - outputs/fe_within/ (figures, tables, and saved artifacts)

Design notes:
 - Computes within-entity transformations for fixed-effects inspection.
 - Supports optional alignment with the baseline fixed-effects specification.
 - Records outputs in a grouped directory for reproducibility.
"""

from pathlib import Path
import argparse, json, yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timezone

DEFAULT_FEATURES = Path("data/processed/features_lean_imputed.csv")
CONFIG_PATH = Path("config/model.yml")

# NEW: unified output folder
OUTDIR = Path("outputs/fe_within")
OUTDIR.mkdir(parents=True, exist_ok=True)

VARS_DEFAULT = ["trade_exposure", "gov_index_zmean", "inflation_consumer_prices_pct"]
GROUP_COL = "iso3"
TARGET = "gdp_growth_pct"


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")

def run_within_demean(df: pd.DataFrame, group: str, target: str, var: str):
    cols = [group, target, var]
    sub = df[cols].dropna().copy()
    if sub.empty:
        raise ValueError(f"No data for {var} after dropna (cols={cols}).")

    # demean
    sub["y_w"] = sub[target] - sub.groupby(group)[target].transform("mean")
    sub["x_w"] = sub[var] - sub.groupby(group)[var].transform("mean")

    X = sm.add_constant(sub["x_w"], has_constant="add")
    res = sm.OLS(sub["y_w"], X).fit()

    sd_target = float(df[target].dropna().std(ddof=0))
    sd_var = float(df[var].dropna().std(ddof=0))
    std_effect = float(res.params.iloc[1]) * sd_var / sd_target

    return {
        "var": var,
        "n_obs": int(len(sub)),
        "coef": float(res.params.iloc[1]),
        "pvalue": float(res.pvalues.iloc[1]),
        "std_err": float(res.bse.iloc[1]) if hasattr(res, "bse") else None,
        "standardized_effect": std_effect,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": "WithinDemean_OLS"
    }


def load_baseline_predictors_from_config(cfg_path: Path):
    if not cfg_path.exists():
        return None
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf8"))
        baseline = (
            cfg.get("predictors", {}).get("baseline") or []
        ) + (
            cfg.get("predictors", {}).get("extra_controls") or []
        )
        out = []
        for b in baseline:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                for k in ("predictors", "features", "name", "term", "predictor"):
                    if k in b and isinstance(b[k], str):
                        out.append(b[k])
                        break
        return out
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vars", nargs="+", default=VARS_DEFAULT)
    p.add_argument("--features", default=str(DEFAULT_FEATURES))
    p.add_argument("--align-with-fe", action="store_true")
    p.add_argument("--no-align", action="store_true")
    args = p.parse_args()

    df_full = pd.read_csv(args.features, low_memory=False)

    if args.align_with_fe and not args.no_align:
        baseline = load_baseline_predictors_from_config(CONFIG_PATH)
        if baseline:
            print("[INFO] Aligning sample with FE baseline predictors from config/model.yml")
        else:
            print("[WARN] Could not load baseline predictors; using minimal mode.")
    else:
        baseline = None

    results = {}
    for var in args.vars:
        try:
            if baseline:
                req_cols = list({GROUP_COL, TARGET} | set(baseline) | {var})
                sub_df = df_full[req_cols].dropna().copy()

                if sub_df.empty:
                    raise ValueError(f"No rows after aligning with baseline predictors for var={var}")

                sd_target = float(sub_df[TARGET].std(ddof=0))
                sd_var = float(sub_df[var].std(ddof=0))

                sub_df["y_w"] = sub_df[TARGET] - sub_df.groupby(GROUP_COL)[TARGET].transform("mean")
                sub_df["x_w"] = sub_df[var] - sub_df.groupby(GROUP_COL)[var].transform("mean")

                X = sm.add_constant(sub_df["x_w"], has_constant="add")
                res = sm.OLS(sub_df["y_w"], X).fit()

                coef = float(res.params.iloc[1])
                pval = float(res.pvalues.iloc[1])
                std_err = float(res.bse.iloc[1])
                std_effect = coef * sd_var / sd_target

                out = {
                    "var": var,
                    "n_obs": int(len(sub_df)),
                    "coef": coef,
                    "pvalue": pval,
                    "std_err": std_err,
                    "standardized_effect": std_effect,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_type": "WithinDemean_Aligned"
                }

            else:
                out = run_within_demean(df_full, GROUP_COL, TARGET, var)

            results[var] = {"status": "ok", "result": out}
            write_json(OUTDIR / f"fe_within_{var}.json", out)
            print(f"[OK] {var}: coef={out['coef']:.6g} std_effect={out['standardized_effect']:.6g} n={out['n_obs']}")

        except Exception as e:
            results[var] = {"status": "error", "error": str(e)}
            print(f"[ERROR] {var}: {e}")

    write_json(OUTDIR / "fe_within_summary.json", results)
    print("Wrote outputs to:", OUTDIR)

if __name__ == "__main__":
    main()
