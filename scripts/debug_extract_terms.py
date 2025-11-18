# save as scripts/debug_extract_terms.py and run:
# python scripts/debug_extract_terms.py

import json, joblib, pickle, sys
from pathlib import Path
import pandas as pd
import numpy as np
from pprint import pprint

ROOT = Path(".")
reports = ROOT / "reports"
artifacts = ROOT / "artifacts"

print("1) Show model_table rows containing 'trade' (case-insensitive):")
mt = reports / "model_table.csv"
if mt.exists():
    try:
        df = pd.read_csv(mt)
        # show columns first
        print(" model_table columns:", list(df.columns)[:20])
        trade_rows = df[df['term'].str.contains("trade", case=False, na=False)]
        if trade_rows.empty:
            print(" NO rows in model_table.csv mention 'trade' in 'term' column.")
        else:
            print(" Rows matching 'trade' in model_table.csv:")
            print(trade_rows.to_string(index=False))
    except Exception as e:
        print(" ERROR reading model_table.csv:", repr(e))
else:
    print(" model_table.csv not found at", mt.resolve())

print("\n2) Inspect FE fitted param names (artifact search):")
fe_candidates = list(artifacts.glob("fe_result.*")) + list(artifacts.glob("fe_model.*"))
if not fe_candidates:
    print(" No FE artifact found in artifacts/ (looked for fe_result.* / fe_model.*)")
else:
    for p in fe_candidates:
        print(" Trying to load FE artifact:", p)
        try:
            obj = None
            try:
                obj = joblib.load(p)
            except Exception:
                with open(p, "rb") as fh:
                    obj = pickle.load(fh)
            print("  Loaded. Type:", type(obj))
            # Try to access params
            params = None
            if hasattr(obj, "params"):
                params = obj.params
            elif hasattr(obj, "fitted") and hasattr(obj.fitted, "params"):
                params = obj.fitted.params
            # if params is pandas Series/Index
            if params is not None:
                try:
                    names = list(params.index)
                except Exception:
                    names = list(params.keys()) if isinstance(params, dict) else []
                print("  Number of params:", len(names))
                print("  First 30 param names:", names[:30])
                # show any param that contains 'trade' substring
                trade_like = [n for n in names if "trade" in str(n).lower()]
                print("  Params matching 'trade':", trade_like)
            else:
                print("  Could not find .params on object; repr(obj) preview:")
                print(repr(obj)[:500])
        except Exception as e:
            print("  Failed to load or inspect FE artifact:", repr(e))

print("\n3) Inspect ElasticNet artifact and feature names:")
en_candidates = list(artifacts.glob("en_model.*")) + list(artifacts.glob("elasticnet_cv.*")) + list(artifacts.glob("elasticnet_model.*"))
if not en_candidates:
    print(" No ElasticNet artifact found in artifacts/ for expected names (en_model.*, elasticnet_cv.*).")
else:
    for p in en_candidates:
        print(" Trying to load ElasticNet artifact:", p)
        try:
            en = None
            try:
                en = joblib.load(p)
            except Exception:
                with open(p, "rb") as fh:
                    en = pickle.load(fh)
            print("  Loaded. Type:", type(en))
            # If pipeline, show steps
            if hasattr(en, "named_steps"):
                print("  Pipeline steps:", list(en.named_steps.keys()))
                # attempt to retrieve final estimator
                last = list(en.named_steps.items())[-1][1]
                print("  Final estimator type:", type(last))
                if hasattr(last, "coef_"):
                    coef = getattr(last, "coef_")
                    print("  coef_ shape/type:", np.asarray(coef).shape, type(coef))
                if hasattr(last, "feature_names_in_"):
                    print("  feature_names_in_ present (len):", len(getattr(last, "feature_names_in_")))
            else:
                # if it's a raw estimator
                if hasattr(en, "coef_"):
                    print("  estimator has coef_ shape:", np.asarray(en.coef_).shape)
                if hasattr(en, "feature_names_in_"):
                    print("  estimator has feature_names_in_ length:", len(getattr(en, "feature_names_in_")))
            # stop after first successful load
            break
        except Exception as e:
            print("  Failed to load/inspect en artifact:", repr(e))

# show artifacts/feature_names.json if present
fn = artifacts / "feature_names.json"
print("\n4) artifacts/feature_names.json:")
if fn.exists():
    try:
        with open(fn, "r", encoding="utf8") as fh:
            data = json.load(fh)
        print(" feature_names.json length:", len(data))
        print(" first 40 feature names:", data[:40])
        # show any feature that contains 'trade'
        trade_feats = [f for f in data if "trade" in f.lower()]
        print(" features containing 'trade':", trade_feats[:20])
    except Exception as e:
        print(" Failed to read/parse feature_names.json:", repr(e))
else:
    print(" feature_names.json not found at", fn.resolve())

print("\n--- END DEBUG ---")
