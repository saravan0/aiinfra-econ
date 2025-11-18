#!/usr/bin/env python3
# scripts/print_fe_params_and_trade_effect.py
import joblib, pickle
from pathlib import Path
import numpy as np
import json

ROOT = Path(".")
artifacts = ROOT / "artifacts"

sd_trade = 0.5925754369108269
sd_target = 5.5232021351783205

fe_candidates = list(artifacts.glob("fe_result.*")) + list(artifacts.glob("fe_model.*"))
if not fe_candidates:
    raise SystemExit("No FE artifact found in artifacts/ (fe_result.* / fe_model.*)")

for p in fe_candidates:
    print("Loading FE artifact:", p)
    try:
        try:
            obj = joblib.load(p)
        except Exception:
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
    except Exception as e:
        print(" FAILED to load:", e)
        continue

    # Try get model obj and exog names
    model_obj = getattr(obj, "model", getattr(getattr(obj, "fitted", None), "model", None))
    exog_names = None
    if model_obj is not None:
        if hasattr(model_obj, "exog_names"):
            exog_names = list(model_obj.exog_names)
        elif hasattr(model_obj, "data") and hasattr(model_obj.data, "param_names"):
            exog_names = list(model_obj.data.param_names)

    print(" model_obj type:", type(model_obj))
    print(" exog_names (len):", None if exog_names is None else len(exog_names))
    if exog_names is not None:
        print(" exog_names sample (first 80):", exog_names[:80])

    # Try params
    params = None
    try:
        params = getattr(obj, "params", None)
    except Exception:
        params = None

    # Some wrappers put params under .fitted
    if params is None and hasattr(obj, "fitted"):
        try:
            params = getattr(obj.fitted, "params", None)
        except Exception:
            params = None

    print(" params type:", type(params))
    # If params is pandas Series-like with index:
    try:
        import pandas as pd
        if hasattr(params, "index"):
            print(" params appears pandas-like with index length:", len(params.index))
            # show first 60 names
            print(" params.index sample:", list(params.index)[:80])
            # show first 60 values
            print(" params.values sample:", list(params.values)[:80])
            # print pairs for first 80
            pairs = list(zip(list(params.index)[:120], list(params.values)[:120]))
            print(" first param pairs (name, value):")
            for name, val in pairs:
                print(f"  {name}  ->  {val}")
            # try to find x2 or trade directly
            target_name = None
            if 'x2' in params.index:
                target_name = 'x2'
            else:
                # fallback: assume second param after 'const'
                idx = None
                if 'const' in params.index:
                    try:
                        idx = list(params.index).index('const') + 1
                        target_name = list(params.index)[idx] if idx < len(params.index) else None
                    except Exception:
                        target_name = None
            if target_name:
                b_val = params[target_name]
                print("\nAssumed FE param name for trade_exposure:", target_name, " -> coef:", b_val)
                try:
                    b_float = float(b_val)
                    std_eff = b_float * sd_trade / sd_target
                    print(" Standardized FE effect (b * sd_trade / sd_target) = {:.6f}".format(std_eff))
                except Exception:
                    pass
            else:
                print(" Could not auto-locate trade param by name in pandas index.")
            continue
    except Exception:
        pass

    # If params is ndarray-like or list-like
    try:
        arr = None
        if isinstance(params, (list, tuple, np.ndarray)):
            arr = np.asarray(params)
        # other possibility: obj.params is a numpy ndarray stored differently
        if arr is None:
            # try to access params via obj.fitted.params
            if hasattr(obj, "fitted") and hasattr(obj.fitted, "params"):
                try:
                    arr = np.asarray(obj.fitted.params)
                except Exception:
                    arr = None
        if arr is None and hasattr(obj, "params"):
            try:
                arr = np.asarray(obj.params)
            except Exception:
                arr = None

        if arr is not None:
            print(" params array shape:", arr.shape)
            # show first 80 values
            print(" params array sample (first 120):", arr[:120].tolist())
            # If we have exog_names and length matches, pair by position
            if exog_names is not None and len(exog_names) == len(arr):
                print(" Pairing exog_names -> params by position (first 120):")
                for name, val in zip(exog_names[:120], arr[:120]):
                    print(f"  {name}  ->  {val}")
                # Based on earlier debug, trade_exposure was 2nd data column -> exog_names likely ['const','x1','x2',...]
                # So trade_exposure maps to exog_name 'x2' (index 2). Find its position:
                # We will look for 'x2' in exog_names, else assume index 2 after const.
                target_exog = None
                if 'x2' in exog_names:
                    idx = exog_names.index('x2')
                    target_exog = ('x2', idx)
                else:
                    # try to find const position and take next
                    if 'const' in exog_names:
                        const_pos = exog_names.index('const')
                        if const_pos + 1 < len(exog_names):
                            target_exog = (exog_names[const_pos + 1], const_pos + 1)
                if target_exog:
                    name, pos = target_exog
                    b = float(arr[pos])
                    print("\nAssuming FE exog name for trade_exposure is:", name, " at position", pos, " -> coef:", b)
                    std_eff = b * sd_trade / sd_target
                    print(" Standardized FE effect (b * sd_trade / sd_target) = {:.6f}".format(std_eff))
                else:
                    print(" Could not determine the array position for trade_exposure.")
            else:
                # If exog_names length mismatches, try to use known mapping: trade was second data column -> x2 -> arr[2] if const at 0
                # Heuristic:
                if arr.shape[0] > 2:
                    b = float(arr[2])
                    print("Heuristic: taking arr[2] as trade_exposure coef ->", b)
                    print(" Standardized FE effect (b * sd_trade / sd_target) = {:.6f}".format(b * sd_trade / sd_target))
                else:
                    print(" params array too small to heuristically pick element 2.")
            continue
    except Exception as e:
        print(" Failed to interpret params array:", e)

    # Last resort: attempt to access obj.params as dict-like
    try:
        if hasattr(params, "items"):
            print(" params appears dict-like; showing items (first 120):")
            i = 0
            for k, v in params.items():
                print(" ", k, "->", v)
                i += 1
                if i > 119:
                    break
    except Exception:
        pass

    print(" Done with artifact:", p)
