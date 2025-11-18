# scripts/extract_fe_trade_coef.py
import joblib, pickle, json
from pathlib import Path
import numpy as np

ROOT = Path('.')
artifacts = ROOT / 'artifacts'
# adjust this if different
fe_candidates = list(artifacts.glob("fe_result.*")) + list(artifacts.glob("fe_model.*"))
if not fe_candidates:
    raise SystemExit("No FE artifact found in artifacts/ (fe_result.* / fe_model.*)")

# sds computed earlier
sd_trade = 0.5925754369108269
sd_target = 5.5232021351783205

for p in fe_candidates:
    try:
        try:
            obj = joblib.load(p)
        except Exception:
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
        print("Loaded FE artifact:", p)
        # Try to locate params and model exog names
        params = None
        if hasattr(obj, "params"):
            params = obj.params
        elif hasattr(obj, "fitted") and hasattr(obj.fitted, "params"):
            params = obj.fitted.params

        # show available model naming helpers
        model_obj = getattr(obj, "model", getattr(obj, "fitted", None) and getattr(obj.fitted, "model", None))
        print("model_obj type:", type(model_obj))
        exog_names = None
        if model_obj is not None:
            if hasattr(model_obj, "exog_names"):
                exog_names = list(model_obj.exog_names)
                print("model.exog_names (len):", len(exog_names))
                print("first 40 exog_names:", exog_names[:40])
            if hasattr(model_obj, "data") and hasattr(model_obj.data, "param_names"):
                print("model.data.param_names (first 40):", list(model_obj.data.param_names)[:40])

        # If params is present, list first 40 param index labels
        if params is not None:
            try:
                p_index = list(params.index)
            except Exception:
                p_index = None
            print("params index length:", None if p_index is None else len(p_index))
            if p_index:
                print("first 40 params index:", p_index[:40])

            # Based on earlier debug, trade_exposure was the 2nd data column (after gov_index_zmean)
            # so mapping is: const -> 'const', first data col -> 'x1', second data col -> 'x2'
            # Find candidate x names and print their coef
            candidates = []
            if p_index:
                # find any param name equal to 'x1','x2','x3',...
                candidates = [name for name in p_index if isinstance(name, str) and name.lower().startswith('x')]
                print("Found x-params sample (first 40):", candidates[:40])
            # We'll assume trade_exposure mapped to x2 (because DEBUG FE DESIGN COLS showed trade_exposure as 2nd data column)
            target_x = None
            # Heuristic: if 'x2' in params use it, else try to map by position
            if p_index and 'x2' in p_index:
                target_x = 'x2'
            elif p_index:
                # find the index of trade in the earlier saved design mapping (we don't have it here)
                # fallback: use the second element after 'const' if exists
                if len(p_index) >= 2:
                    # find the name of second param index (skip const if present)
                    if p_index[0] == 'const' and len(p_index) >= 2:
                        target_x = p_index[1]
                    elif len(p_index) >= 2:
                        target_x = p_index[1]
            if target_x is None:
                print("Could not heuristically determine the FE param name for trade_exposure from params index.")
            else:
                print("Assuming FE param name for trade_exposure is:", target_x)
                try:
                    b = float(params[target_x])
                except Exception:
                    b = None
                se = None
                try:
                    se = float(getattr(params, 'bse', None))  # not typical; will likely fail
                except Exception:
                    pass
                # some RegressionResultsWrapper stores .bse separately
                try:
                    bse_attr = getattr(obj, "bse", None) or getattr(getattr(obj, "fitted", None), "bse", None)
                    if bse_attr is not None and target_x in list(getattr(obj, "params").index):
                        se = float(bse_attr[target_x])
                except Exception:
                    pass

                print(" Raw FE coef (assumed):", b, " std_err (if found):", se)
                if b is not None:
                    std_eff = b * sd_trade / sd_target
                    print(" Standardized FE effect (b * sd_trade / sd_target) = {:.6f}".format(std_eff))
                else:
                    print(" Could not read raw FE coefficient value; please inspect params index above.")
        else:
            print("No params found on loaded object.")
    except Exception as e:
        print("Failed to load/inspect", p, ":", repr(e))
