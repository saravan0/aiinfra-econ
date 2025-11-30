# scripts/rescale_en_coefs.py
import joblib, json
from pathlib import Path
import numpy as np
p = Path("artifacts/en_model.joblib")
pipe = joblib.load(p)
# assumes pipeline = [StandardScaler(), ElasticNetCV(...) ]
scaler = pipe.named_steps.get("standardscaler") or pipe.named_steps.get("standardscaler".lower())
enet = pipe.named_steps[list(pipe.named_steps.keys())[-1]]
scales = getattr(scaler, "scale_", None)
feat_names = None
fn = Path("artifacts/feature_names.json")
if fn.exists():
    feat_names = json.load(fn)
coefs_scaled = np.asarray(getattr(enet, "coef_", None))
# if scaler.scale_ is present and matches length:
if scales is not None and coefs_scaled is not None and feat_names:
    original_coefs = coefs_scaled / scales
    # write out mapping:
    out = {name: float(c) for name, c in zip(feat_names, original_coefs)}
    Path("outputs/elasticnet_unscaled_coefs.json").write_text(json.dumps(out, indent=2))
    print("Wrote outputs/elasticnet_unscaled_coefs.json")
else:
    print("Missing scaler scale_ or coef_ or feature_names.json; cannot rescale.")
