# scripts/run_model_debug.py
"""
Debug wrapper: import src.model.train, show which file was loaded,
print config, check features file exists, then call main() while
capturing and printing any exception/traceback.
"""
import sys, traceback
from pathlib import Path
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("Python exec:", sys.executable)
print("sys.path[0]:", sys.path[0])

try:
    import importlib
    mod = importlib.import_module("src.model.train")
    print("Imported module:", mod)
    print("Module file:", getattr(mod, "__file__", "N/A"))
except Exception as e:
    print("FAILED to import src.model.train:", e)
    traceback.print_exc()
    raise SystemExit(1)

# show config file contents
cfg_path = Path("config/model.yml")
print("\nUsing config path:", cfg_path.resolve())
if cfg_path.exists():
    try:
        import yaml
        cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf8"))
        import json
        print("Config (preview):")
        print(json.dumps(cfg, indent=2)[:2000])
    except Exception as e:
        print("Could not read config:", e)
else:
    print("Config file not found:", cfg_path)

# check feature file existence
try:
    fpath = Path(cfg["data"]["features_path"])
    print("\nFeatures path from config:", fpath, "exists?", fpath.exists())
    if fpath.exists():
        print("Features size (bytes):", fpath.stat().st_size)
        import pandas as pd
        df = pd.read_csv(fpath, nrows=3)
        print("Features columns:", list(df.columns)[:50])
        print("Sample rows:")
        print(df.head(3).to_string(index=False))
except Exception as e:
    print("Error checking features file:", e)

# call main() safely and catch exceptions
print("\nCalling main() from src.model.train ...")
try:
    # Some modules define main() to accept argv; call with None
    mod.main()
    print("main() returned normally")
except SystemExit as se:
    print("main() called SystemExit:", se)
except Exception as e:
    print("Exception while running main():", e)
    traceback.print_exc()
