# scripts/run_robustness_debug.py
from pathlib import Path
import sys, traceback

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

print("Python exec:", sys.executable)
print("sys.path[0]:", sys.path[0])

# import module and show file
try:
    import importlib
    mod = importlib.import_module("src.model.robustness")
    print("Imported src.model.robustness ->", getattr(mod, "__file__", "N/A"))
except Exception as e:
    print("FAILED to import src.model.robustness:", e)
    traceback.print_exc()
    raise SystemExit(1)

# print config preview
cfg_path = Path("config/model.yml")
print("\nConfig path:", cfg_path.resolve(), "exists?", cfg_path.exists())
if cfg_path.exists():
    try:
        import yaml, json
        cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf8"))
        s = json.dumps(cfg, indent=2)
        print("Config (preview):")
        print(s[:2000])
    except Exception as e:
        print("Failed to read config:", e)

# check features file
try:
    feat = Path(cfg["data"]["features_path"])
    print("\nFeatures path:", feat.resolve(), "exists?", feat.exists())
    if feat.exists():
        print("Features size (bytes):", feat.stat().st_size)
        import pandas as pd
        df = pd.read_csv(feat, nrows=5)
        print("Features columns:", list(df.columns))
        print("Sample row:")
        print(df.head(2).to_string(index=False))
except Exception as e:
    print("Error inspecting features file:", e)
    traceback.print_exc()

# Now call main() and capture stdout/tracebacks
print("\nCalling robustness.main() ...")
try:
    # call with same CLI signature as user would
    mod.main(["--config", str(cfg_path)])
    print("robustness.main() returned normally")
except SystemExit as se:
    print("robustness.main() called SystemExit:", se)
except Exception as e:
    print("Exception while running robustness.main():", e)
    traceback.print_exc()
