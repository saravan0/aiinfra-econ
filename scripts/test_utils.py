# scripts/test_utils.py
"""
Quick verification for src/model/utils.py

This script is safe to run from anywhere; it ensures the repository root is
on sys.path so `import src` works regardless of how Python was invoked.
"""
from pathlib import Path
import sys
import os

# --- ensure repo root is on sys.path ---
# If this script is located in <repo>/scripts/, repo root is parent.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Simple diagnostics
print("Python executable:", sys.executable)
print("sys.path[0]:", sys.path[0])
print("Repo root used:", REPO_ROOT)

# Now safe to import project modules
from src.model import utils
import pandas as pd

def main():
    # synthetic test data
    df = pd.DataFrame({
        "gov_index_zmean": [0.1, 0.5, -0.2, 1.3, 0.7],
        "gdp_growth_pct": [2.1, 1.8, 3.0, 0.5, 1.2]
    })

    out_path = Path("reports/test_plot_utils.png")
    # Ensure reports dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = utils.plot_scatter_with_fit(df, "gov_index_zmean", "gdp_growth_pct", fname=out_path)
    exists = out_path.exists()
    print("Saved plot exists:", exists)
    if exists:
        try:
            sha = utils.sha256sum(out_path)
        except Exception as e:
            sha = f"sha error: {e}"
        print("SHA256:", sha)
    else:
        print("Plot was not written. Check permissions / path.")

    print("Reports directory listing:", utils.list_dir(Path("reports")))

if __name__ == "__main__":
    main()
