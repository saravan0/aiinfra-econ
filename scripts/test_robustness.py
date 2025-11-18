# scripts/test_robustness.py
from pathlib import Path
import sys
HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.model import robustness
import json
print("Running quick robustness test (debug max_rows from config will cap rows).")
robustness.main(["--config", "config/model.yml"])
print("Done.")
