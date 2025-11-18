# scripts/test_model_defs.py
from pathlib import Path
import sys
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import model_defs
import pandas as pd

def main():
    df = pd.DataFrame({
        "iso3": ["USA","USA","FRA","IND","IND","CHN","CHN","CHN","DEU","DEU","DEU"],
        "gov_index_zmean": [0.5, 0.6, -0.2, 0.1, 0.2, 1.2, 0.9, 1.0, 0.3, 0.4, 0.2],
        "gdp_growth_pct": [2.0, 1.8, 3.1, 2.5, 2.3, 0.9, 1.1, 1.0, 1.5, 1.6, 1.4]
    })

    # test safe_select_columns: request an existing and a missing column
    cols = model_defs.safe_select_columns(df, ["gov_index_zmean","nonexistent_col"])
    print("safe_select_columns ->", cols)

    # test prepare_design_matrix
    X, cols = model_defs.prepare_design_matrix(df, ["gov_index_zmean"], add_constant=True)
    print("prepare_design_matrix cols:", cols)
    print("X shape:", X.shape)

    # test fixed effects
    df_fe = model_defs.country_fixed_effects(df, country_col="iso3", drop_first=True, prefix="FE")
    print("Fixed effects columns present:", [c for c in df_fe.columns if c.startswith("FE_")][:10])
    print("df_fe shape:", df_fe.shape)

    # test small-data guard
    ok = model_defs.check_enough_data_for_regression(df, ["gov_index_zmean","gdp_growth_pct"])
    print("Enough data for regression (>=10):", ok)

if __name__ == "__main__":
    main()
