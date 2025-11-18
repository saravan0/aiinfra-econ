# soft_model_sample_preview.py
import pandas as pd

df = pd.read_csv("data/processed/features_lean.csv", low_memory=False)

target = "gdp_growth_pct"
core = ["gov_index_zmean", "trade_exposure", "inflation_consumer_prices_pct"]
extra = ["fdi_inflow_usd_ln_safe", "imports_usd_ln_safe", "exports_usd_ln_safe"]

def count_for(cols):
    """Count rows where target + cols are all non-missing."""
    return df[[target] + cols].dropna().shape[0]

print("Total rows:", df.shape[0])
print()

print("Strict core+extras:", count_for(core + extra))
print("Core only:", count_for(core))
print()

# Country-mean imputation
df2 = df.copy()
for col in extra:
    if col in df2.columns:
        df2[col + "_imp"] = df2.groupby("iso3")[col].transform(
            lambda s: s.fillna(s.mean())
        )

cols_after_impute = core + [col + "_imp" for col in extra if col + "_imp" in df2.columns]
n_after_impute = df2[[target] + cols_after_impute].dropna().shape[0]

print("Core + extras (country-mean imputed):", n_after_impute)
