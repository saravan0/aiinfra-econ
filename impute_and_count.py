import pandas as pd
from pathlib import Path
import numpy as np

df = pd.read_csv('data/processed/features_lean.csv', low_memory=False)
target = 'gdp_growth_pct'
core = ['gov_index_zmean', 'trade_exposure', 'inflation_consumer_prices_pct']
extra = ['fdi_inflow_usd_ln_safe', 'imports_usd_ln_safe', 'exports_usd_ln_safe']

print('Rows before impute (core+extra):', df[[target]+core+extra].dropna().shape[0])

# country-mean impute for extra controls (conservative)
df_imputed = df.copy()
for col in extra:
    if col in df_imputed.columns:
        df_imputed[col + '_imputed_countrymean'] = df_imputed.groupby('iso3')[col].transform(lambda s: s.fillna(s.mean()))

# count after imputation
cols_after = core + [c + '_imputed_countrymean' for c in extra if c in df_imputed.columns]
n_after = df_imputed[[target] + cols_after].dropna().shape[0]
print('Rows after country-mean impute for extras:', n_after)

# Also try global-mean fallback
df_glob = df.copy()
for col in extra:
    if col in df_glob.columns:
        df_glob[col + '_imputed_globalmean'] = df_glob[col].fillna(df_glob[col].mean())
cols_glob = core + [c + '_imputed_globalmean' for c in extra if c in df_glob.columns]
print('Rows after global-mean impute for extras:', df_glob[[target] + cols_glob].dropna().shape[0])
