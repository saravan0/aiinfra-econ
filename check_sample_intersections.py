import pandas as pd
from pathlib import Path

df = pd.read_csv('data/processed/features_lean.csv', low_memory=False)
print('Total rows:', len(df))

target = 'gdp_growth_pct'
core = ['gov_index_zmean', 'trade_exposure', 'inflation_consumer_prices_pct']
extra = ['fdi_inflow_usd_ln_safe', 'imports_usd_ln_safe', 'exports_usd_ln_safe']

def n_with(cols):
    cols_req = [target] + cols
    n = df[cols_req].dropna().shape[0]
    return n

print('n (target only):', n_with([]))
print('n (core):', n_with(core))
print('n (core + extra):', n_with(core + extra))
print('n (core + 1 extra each):')
for e in extra:
    print('  ', e, n_with(core + [e]))
