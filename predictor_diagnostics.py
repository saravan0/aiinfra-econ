import pandas as pd, numpy as np, json
from pathlib import Path

reports = Path('reports')
reports.mkdir(parents=True, exist_ok=True)

predictors = [
  'gov_index_zmean',
  'trade_exposure',
  'inflation_consumer_prices_pct',
  'fdi_inflow_usd_ln_safe',
  'imports_usd_ln_safe',
  'exports_usd_ln_safe',
  'imports_usd_ln', 'exports_usd_ln',
  'imports_usd', 'exports_usd'
]

df = pd.read_csv('data/processed/features_lean.csv', low_memory=False)
n_total = len(df)

rows = []
for p in predictors:
    present = p in df.columns
    if not present:
        rows.append({
            'predictor': p,
            'present': False,
            'n_nonnull': 0,
            'pct': 0.0,
            'dtype': None,
            'n_strings': None,
            'min': None,
            'max': None,
            'n_zero': None,
            'n_neg': None,
            'n_inf': None
        })
        continue

    s = df[p]
    n_nonnull = s.notna().sum()
    pct = 100.0 * n_nonnull / n_total

    coerced = pd.to_numeric(s, errors='coerce')
    n_strings = int((s.notna() & coerced.isna()).sum())
    n_zero = int((coerced == 0).sum())
    n_neg = int((coerced < 0).sum())
    n_inf = int(np.isinf(coerced.fillna(0)).sum())
    _min = None if coerced.dropna().empty else float(coerced.min())
    _max = None if coerced.dropna().empty else float(coerced.max())

    rows.append({
        'predictor': p,
        'present': True,
        'n_nonnull': int(n_nonnull),
        'pct': round(pct, 2),
        'dtype': str(s.dtype),
        'n_strings': n_strings,
        'min': _min,
        'max': _max,
        'n_zero': n_zero,
        'n_neg': n_neg,
        'n_inf': n_inf
    })

diag_df = pd.DataFrame(rows)
diag_df.to_csv(reports / 'predictor_diagnostics.csv', index=False)

# duplicate checks
pairs = []
check_pairs = [
    ('imports_usd_ln','imports_usd_ln_safe'),
    ('exports_usd_ln','exports_usd_ln_safe'),
    ('gdp_usd_ln','gdp_usd_ln_safe'),
    ('gdp_per_capita_usd_ln','gdp_per_capita_usd_ln_safe')
]

for a,b in check_pairs:
    if a in df.columns and b in df.columns:
        xa = pd.to_numeric(df[a], errors='coerce')
        xb = pd.to_numeric(df[b], errors='coerce')
        joint = pd.concat([xa, xb], axis=1).dropna()
        corr = None if joint.empty else float(joint.corr().iloc[0,1])
        pairs.append({'a': a, 'b': b, 'n_joint': len(joint), 'pearson': corr})
    else:
        pairs.append({'a': a, 'b': b, 'n_joint': 0, 'pearson': None})

pd.DataFrame(pairs).to_csv(reports / 'predictor_duplicate_checks.csv', index=False)

print('Diagnostics complete. Files saved to reports/:')
print('- predictor_diagnostics.csv')
print('- predictor_duplicate_checks.csv')
