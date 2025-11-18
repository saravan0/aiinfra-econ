# Predictor stability report

Generated: 2025-11-14T20:42:57.175641Z

Input lean file: S:\aiinfra-econ\data\processed\features_lean.csv

Total rows (original): 6940


## Coverage (original)

| predictor                     | present   |   n_nonnull |   pct |
|:------------------------------|:----------|------------:|------:|
| gov_index_zmean               | True      |        5083 | 73.24 |
| trade_exposure                | True      |        6818 | 98.24 |
| inflation_consumer_prices_pct | True      |        4609 | 66.41 |
| fdi_inflow_usd_ln_safe        | True      |        5704 | 82.19 |
| imports_usd_ln_safe           | True      |        5573 | 80.3  |
| exports_usd_ln_safe           | True      |        5522 | 79.57 |


## Sample counts (original)

- total_rows: 6940
- n_target_only: 6341
- n_core: 3407
- n_core_plus_extras: 2787
- core_plus_fdi_inflow_usd_ln_safe: 3115
- core_plus_imports_usd_ln_safe: 3050
- core_plus_exports_usd_ln_safe: 3050


## After imputation (if applied)

- total_rows: 6940
- n_target_only: 6341
- n_core: 3407
- n_core_plus_extras: 3198
- core_plus_fdi_inflow_usd_ln_safe: 3407
- core_plus_imports_usd_ln_safe: 3198
- core_plus_exports_usd_ln_safe: 3198


### Quick recommendation

- Use the imputed file as model input if `n_core_plus_extras` increases substantially (>=10% increase).
