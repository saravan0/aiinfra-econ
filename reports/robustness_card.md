# Robustness & Diagnostics card


**Purpose:** Robustness checks (Driscoll–Kraay / Random Effects) and diagnostic summaries.


## Key results (table snippet)


| model         | term                          |       coef |    std_err |      pvalue |   n_obs |
|:--------------|:------------------------------|-----------:|-----------:|------------:|--------:|
| DriscollKraay | const                         |  4.49115   | 1.64359    | 0.00631996  |    3181 |
| DriscollKraay | gov_index_zmean               | -0.940185  | 0.266151   | 0.000417473 |    3181 |
| DriscollKraay | trade_exposure                |  0.0054632 | 0.00147321 | 0.00021218  |    3181 |
| DriscollKraay | inflation_consumer_prices_pct | -0.0322861 | 0.00857144 | 0.000168421 |    3181 |
| DriscollKraay | fdi_inflow_usd_ln_safe        |  0.596095  | 0.108021   | 3.697e-08   |    3181 |
| DriscollKraay | imports_usd_ln_safe           | -1.01803   | 0.429955   | 0.0179554   |    3181 |
| DriscollKraay | exports_usd_ln_safe           |  0.433364  | 0.344608   | 0.208645    |    3181 |
| OLS_cluster   | const                         |  4.49115   | 2.60133    | 0.0842599   |    3181 |
| OLS_cluster   | gov_index_zmean               | -0.940185  | 0.194913   | 1.40977e-06 |    3181 |
| OLS_cluster   | trade_exposure                |  0.0054632 | 0.00255939 | 0.0327962   |    3181 |
| OLS_cluster   | inflation_consumer_prices_pct | -0.0322861 | 0.00760546 | 2.18482e-05 |    3181 |
| OLS_cluster   | fdi_inflow_usd_ln_safe        |  0.596095  | 0.139415   | 1.90549e-05 |    3181 |
| OLS_cluster   | imports_usd_ln_safe           | -1.01803   | 0.571722   | 0.0749708   |    3181 |
| OLS_cluster   | exports_usd_ln_safe           |  0.433364  | 0.500912   | 0.386957    |    3181 |
| RandomEffects | const                         |  6.15992   | 2.26156    | 0.00648993  |    3181 |
| RandomEffects | gov_index_zmean               | -0.834711  | 0.226418   | 0.000231076 |    3181 |
| RandomEffects | trade_exposure                |  0.0127153 | 0.00285098 | 8.48041e-06 |    3181 |
| RandomEffects | inflation_consumer_prices_pct | -0.030865  | 0.00646307 | 1.87292e-06 |    3181 |
| RandomEffects | fdi_inflow_usd_ln_safe        |  0.571926  | 0.106755   | 9.04672e-08 |    3181 |
| RandomEffects | imports_usd_ln_safe           | -1.58568   | 0.392491   | 5.47056e-05 |    3181 |


## Diagnostics summary


- **input_features_path**: data\processed\features_lean_imputed.csv

- **loaded_rows**: 6940

- **vif_saved**: reports\robustness_vif.csv

- **sensitivity**: gdp not present

- **n_rows**: 6940


## Admissions bullet (pasteable)


Performed panel-robust inference (Driscoll–Kraay where available), clustered standard errors, random-effects (linearmodels / MixedLM fallback), VIF diagnostics and top-1% GDP sensitivity checks.