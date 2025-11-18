# Robustness diagnostics

<!-- ROBUSTNESS-START -->

### Multicollinearity diagnostics and remedial action

Variance inflation factor (VIF) diagnostics indicate strong multicollinearity among the trade-related covariates.
Specifically, `exports_usd_ln_safe` (VIF = 36.25), `imports_usd_ln_safe` (VIF = 34.72), `fdi_inflow_usd_ln_safe` (VIF = 4.97).
Pairwise correlation between exports and imports is 0.9892, indicating near-linear dependence.

To preserve interpretability and numerical stability, the recommended, admissions-friendly correction is to remove `imports_usd_ln_safe` (it adds little independent information beyond `exports_usd_ln_safe`) and re-run the fixed-effects estimations. An alternative, defensible approach is to replace the three highly collinear series with a single PCA-derived factor (`trade_f1`).

Pre-fix condition-number diagnostics: `Condition number (standardized X): 14.798782889265171`.

Diagnostics snapshot generated on 2025-11-15 03:41 UTC. Full pre/post diagnostics are saved in the `reports/` folder.

**Top VIFs**

| feature | vif |
|---:|---:|
| const | 191.4877 |
| exports_usd_ln_safe | 36.2481 |
| imports_usd_ln_safe | 34.7174 |
| fdi_inflow_usd_ln_safe | 4.9659 |
| gov_index_zmean | 1.4702 |
| trade_exposure | 1.1664 |

**Selected correlations**

| pair | corr |
|---|---:|
| exports_usd_ln_safe — imports_usd_ln_safe | 0.9892 |
| exports_usd_ln_safe — fdi_inflow_usd_ln_safe | 0.9296 |
| exports_usd_ln_safe — trade_exposure | 0.0335 |
| imports_usd_ln_safe — fdi_inflow_usd_ln_safe | 0.9294 |
| imports_usd_ln_safe — trade_exposure | 0.0128 |
| fdi_inflow_usd_ln_safe — trade_exposure | 0.1215 |


<!-- ROBUSTNESS-END -->

