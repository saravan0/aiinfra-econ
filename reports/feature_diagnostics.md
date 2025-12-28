# Feature diagnostics — Generated: 2025-12-02T14:14:49.214070Z

## Summary (per-feature)
- Total features (numeric): 18
- Correlation threshold (flag): 0.8
- Missingness threshold (flag %): 50.0
- Zero-fraction threshold (flag): 0.9

## High-missing features

- current_account_balance_usd_ln_safe — pct_missing=68.94%

## Near-zero variance features
- None

## Many-zeros features
- None

## High-correlation pairs (abs(corr) >= 0.80)
- gov_index_zmean <-> gdp_per_capita_usd_ln_safe: corr=0.814
- gov_index_zmean <-> voice_accountability_imputed: corr=0.865
- gov_index_zmean <-> political_stability_imputed: corr=0.838
- gov_index_zmean <-> gov_effectiveness_imputed: corr=0.951
- gov_index_zmean <-> reg_quality_imputed: corr=0.932
- gov_index_zmean <-> rule_of_law_imputed: corr=0.975
- gov_index_zmean <-> control_corruption_imputed: corr=0.955
- gdp_usd_ln_safe <-> exports_usd_ln_safe: corr=0.959
- gdp_usd_ln_safe <-> imports_usd_ln_safe: corr=0.971
- gdp_usd_ln_safe <-> fdi_inflow_usd_ln_safe: corr=0.862
- gdp_usd_ln_safe <-> current_account_balance_usd_ln_safe: corr=0.829
- gdp_usd_ln_safe <-> total_reserves_usd_ln_safe: corr=0.878
- gdp_per_capita_usd_ln_safe <-> gov_effectiveness_imputed: corr=0.842
- gdp_per_capita_usd_ln_safe <-> reg_quality_imputed: corr=0.814
- gdp_per_capita_usd_ln_safe <-> rule_of_law_imputed: corr=0.802
- exports_usd_ln_safe <-> imports_usd_ln_safe: corr=0.985
- exports_usd_ln_safe <-> fdi_inflow_usd_ln_safe: corr=0.891
- exports_usd_ln_safe <-> current_account_balance_usd_ln_safe: corr=0.830
- exports_usd_ln_safe <-> total_reserves_usd_ln_safe: corr=0.839
- imports_usd_ln_safe <-> fdi_inflow_usd_ln_safe: corr=0.885
- imports_usd_ln_safe <-> current_account_balance_usd_ln_safe: corr=0.810
- imports_usd_ln_safe <-> total_reserves_usd_ln_safe: corr=0.854
- voice_accountability_imputed <-> rule_of_law_imputed: corr=0.805
- gov_effectiveness_imputed <-> reg_quality_imputed: corr=0.936
- gov_effectiveness_imputed <-> rule_of_law_imputed: corr=0.941
- gov_effectiveness_imputed <-> control_corruption_imputed: corr=0.925
- reg_quality_imputed <-> rule_of_law_imputed: corr=0.910
- reg_quality_imputed <-> control_corruption_imputed: corr=0.880
- rule_of_law_imputed <-> control_corruption_imputed: corr=0.946

## Exact duplicate columns
- None

## Top 20 country observation counts (panel completeness)
