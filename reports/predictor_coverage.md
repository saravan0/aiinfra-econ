# Predictor coverage summary

- features file: `S:\aiinfra-econ\data\processed\features_lean.csv`
- total rows (observations): **6,940**

## Per-predictor coverage

- `trade_exposure` — present=True, n_nonnull=6,818, pct=98.2%
- `gdp_per_capita_usd_ln_safe` — present=True, n_nonnull=6,413, pct=92.4%
- `gdp_usd_ln_safe` — present=True, n_nonnull=6,413, pct=92.4%
- `fdi_inflow_usd_ln_safe` — present=True, n_nonnull=5,704, pct=82.2%
- `imports_usd_ln_safe` — present=True, n_nonnull=5,573, pct=80.3%
- `exports_usd_ln_safe` — present=True, n_nonnull=5,522, pct=79.6%
- `gov_index_zmean` — present=True, n_nonnull=5,083, pct=73.2%
- `inflation_consumer_prices_pct` — present=True, n_nonnull=4,609, pct=66.4%
- `total_reserves_usd_ln_safe` — present=True, n_nonnull=4,314, pct=62.2%
- `current_account_balance_usd_ln_safe` — present=True, n_nonnull=1,511, pct=21.8%

## Year coverage (top / bottom years)

Top years by avg predictor coverage:
- 2014: avg_coverage=81.9%
- 2013: avg_coverage=81.6%
- 2012: avg_coverage=81.4%

Bottom years by avg predictor coverage:
- 2001: avg_coverage=66.2%
- 1996: avg_coverage=20.0%
- 1998: avg_coverage=20.0%

## Country-level (top sample)

- CHN: n_obs=27, avg_predictor_pct=93.3%
- MYS: n_obs=27, avg_predictor_pct=93.0%
- SWE: n_obs=27, avg_predictor_pct=92.6%
- DEU: n_obs=27, avg_predictor_pct=92.2%
- ISR: n_obs=27, avg_predictor_pct=92.2%
- DNK: n_obs=27, avg_predictor_pct=91.9%
- KOR: n_obs=27, avg_predictor_pct=91.9%
- NOR: n_obs=27, avg_predictor_pct=91.9%
- JPN: n_obs=27, avg_predictor_pct=91.9%
- SGP: n_obs=27, avg_predictor_pct=91.1%

## Quick recommendations
- If any baseline predictor has <30% coverage → consider (i) imputing conservatively, (ii) dropping that predictor from baseline, OR (iii) expanding panel (add years/countries).
- If many monetary predictors have tiny coverage but share the same pattern, consider creating a `monetary_available` flag and using group-wise imputation.
- If coverage collapses after filtering by sample_filter, run the same analysis with `max_rows=None` and without the filter to see raw coverage.
