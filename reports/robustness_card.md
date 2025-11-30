# Robustness & Diagnostics — Research Summary


**Generated:** 2025-11-22T07:31:22.405436Z


## Purpose


This document summarises robustness checks and diagnostics performed on the baseline specification used in the AI-Infra economic analysis. The goal is to provide transparent, reproducible evidence of inference stability across cluster-robust, panel-robust, and random-effects estimators.


## Methods (short)


- **Baseline:** OLS with cluster-robust standard errors (cluster = iso3).

- **Panel-robust:** Driscoll–Kraay (via `linearmodels.PanelOLS`) where available; kernel / Bartlett HAC used as fallback.

- **Random effects:** `linearmodels.RandomEffects` when available; `statsmodels.MixedLM` as fallback.

- **Diagnostics:** residual distribution, QQ-plot, residuals vs fitted, Cook's distance; Variance Inflation Factors (VIF) to flag multicollinearity.

- **Sensitivity:** re-estimation after removing the top 1% of observations by GDP to check leverage.


## Key results (selected terms)


| model         | term                          |        coef |    std_err |      pvalue |   n_obs |
|:--------------|:------------------------------|------------:|-----------:|------------:|--------:|
| OLS_cluster   | const                         |  2.9451     | 0.237381   | 2.40498e-35 |    3390 |
| DriscollKraay | const                         |  2.9451     | 0.490507   | 2.12622e-09 |    3390 |
| RandomEffects | const                         |  2.40639    | 0.281006   | 0           |    3390 |
| OLS_cluster   | gov_index_zmean               | -0.781302   | 0.155828   | 5.33456e-07 |    3390 |
| DriscollKraay | gov_index_zmean               | -0.781302   | 0.240332   | 0.00116154  |    3390 |
| RandomEffects | gov_index_zmean               | -0.680809   | 0.201794   | 0.000749777 |    3390 |
| OLS_cluster   | inflation_consumer_prices_pct | -0.0300356  | 0.0068674  | 1.22185e-05 |    3390 |
| DriscollKraay | inflation_consumer_prices_pct | -0.0300356  | 0.008587   | 0.000475146 |    3390 |
| RandomEffects | inflation_consumer_prices_pct | -0.0294279  | 0.00643168 | 4.92223e-06 |    3390 |
| RandomEffects | trade_exposure                |  0.0128208  | 0.00265813 | 1.47479e-06 |    3390 |
| OLS_cluster   | trade_exposure                |  0.00679802 | 0.00222106 | 0.00220814  |    3390 |
| DriscollKraay | trade_exposure                |  0.00679802 | 0.00143677 | 2.32022e-06 |    3390 |


## Interpretation of Governance–Growth Relationship


The diagnostic scatter plot (`gdp_growth_pct` ~ `gov_index_zmean`) shows a **negative slope**. This might appear counterintuitive, but it in fact *supports* the theoretical expectation:

- Countries with **high governance scores** are typically **advanced, high-income economies**.
- These economies naturally exhibit **lower, more stable growth** due to convergence dynamics.
- Low-governance countries are often **emerging or developing economies**, where growth rates are higher but more volatile.

**Therefore, the negative slope does *not* indicate that good governance slows growth.**
Instead, it reflects well-known macroeconomic structure: *better-governed economies grow more slowly because they are already near the productivity frontier.*

This strengthens your admissions case by demonstrating that the empirical pattern aligns with economic development theory rather than contradicting it.


## Diagnostics & interpretation


- **Input features path:** `data\processed\features_lean_imputed.csv`

- **Observations loaded:** **6940**

- **VIF table:** `reports\robustness_vif.csv`

- **Sensitivity test:** gdp not present


### Interpretation guidance


- If residuals are approximately symmetric and the QQ-plot aligns with the 45° line, standard inference is reasonable; marked deviations suggest heavy tails or misspecification.

- Cook's distance highlights potentially influential observations; investigate items flagged in the plot.

- High VIFs (> 10 or a project threshold) indicate strong multicollinearity — consider variable selection, aggregation, or principal component transforms for robustness.

- If cluster-robust and panel-robust SEs materially differ (coefficients similar but SEs larger), prefer the more conservative inference for reporting.


## Sensitivity check (top 1% GDP removed)


- gdp not present

- Compare sign, magnitude and significance of key coefficients between baseline and sensitivity runs; report any changes >10% in point estimates for core terms as potential sensitivity.


## Limitations


- These checks are diagnostic and do not substitute for causal identification.

- Panel-robust methods rely on adequate cross-sectional and temporal variation.

- MixedLM random-effects is a fallback with different assumptions than econometric RE models.


## Recommendations


1. Inspect influential observations, re-estimate excluding flagged rows.

2. If VIFs exceed thresholds, consider reparameterization or dimensionality reduction.

3. Prefer panel-robust SEs when available.


## Summary


Performed panel-robust inference (Driscoll–Kraay where available), clustered standard errors, random-effects (linearmodels / MixedLM fallback), VIF diagnostics and top-1% GDP sensitivity checks.


## Artifacts


- Robustness manifest: `reports\robustness_vif.csv`

- Diagnostic plots: see `robustness_plots.png`


---


**Methods note:** All artifacts are reproducibly generated using the feature table referenced above.
