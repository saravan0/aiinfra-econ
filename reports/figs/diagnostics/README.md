# FE Diagnostics — Stage 1 (AI-Infra Economic Dashboard)

**Generated:** <timestamp when produced by script>

## Purpose
This folder contains the fixed-effects regression diagnostics, partial residual plots, influence tables, and robustness checks used in the Stage-1 analysis of the AI-Infrastructure × Global Economics project. These artifacts support reproducibility, interpretation, and inclusion of figures/tables in the application portfolio (SOP, CV, LOR, and research appendix).

## Files (important)
- `fe_diagnostics_result.joblib` — fitted FE model (statsmodels RegressionResultsWrapper).
- `fe_diagnostics_summary.json` — numeric summary (AIC/BIC, R², skew/kurtosis, max Cook's D, etc.).
- `robustness/robustness_summary.csv` and `.json` — coefficient/se comparisons across:
  - baseline FE
  - winsorized (1% and 2%) fits
  - HC3 robust SEs (attached in CSV where applicable)
- `influence_top_by_cooks.csv` — top observations ranked by Cook's D (pos_index, row_label, cooks_d, leverage, studentized_resid).
- `partials/partial_resid_<var>.(png|pdf|svg)` — partial residual scatter + OLS line + 95% CI for each predictor.
- `resid_vs_fitted.*`, `qq_studentized.*`, `scale_location.*`, `leverage_cooks.*`, `resid_hist.*` — standard diagnostic plots.

## How to interpret (quick)
- **FE vs OLS & ElasticNet:** FE results control for unobserved time-invariant country effects. Compare standardized effects in the Stage-1 snapshot to see direction/size consistency across methods.
- **Q-Q / Scale-Location:** departures in upper tails on the Q-Q indicate a few high-influence observations; use `influence_top_by_cooks.csv` for follow-up.
- **Robustness:** HC3 and winsorized fits are reported — if coefficient signs and significance are stable across these, results are robust.

## Recommended next steps for write-ups
1. Report baseline FE coefficient and standardized effect (σ units) with HC3 SEs in the methods table.
2. In the appendix include: partial residual plots, the influence table, and the robustness_summary.csv.
3. For admissions: add the 1–2 sentence interpretation (see `interpretation.txt`) to SOP or project one-pager.

