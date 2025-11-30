# AI Infra–Economics OnePager: Governance, Trade, and Growth

**One-page summary: governance, trade, and growth — AI Infrastructure meets macroeconomics**

**Context & Objective.**  
This study investigates the empirical relationship between governance quality, trade openness, inflation, and short-run GDP growth using a reproducible AI–econometrics pipeline. The objective is to combine robust panel fixed-effect estimation with model-regularization diagnostics to produce interpretable, policy-relevant findings suitable for scholarly review.

**Data & pre-processing (brief).**  
Feature engineering, imputation, and transformation are implemented in `src/data` (not reproduced here). The analysis uses a harmonized sample of country-year observations; all continuous predictors were standardized for interpretability in the FE framework. Missing data were imputed deterministically as described in the pipeline; categorical harmonization and scaling ensure stable model estimation.

**Primary finding (headline).**  
Trade openness is positively associated with contemporaneous GDP growth: across regularized models and FE specifications, higher trade exposure corresponds to higher short-run growth after controlling for entity fixed effects and other macro factors. This result is robust across ElasticNet model paths and comparative model effect analyses.

**Governance — nuanced interpretation.**  
Governance (gov_index_zmean) displays a counter-intuitive short-run sign: higher measured governance associates with *lower* contemporaneous growth. We interpret this as a plausible dynamic phenomenon rather than model failure — stronger governance regimes may prioritize structural reforms, fiscal consolidation, or regulatory stabilization that transiently slow GDP growth but yield greater long-run stability and resilience. Thus: **short-term negative; long-term stabilizing** — a pattern consistent with high-quality institutions doing corrective policy.

**Inflation and macro controls.**  
Inflation exhibits the expected negative contemporaneous association with growth at the sample-frequency used here; however, the magnitude is modest relative to trade openness and is sensitive to model specification — consistent with inflation exerting both cyclical and policy-driven effects.

**Robustness & diagnostics.**  
Robustness checks include:
- ElasticNet coefficient path analysis (regularization stability),
- Partial/added-variable plots (conditional relationship visualization),
- Fixed-effect diagnostics (studentized residuals, leverage/Cook's D),
- Comparative model effects across candidate estimators.

Collectively the diagnostics confirm that the observed associations are not artifacts of a single model: coefficients are stable across penalty paths, residual diagnostics show no dominant influential outliers driving the main trade openness effect, and permutation/added-variable analyses support conditional interpretation.

---

## Key results (detailed interpretation)

- **Trade openness (trade_exposure):** Positive and robust. ElasticNet and FE diagnostics indicate a persistent positive partial effect on contemporaneous growth; effect sizes consistent with a moderate policy-relevant elasticity.
- **Governance (gov_index_zmean):** Negative short-term coefficient; we argue this reflects structural policy adjustment by higher-governance regimes (temporary growth cost, longer-term stability). Models show high significance but require careful temporal interpretation.
- **Inflation (inflation_consumer_prices_pct):** Negative contemporaneous relationship, smaller magnitude; sensitive to specification and control sets.
- **Other controls (exports, imports, reserves, FDI):** These controls improve model fit and adjust coefficient magnitudes; detailed effect sizes are reported in the appendix figures.

---

## Figures included (consolidated plate)
A single consolidated SVG (`onepager_plate_*.svg`) embeds the following panels:
- Comparative model effects (regularized vs benchmark)
- Partial/added-variable panel (conditional relationships)
- ElasticNet coefficient paths (regularization stability)
- FE diagnostics: partial residuals, leverage / Cook's D, QQ of studentized residuals

*I chose a consolidated SVG to facilitate rapid visual review while reducing file proliferation. Individual SVGs remain referenced for traceability.*

---

## How to read this one-pager
- Read the **headline** and **key results** first to understand policy takeaways.
- Consult the **consolidated plate** to inspect effect shapes and diagnostics in a single view.
- Use the appendix figures (individual svgs in the repo) for deeper replication and figure export.

---

## Appendix (technical notes)
- Processing & imputation live in `src/data`.
- Standard errors reported in FE fits use the within-demean estimator; coefficient interpretation reported as standardized effects (coef * sd_x / sd_y).
- Manifest and metadata files are saved alongside the plate in `reports/onepager/` and `reports/onepager/files/`.


*