# Negative and Neutral Results

This document records modeling choices, specifications, and diagnostics that did not materially alter conclusions or improve
performance. These results are documented explicitly to support transparency and to contextualize the stability claims made
elsewhere in the project.

The intent is not to catalogue exhaustive experiments, but to surface outcomes that help interpret model behavior and
methodological robustness.

## Model Specifications with Limited Impact

Several alternative specifications were evaluated during development and validation but did not yield meaningful improvements in
forecast accuracy or diagnostic behavior relative to the baseline configurations.

In particular:
- Variants that increased model complexity without additional regularization did not improve out-of-sample performance.
- Alternative fixed-effects parameterizations produced coefficient patterns and forecast errors that were qualitatively similar
to the baseline specification.
- Changes in regularization strength outside a narrow effective range tended to either over-smooth coefficients or introduce
instability without improving predictive accuracy.

These outcomes motivated the retention of a restrained modeling configuration emphasizing interpretability and stability over
marginal performance gains.

## Feature Contributions Suppressed by Regularization

Regularized models consistently attenuated or suppressed several candidate predictors that appeared weakly correlated with the
outcome in isolation. In high-dimensional settings, these variables did not exhibit stable contributions once multicollinearity
and joint effects were accounted for.

Rather than forcing inclusion, these results were treated as evidence of limited explanatory power under realistic multivariate
constraints. This behavior aligns with the project’s objective of stress-testing economic relationships under regularization
rather than maximizing apparent in-sample fit.

## Diagnostic Stability and Neutral Findings

Multiple diagnostic checks were performed to assess sensitivity to modeling choices and temporal segmentation. In several cases, diagnostics confirmed stability rather than revealing new structure.

Examples include:
- Interpretability analyses that preserved relative feature influence ordering across time.
- Nonlinear trend inspection that did not materially alter conclusions drawn from linear specifications.
- Forecast error comparisons that showed consistent relative performance across adjacent forecast horizons.

These neutral findings are reported to clarify that observed stability is not an artifact of selective reporting.

## Forecast Evaluation Variants

Forecast evaluation under the expanding-window scheme was tested across multiple horizons. While absolute forecast errors varied
over time, relative performance patterns remained stable across configurations.

No alternative evaluation setup tested during development produced systematic reversals in model ranking or materially different
diagnostic behavior. This consistency supports the use of a single canonical evaluation protocol for reporting.

## Interpretation and Implications

The presence of negative and neutral results is informative in itself. The absence of dramatic performance gains under
alternative specifications suggests that the primary findings reflect structural properties of the data and modeling framework
rather than tuning artifacts.

Documenting these outcomes reinforces the project’s emphasis on reproducibility, methodological discipline, and interpretability
over opportunistic optimization.
