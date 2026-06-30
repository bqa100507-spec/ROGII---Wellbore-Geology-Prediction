# Evaluation Protocol

ChatGPT should decide whether to keep a branch based on RMSE plus plots, not RMSE alone.

## Required Metrics

- Overall RMSE baseline vs experiment.
- Per-well RMSE delta.
- RMSE by distance-from-Prediction-Start bins.
- RMSE by confidence decile, if confidence is available.

## Required Plots

- Cumulative error drift before/after.
- GR alignment before/after.
- Correction offset + confidence.
- Per-well delta RMSE ranking.
- Error vs GR texture/uniqueness, if available.

## Decision Rule

Keep the branch only if local RMSE improves, or diagnostics clearly show reduced drift/phase error without severe per-well regression.

Reject or iterate if:
- Overall RMSE improves but a few wells regress severely.
- Drift is reduced only in low-confidence regions by accident.
- The method depends on train-only columns.
- Runtime is not bounded enough for Kaggle.
