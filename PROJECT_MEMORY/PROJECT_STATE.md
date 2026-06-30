# Project State

Last updated: 2026-06-30.

## Current Best

- Current best Public LB: `11.797`.
- Current best local RMSE: about `12.5`.
- Current best approach: Autoregressive LightGBM + structural offset + lookahead vision + delta-anchor + offset-clipping + gradient-eyes.
- Current pipeline: the model predicts `delta_tvt`, then recursively updates TVT history before predicting later rows.

## Successful Ideas

1. Structural offset: moved the project from about `20` RMSE to about `15`.
2. GR / lookahead vision: moved the project from about `15` RMSE to about `12`.
3. Delta-anchor.
4. Offset-clipping.
5. Gradient-eyes.
6. Macro-noise as an auxiliary stabilizer.

## Failed Or Risky Ideas

1. Delta-TVT cumulative integration.
2. BUDA alert overfit.
3. Hard physics model as direct feature/model.
4. Huber tuning.
5. Ridge baseline.
6. Two-stage cascade.
7. Meta-feature stacking leakage.
8. LGBM-CatBoost ensemble.
9. PyTorch sequence model.
10. Kalman filter.
11. Custom objective, cancelled.
12. Candidate paths as feature, because it diluted model weights.

## Current Recommended Roadmap

1. Plot Pack / Evaluation Report.
2. GR Phase-Locked Drift Corrector.
3. PF-lite.
4. DP / Viterbi alignment.
5. Dual-pipeline confidence blend.
6. Submission notebook merge.

## Decision Rule

Keep a branch only if local RMSE improves, or diagnostics clearly show reduced drift/phase error without severe per-well regression.

Do not submit or merge based on overall RMSE alone. Always inspect per-well regressions and drift/phase diagnostics.
