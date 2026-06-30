# 000 Baseline

- Experiment name: Autoregressive LightGBM baseline
- Branch: historical baseline
- Date: 2026-06-30
- Idea: Predict `delta_tvt` recursively with LightGBM and update TVT history after every prediction.
- Hypothesis: Recursive delta prediction is safer than direct cumulative TVT integration for long missing intervals.
- Implementation summary: Existing `src/train.py`, `src/predict.py`, feature generation, and recursive inference pipeline.
- Files changed: historical
- Local RMSE: about `12.5` for the current best evolved pipeline; earlier baseline was worse.
- Public LB: best known `11.797`.
- Diagnostics: Use per-well RMSE, drift plots, GR alignment, and distance-from-PS bins.
- Result: Keep as the main pipeline foundation.
- Decision: Preserve backward compatibility.
- Next action: Evaluate only bounded changes that reduce drift or phase error.
