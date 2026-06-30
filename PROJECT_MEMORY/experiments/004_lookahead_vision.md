# 004 Lookahead Vision

- Experiment name: GR / lookahead vision
- Branch: historical experiment
- Date: 2026-06-30
- Idea: Use future-known non-target GR texture/lookahead features to help locate geological phase.
- Hypothesis: GR is available in test and can give local context without target leakage.
- Implementation summary: Historical approach moved performance from about `15` RMSE to about `12`.
- Files changed: historical feature engineering
- Local RMSE: major improvement.
- Public LB: contributed to current best `11.797`.
- Diagnostics: Improved drift and phase behavior when implemented causally for TVT.
- Result: Successful.
- Decision: Keep.
- Next action: Preserve leakage guard: GR lookahead is allowed because GR exists in test, but TVT future is not allowed.
