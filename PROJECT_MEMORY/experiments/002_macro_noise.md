# 002 Macro Noise

- Experiment name: Macro-noise auxiliary
- Branch: historical experiment
- Date: 2026-06-30
- Idea: Add macro-noise style auxiliary signal/regularization to improve robustness.
- Hypothesis: Controlled broad noise can reduce overconfidence and make recursive prediction less brittle.
- Implementation summary: Treated as an auxiliary stabilizer, not the main source of signal.
- Files changed: historical
- Local RMSE: useful only as supporting idea.
- Public LB: not tracked separately.
- Diagnostics: Must be checked for per-well regressions.
- Result: Conditionally useful.
- Decision: Keep as a supporting technique, not a primary roadmap item.
- Next action: Use only if diagnostics show stability gains without drift amplification.
