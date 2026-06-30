# 001 Structural Offset

- Experiment name: Structural offset
- Branch: historical experiment
- Date: 2026-06-30
- Idea: Add structural offset features/correction to reduce systematic TVT drift.
- Hypothesis: A coarse structural offset gives the model a better geological anchor than raw trajectory features alone.
- Implementation summary: Historical approach moved performance from about `20` RMSE to about `15`.
- Files changed: historical
- Local RMSE: improved substantially.
- Public LB: contributed to the current best path.
- Diagnostics: Reduced systematic drift in many wells.
- Result: Successful.
- Decision: Keep the concept.
- Next action: Do not remove structural offset without an ablation showing no regression.
