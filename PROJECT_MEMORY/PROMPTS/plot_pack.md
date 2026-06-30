# Diagnostic Plot Pack Prompt

Use this prompt when asking an agent to build or update the evaluation report / diagnostic plot pack.

```text
You are working in the Kaggle ROGII Wellbore Geology Prediction repo.

Read PROJECT_MEMORY/AGENTS.md, PROJECT_MEMORY/PROJECT_STATE.md, PROJECT_MEMORY/docs/evaluation_protocol.md, and PROJECT_MEMORY/docs/coding_style.md.

Create or update a diagnostic plot pack that compares baseline vs experiment predictions. Do not train models and do not run Colab notebooks.

Required metrics:
- Overall RMSE baseline vs experiment.
- Per-well RMSE delta.
- RMSE by distance-from-Prediction-Start bins.
- RMSE by confidence decile if diagnostics include confidence.

Required plots:
- Cumulative error drift before/after.
- GR alignment before/after.
- Correction offset + confidence.
- Per-well delta RMSE ranking.
- Error vs GR texture/uniqueness when available.

Implementation requirements:
- Use numpy-first calculations for metrics.
- Keep pandas for CSV I/O and light joins.
- Make the report deterministic.
- Support CLI/config arguments for baseline path, experiment path, diagnostics path, output dir, top-k wells, and selected wells.
- Do not depend on train-only columns for inference diagnostics.

Run python -m pytest -q if tests are available. Commit and push the branch.
```
