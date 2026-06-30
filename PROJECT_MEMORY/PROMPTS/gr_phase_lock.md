# GR Phase-Locked Drift Corrector Prompt

Use this prompt when asking an agent to implement or iterate the GR Phase-Locked Drift Corrector.

```text
You are working in the Kaggle ROGII Wellbore Geology Prediction repo.

Read PROJECT_MEMORY/AGENTS.md, PROJECT_MEMORY/PROJECT_STATE.md, PROJECT_MEMORY/docs/competition.md, PROJECT_MEMORY/docs/coding_style.md, PROJECT_MEMORY/docs/evaluation_protocol.md, and this prompt.

Goal:
Implement or improve a post-processing corrector that phase-locks predicted TVT against typewell GR using only inference-available data:
- submission id,tvt predictions
- test horizontal MD, X, Y, Z, GR, TVT_input
- test typewell TVT, GR

Do not use train-only columns ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA. Do not read true TVT after Prediction Start. Do not train on Colab or run notebooks.

Algorithm shape:
1. Parse submission ids into well_id and row_index.
2. For each well, align prediction rows to horizontal GR.
3. Normalize horizontal GR and typewell GR robustly.
4. Interpolate typewell GR at candidate TVT offsets.
5. Score candidate offsets with robust loss over local windows.
6. Require enough GR texture, uniqueness, and improvement before applying a correction.
7. Smooth offsets with EWMA and cap max absolute offset and max step per row.
8. Apply correction with confidence-scaled alpha.
9. Emit corrected submission and diagnostics including raw_offset, smooth_offset, confidence, GR cost before/after, texture, uniqueness, and used_correction.

Runtime guards:
- Bound offset_min, offset_max, offset_step.
- Bound window_size.
- Keep loops per well and per offset vectorized with numpy where feasible.
- Avoid df.iterrows() and df.apply() in the core algorithm.

Validation:
- Unit test id parsing, no NaN/inf output, deterministic output, low confidence on flat GR, and no dependency on train-only columns.
- Run python -m pytest -q.
- Use the diagnostic plot pack before deciding keep/delete/iterate.

Decision rule:
Keep only if local RMSE improves or diagnostics show reduced drift/phase error without severe per-well regression.
```
