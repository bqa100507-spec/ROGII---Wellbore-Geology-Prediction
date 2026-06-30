# 003 Dynamic Alignment

- Experiment name: Dynamic alignment family
- Branch: current/planned experiments
- Date: 2026-06-30
- Idea: Use GR/typewell alignment to correct phase drift in predicted TVT trajectories.
- Hypothesis: Horizontal GR texture can reveal phase mismatch against typewell GR when the predicted TVT path drifts.
- Implementation summary: GR Phase-Locked Drift Corrector is the current concrete version; DP/Viterbi alignment is a planned bounded variant.
- Files changed: `src/gr_phase_lock.py`, diagnostics/tests in current experiment branches.
- Local RMSE: pending user Colab/local evaluation.
- Public LB: not submitted unless user requests.
- Diagnostics: GR alignment before/after, correction offset/confidence, RMSE by confidence decile, per-well delta RMSE.
- Result: Promising but must be guarded by diagnostics.
- Decision: Iterate only if drift/phase error improves without severe per-well regression.
- Next action: Run Plot Pack / Evaluation Report after each alignment branch.
