# 005 Failed Trials

Do not retry these without a new reason and a clear diagnostic plan.

1. Delta-TVT cumulative integration: unstable and drift-prone.
2. BUDA alert overfit: train-only and not available at inference.
3. Hard physics model as direct feature/model: too rigid and risky.
4. Huber tuning: did not solve the main error mode.
5. Ridge baseline: too weak.
6. Two-stage cascade: did not justify complexity.
7. Meta-feature stacking leakage: leakage risk.
8. LGBM-CatBoost ensemble: did not justify extra complexity/runtime.
9. PyTorch sequence model: too expensive/risky for current gains.
10. Kalman filter: did not match the error structure well enough.
11. Custom objective: cancelled.
12. Candidate paths as feature: diluted model weights.

Decision: Avoid re-running these directions unless the user explicitly asks or new diagnostics show a specific fix.
