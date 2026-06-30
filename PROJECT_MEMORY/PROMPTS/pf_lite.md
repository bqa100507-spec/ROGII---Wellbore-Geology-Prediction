# PF-lite Prompt

Placeholder for the PF-lite experiment prompt.

```text
TODO: Fill in the detailed PF-lite design after the approach is finalized.

Current intent:
- Build a lightweight particle/filter style correction layer for TVT drift.
- Use only inference-available data: MD, X, Y, Z, GR, TVT_input, typewell TVT, typewell GR, and model predictions.
- Do not use train-only columns ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA.
- Keep runtime bounded for Kaggle 9h.
- Prefer numpy arrays and avoid pandas row loops.
- Emit diagnostics with confidence and per-well correction summaries.
- Evaluate with PROJECT_MEMORY/docs/evaluation_protocol.md before keeping the branch.
```
