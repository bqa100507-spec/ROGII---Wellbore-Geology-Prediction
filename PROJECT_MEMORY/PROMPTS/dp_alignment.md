# DP / Viterbi Alignment Prompt

Placeholder for the DP / Viterbi alignment experiment prompt.

```text
TODO: Fill in the detailed dynamic-programming alignment design after the approach is finalized.

Current intent:
- Explore a bounded DP/Viterbi path alignment between predicted TVT trajectory and typewell GR.
- Use only inference-available data.
- Do not read true TVT after Prediction Start.
- Do not use train-only columns ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA.
- Penalize jagged offsets and impossible TVT jumps.
- Add strict runtime guards: candidate count, band width, stride, max rows, and early exit.
- Prefer numpy arrays for cost matrices and transitions.
- Emit path, confidence, offset, and cost diagnostics.
- Evaluate with PROJECT_MEMORY/docs/evaluation_protocol.md before keeping the branch.
```
