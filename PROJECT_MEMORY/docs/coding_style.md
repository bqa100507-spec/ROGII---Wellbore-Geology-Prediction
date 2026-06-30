# Coding Style

- Core algorithms should be numpy-first.
- Use pandas for I/O, joins, schema checks, and light data preparation.
- Avoid row-wise pandas operations in performance-sensitive code.
- Avoid `df.iterrows()` and `df.apply()` in core algorithm paths when numpy can replace them.
- Precompute arrays and reusable lookup tables before heavy loops.
- Use deterministic seeds.
- Expose CLI flags or config dictionaries for experiment behavior.
- Preserve backward compatibility for existing train/predict commands.
- Avoid heavy dependencies unless the gain is clear and Kaggle-compatible.
- Keep implementation easy to inline or bundle into a Kaggle submission notebook.

Runtime-sensitive code should include limits such as max wells, max candidates, max iterations, stride, window size, timeout-like guards, or early-exit thresholds.
