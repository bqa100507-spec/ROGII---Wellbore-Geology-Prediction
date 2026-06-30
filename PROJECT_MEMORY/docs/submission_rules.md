# Submission Rules

- Only merge into the submission notebook when the user explicitly requests it.
- The submission notebook must be self-contained, or include all needed source code in a Kaggle-compatible way.
- It must load Kaggle input data.
- It must train or load models as needed.
- It must run inference.
- It must create `/kaggle/working/submission.csv`.
- It must check `id,tvt` schema.
- It must check row count.
- It must check no NaN.
- It must check no inf.
- It must finish under the 9 hour Kaggle runtime limit.

Do not assume local files outside `/kaggle/input` and `/kaggle/working` are available in the submitted notebook.
