# Submission Merge Prompt

Use this prompt only when the user explicitly asks to merge code into the Kaggle submission notebook.

```text
You are working in the Kaggle ROGII Wellbore Geology Prediction repo.

Read PROJECT_MEMORY/AGENTS.md, PROJECT_MEMORY/PROJECT_STATE.md, PROJECT_MEMORY/docs/submission_rules.md, PROJECT_MEMORY/docs/competition.md, and PROJECT_MEMORY/docs/coding_style.md.

Merge the approved pipeline into notebook/submission.ipynb only because the user explicitly requested submission merge.

Requirements:
- The notebook must be self-contained, or include all needed source code in a Kaggle-compatible way.
- It must load Kaggle input data.
- It must train or load models as needed.
- It must run inference.
- It must create /kaggle/working/submission.csv.
- It must validate id,tvt schema, row count, no NaN, and no inf.
- It must finish under 9 hours.
- It must not use train-only columns during inference.
- It must not read true TVT after Prediction Start.

Run lightweight local tests where possible. Do not submit to Kaggle automatically.
```
