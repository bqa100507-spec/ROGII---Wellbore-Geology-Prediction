# Pipeline Prompt

Use this prompt when asking an agent to implement a normal experiment branch.

```text
You are working in the Kaggle ROGII Wellbore Geology Prediction repo.

First read PROJECT_MEMORY/AGENTS.md, PROJECT_MEMORY/PROJECT_STATE.md, PROJECT_MEMORY/docs/workflow.md, PROJECT_MEMORY/docs/competition.md, PROJECT_MEMORY/docs/coding_style.md, PROJECT_MEMORY/docs/submission_rules.md, and PROJECT_MEMORY/docs/evaluation_protocol.md.

Create a new branch for the experiment unless I explicitly say the current branch is correct. Check git status before editing, and do not overwrite a dirty working tree.

Implement the requested experiment without training on Colab and without running notebooks. Do not merge into the submission notebook unless I explicitly ask.

Respect inference constraints:
- Do not use train-only columns ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA.
- Test horizontal has only MD, X, Y, Z, GR, TVT_input.
- Test typewell has only TVT, GR.
- Do not read true TVT after Prediction Start.
- Recursive inference must update working TVT history after each prediction.

Prefer numpy arrays for core algorithms. Avoid pandas row loops. Add runtime guards so Kaggle can finish under 9 hours. Kaggle has 2 T4 GPUs, but use GPU only if it clearly helps.

Run the lightest relevant tests. For docs-only changes, run:
python -m pytest -q

For model/inference changes, also run bounded smoke train/predict if feasible:
python src/train.py --data_dir data --exp_name smoke_fast --max_wells 8 --max_train_rows_per_well 120 --n_estimators 80 --early_stopping_rounds 20 --log_period 0 --recursive_valid_wells 0
python src/predict.py --data_dir data --model_dir model/smoke_fast --output data/submission_smoke.csv

Commit and push the branch. Report branch name, commit hash, changed files, tests, push status, and exact next commands for me.
```
