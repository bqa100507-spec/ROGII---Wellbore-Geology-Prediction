# Workflow

1. User creates or requests an experiment.
2. Agent checks `git status --short --branch`.
3. Agent creates a new branch unless the user explicitly says the current branch is correct.
4. Agent reads `PROJECT_MEMORY/PROJECT_STATE.md` and the relevant prompt in `PROJECT_MEMORY/PROMPTS/`.
5. Agent implements the requested change.
6. Agent runs local smoke tests that match the scope of the change.
7. Agent commits and pushes the branch.
8. User trains on Colab via `notebook/train_colab.ipynb`.
9. User evaluates locally via `notebook/errorAnalysis.ipynb`.
10. User sends RMSE and plots to ChatGPT.
11. ChatGPT decides keep, delete, or iterate based on RMSE and diagnostics.
12. Agent merges into the submission notebook only when the user explicitly requests it.

Default lightweight check:
```bash
python -m pytest -q
```

For model or inference changes, prefer a bounded smoke run before handing off:
```bash
python src/train.py --data_dir data --exp_name smoke_fast --max_wells 8 --max_train_rows_per_well 120 --n_estimators 80 --early_stopping_rounds 20 --log_period 0 --recursive_valid_wells 0
python src/predict.py --data_dir data --model_dir model/smoke_fast --output data/submission_smoke.csv
```
