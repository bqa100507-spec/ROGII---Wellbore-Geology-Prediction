# Agent Instructions

This is the Kaggle ROGII Wellbore Geology Prediction repository.

Before starting any task:
- Read `PROJECT_MEMORY/PROJECT_STATE.md`.
- Check `git status --short --branch`.
- Do not overwrite a dirty working tree. If modified or untracked files exist, report them and stop unless the user explicitly tells you how to proceed.
- Create a new branch for each experiment unless the user explicitly says the current branch is already correct.
- If the user mentions an algorithm or experiment name, read the relevant prompt in `PROJECT_MEMORY/PROMPTS/` before editing code.

Hard rules:
- Do not train on Colab automatically.
- Do not run Colab notebooks automatically.
- Do not optimize blindly for Public LB.
- Do not merge or consolidate the submission notebook unless the user explicitly requests it.
- Do not modify the current train/predict pipeline unless the task asks for it.
- When a branch is complete, run a lightweight smoke test if feasible, commit, and push.
- Final response must print branch name, commit hash, files changed, test result, push status, and commands for the user to run next.

Data and leakage constraints:
- Inference must not use train-only columns: `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`.
- Test horizontal CSV has only `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`.
- Test typewell CSV has only `TVT`, `GR`.
- Inference must not read true `TVT` after Prediction Start.
- Recursive inference must update the working TVT history immediately after each prediction so lag and rolling features see the predicted value.

Implementation style:
- Prefer numpy arrays for core algorithms.
- Use pandas only for I/O and light schema handling.
- Avoid `df.iterrows()` and `df.apply()` in core algorithms when numpy can replace them.
- Precompute arrays before heavy loops.
- Heavy algorithms must expose runtime-limiting config so they can finish under the 9 hour Kaggle limit.
- Kaggle provides 2 T4 GPUs, but use GPU only when it clearly helps.
- Keep code easy to merge into the submission notebook.

Useful local checks:
```bash
python -m pytest -q
python src/train.py --data_dir data --exp_name smoke_fast --max_wells 8 --max_train_rows_per_well 120 --n_estimators 80 --early_stopping_rounds 20 --log_period 0 --recursive_valid_wells 0
python src/predict.py --data_dir data --model_dir model/smoke_fast --output data/submission_smoke.csv
```

The fast train/predict smoke commands are optional unless the user asks for model work. For documentation-only changes, `python -m pytest -q` is usually enough if tests are runnable.
