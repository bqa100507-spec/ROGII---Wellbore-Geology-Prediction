# ROGII - Wellbore Geology Prediction

Autoregressive LightGBM pipeline for recursive multi-step TVT prediction.

## Directory Structure

- `data/`: Put Kaggle dataset here. Contains `train/`, `test/`, and `sample_submission.csv`.
- `model/`: Automatically created to store trained LightGBM artifacts (`model.joblib`, configs, features, metrics).
- `notebook/`: Jupyter Notebooks for Exploratory Data Analysis (EDA), local training, Colab execution, and error analysis.
- `src/`: Core Python modules.
  - `dataset.py`: Data loading and parsing utility.
  - `features.py`: Feature Engineering logic (Autoregressive features, Lags, Rolling stats, Lookahead GR, Deltas).
  - `model.py`: LightGBM architecture, parameters, and artifact serialization.
  - `train.py`: Training script for Holdout and GroupKFold splits.
  - `predict.py`: Inference script for Autoregressive predicting missing TVT gaps recursively.
- `tests/`: PyTest unit tests to ensure no data leakage.

## Install

```bash
pip install -r requirements.txt
```

## Basic Commands

### Train (Holdout Split)
Holdout validation by well is the default for quick iterations. By default, it excludes the test set's well IDs to avoid data leakage.
```bash
python src/train.py --data_dir data --exp_name lgbm_baseline --split holdout --valid_size 0.2
```

### Train (GroupKFold Cross-Validation)
For more robust evaluation across different folds of wells:
```bash
python src/train.py --data_dir data --exp_name lgbm_kfold --split groupkfold --n_splits 5
```

### Predict
Generate recursive predictions for missing `TVT_input` rows. It updates the trajectory sequentially.
```bash
python src/predict.py --data_dir data --model_dir model/lgbm_baseline --output data/submission.csv
```

### Test
Run unit tests to ensure the feature matrix is strictly causal (no leakage from the future).
```bash
pytest
```
