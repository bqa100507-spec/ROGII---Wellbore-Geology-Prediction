# ROGII - Wellbore Geology Prediction

Autoregressive LightGBM baseline for recursive multi-step TVT prediction.

## Project Structure

## Data

Due to copyright restrictions, the dataset is not included in this repository. 
To train or infer, you must upload the competition dataset zip file to Google Drive:
`MyDrive/ROGII/rogii-wellbore-geology-prediction.zip`

When running the `train_colab.ipynb` notebook in Colab, it will automatically mount Google Drive and extract the data into the `data/` folder.

```text
data/
  train/
  test/
  sample_submission.csv
model/
notebook/
src/
  dataset.py
  features.py
  model.py
  train.py
  predict.py
tests/
```

## Install

```bash
pip install -r requirements.txt
```

## Train

Holdout validation by well is the default:

```bash
python src/train.py --data_dir data --exp_name lgbm_baseline --split holdout --valid_size 0.2
```

GroupKFold validation for later experiments:

```bash
python src/train.py --data_dir data --exp_name lgbm_kfold --split groupkfold --n_splits 5
```

The training script excludes the current test wells by default to avoid leakage:
`000d7d20`, `00bbac68`, and `00e12e8b`.

## Predict

```bash
python src/predict.py --data_dir data --model_dir model/lgbm_baseline --output data/submission.csv
```

Prediction is recursive: every missing `TVT_input` row is predicted, written back into
the temporary TVT history, and then used to build lag and rolling features for later
rows in the same well.

## Debug Smoke Run

```bash
python src/train.py --data_dir data --exp_name smoke --max_wells 12 --max_train_rows_per_well 300 --recursive_valid_wells 0
python src/predict.py --data_dir data --model_dir model/smoke --output data/submission_smoke.csv
pytest
```
