from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


MODEL_FILENAME = "model.joblib"
FEATURES_FILENAME = "feature_columns.json"
CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.json"


def get_lightgbm_regressor(random_state: int = 42, **overrides: Any):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("LightGBM is required. Install it with: pip install lightgbm") from exc

    params = {
        "objective": "huber",
        "metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update({key: value for key, value in overrides.items() if value is not None})
    return LGBMRegressor(**params)


def save_artifacts(
    model: Any,
    model_dir: str | Path,
    feature_columns: list[str],
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / MODEL_FILENAME)
    (path / FEATURES_FILENAME).write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    (path / CONFIG_FILENAME).write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (path / METRICS_FILENAME).write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


def load_artifacts(model_dir: str | Path):
    path = Path(model_dir)
    model = joblib.load(path / MODEL_FILENAME)
    feature_columns = json.loads((path / FEATURES_FILENAME).read_text(encoding="utf-8"))
    config = json.loads((path / CONFIG_FILENAME).read_text(encoding="utf-8"))
    return model, feature_columns, config
