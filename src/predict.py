from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Add src to sys.path to prevent shadowing from global packages
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dataset import load_well
from features import (
    InferenceFeatureBuilder,
    prepare_static_features,
    prepare_typewell_index,
)
from model import load_artifacts


def recursive_predict_well(model, feature_columns: list[str], well) -> tuple[pd.DataFrame, np.ndarray]:
    horizontal = well.horizontal.reset_index(drop=True)
    static_features = prepare_static_features(horizontal)
    typewell_index = prepare_typewell_index(well.typewell)
    tvt_work = horizontal["TVT_input"].to_numpy(dtype=float).copy()

    builder = InferenceFeatureBuilder(static_features, horizontal, typewell_index, feature_columns)

    records: list[dict[str, float | str]] = []
    for idx in range(len(horizontal)):
        if np.isfinite(tvt_work[idx]):
            continue
        
        features_array = builder.build_row(tvt_work, idx)
        pred = float(model.booster_.predict(features_array)[0])
        tvt_work[idx] = pred
        row_index = int(horizontal.loc[idx, "row_index"])
        records.append({"id": f"{well.well_id}_{row_index}", "tvt": pred})

    return pd.DataFrame(records), tvt_work



def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def predict_validation(data_dir: str | Path, model_dir: str | Path, output: str | Path) -> pd.DataFrame:
    model, feature_columns, _ = load_artifacts(model_dir)
    
    metrics_path = Path(model_dir) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path} for validation mode")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    
    if metrics.get("split") == "holdout":
        valid_ids = metrics.get("valid_wells", [])
    else:
        valid_ids = []
        for fold in metrics.get("folds", []):
            valid_ids.extend(fold.get("valid_wells", []))
        valid_ids = sorted(list(set(valid_ids)))
        
    if not valid_ids:
        raise ValueError("No validation wells found in metrics.json")

    predictions = []
    for well_id in tqdm(valid_ids, desc="Predicting validation wells"):
        well = load_well(data_dir, "train", well_id)
        
        true_tvt = well.horizontal[["row_index", "TVT"]].copy()
        well.horizontal.drop(columns=["TVT"], inplace=True)
        
        if well.horizontal["TVT_input"].isna().any():
            first_nan_idx = well.horizontal["TVT_input"].isna().idxmax()
            well.horizontal.loc[first_nan_idx:, "TVT_input"] = np.nan
            
        pred_df, _ = recursive_predict_well(model, feature_columns, well)
        
        truth = true_tvt.copy()
        truth["id"] = truth["row_index"].map(lambda idx: f"{well_id}_{int(idx)}")
        merged = pred_df.merge(truth[["id", "TVT"]], on="id", how="left")
        merged["well_id"] = well_id
        predictions.append(merged)

    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=["id", "tvt", "TVT", "well_id"])
    
    valid_mask = pred["TVT"].notna() & pred["tvt"].notna()
    local_rmse = rmse(pred.loc[valid_mask, "TVT"], pred.loc[valid_mask, "tvt"])
    print(f"Local RMSE: {local_rmse:.4f}")
    
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output, index=False)
    return pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive inference for ROGII TVT prediction.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/validation_predictions.csv")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    submission = predict_validation(args.data_dir, args.model_dir, args.output)
    print(f"Saved validation predictions to {args.output}")
    print(f"Rows: {len(submission)}, NaN tvt: {int(submission['tvt'].isna().sum())}")


if __name__ == "__main__":
    main()
