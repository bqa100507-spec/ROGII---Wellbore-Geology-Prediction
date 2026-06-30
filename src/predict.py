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
from joblib import Parallel, delayed

from dataset import load_well
from features import (
    InferenceFeatureBuilder,
    prepare_static_features,
    prepare_typewell_index,
)
from model import load_artifacts


def predict_delta(model, features_array: np.ndarray) -> float:
    predictor = getattr(model, "booster_", model)
    return float(predictor.predict(features_array)[0])


def get_start_coords(horizontal: pd.DataFrame) -> tuple[float, float]:
    valid_xy = horizontal.dropna(subset=["X", "Y"])
    if len(valid_xy) == 0:
        return np.nan, np.nan
    return float(valid_xy.iloc[0]["X"]), float(valid_xy.iloc[0]["Y"])

def find_offset_wells(test_well, train_wells, k=3):
    test_x, test_y = get_start_coords(test_well.horizontal)
    if np.isnan(test_x) or np.isnan(test_y):
        return []

    distances = []
    for tw in train_wells:
        if tw.well_id == test_well.well_id:
            continue
        tw_x, tw_y = get_start_coords(tw.horizontal)
        if np.isnan(tw_x) or np.isnan(tw_y):
            continue
        dist = np.sqrt((tw_x - test_x)**2 + (tw_y - test_y)**2)
        distances.append((dist, tw))
        
    distances.sort(key=lambda x: x[0])
    return [tw for _, tw in distances[:k]]

def recursive_predict_well(model, feature_columns: list[str], well, train_wells=None) -> tuple[pd.DataFrame, np.ndarray]:
    horizontal = well.horizontal.reset_index(drop=True)
    static_features = prepare_static_features(horizontal)
    typewell_index = prepare_typewell_index(well.typewell)
    tvt_work = horizontal["TVT_input"].to_numpy(dtype=float).copy()
    row_indices = horizontal["row_index"].to_numpy(dtype=int)

    start_tvt_idx = horizontal["TVT_input"].last_valid_index()
    start_tvt = float(horizontal.loc[start_tvt_idx, "TVT_input"]) if start_tvt_idx is not None else 0.0

    builder = InferenceFeatureBuilder(static_features, horizontal, typewell_index, feature_columns, start_tvt)

    offset_wells = []
    if train_wells is not None:
        offset_wells = find_offset_wells(well, train_wells, k=3)

    test_md = horizontal["MD"].to_numpy(dtype=float)
    regional_dip_array = np.full(len(horizontal), np.nan)

    if offset_wells:
        interpolated_dips = []
        for ow in offset_wells:
            ow_horz = ow.horizontal
            if "MD" in ow_horz.columns and "TVT" in ow_horz.columns:
                delta_tvt = ow_horz["TVT"].diff().fillna(0).to_numpy(dtype=float)
                ow_md = ow_horz["MD"].to_numpy(dtype=float)
                valid_mask = np.isfinite(ow_md) & np.isfinite(delta_tvt)
                if valid_mask.sum() > 1:
                    ow_md_valid = ow_md[valid_mask]
                    delta_tvt_valid = delta_tvt[valid_mask]
                    sort_idx = np.argsort(ow_md_valid)
                    ow_md_valid = ow_md_valid[sort_idx]
                    delta_tvt_valid = delta_tvt_valid[sort_idx]
                    
                    interp_dip = np.interp(test_md, ow_md_valid, delta_tvt_valid, left=np.nan, right=np.nan)
                    interpolated_dips.append(interp_dip)
                    
        if interpolated_dips:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                regional_dip_array = np.nanmean(np.vstack(interpolated_dips), axis=0)

    records: list[dict[str, float | str]] = []
    for idx in range(len(horizontal)):
        if np.isfinite(tvt_work[idx]):
            continue
        
        regional_dip = regional_dip_array[idx]
        
        features_array = builder.build_row(tvt_work, idx)
        predicted_delta = predict_delta(model, features_array)
        
        if np.isfinite(regional_dip):
            clipped_delta = float(np.clip(predicted_delta, regional_dip - 0.15, regional_dip + 0.15))
        else:
            clipped_delta = float(np.clip(predicted_delta, -0.3, 0.3))
        
        new_tvt = tvt_work[idx - 1] + clipped_delta
        tvt_work[idx] = new_tvt
        records.append({"id": f"{well.well_id}_{row_indices[idx]}", "tvt": new_tvt})

    return pd.DataFrame(records), tvt_work



def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def predict_single_well(well_id: str, data_dir: str | Path, model, feature_columns: list[str], train_wells: list) -> pd.DataFrame | None:
    well = load_well(data_dir, "train", well_id)
    
    true_tvt = well.horizontal[["row_index", "TVT"]].copy()
    well.horizontal.drop(columns=["TVT"], inplace=True)
    
    if well.horizontal["TVT_input"].isna().any():
        first_nan_idx = well.horizontal["TVT_input"].isna().idxmax()
        well.horizontal.loc[first_nan_idx:, "TVT_input"] = np.nan
        
    pred_df, _ = recursive_predict_well(model, feature_columns, well, train_wells)
    
    truth = true_tvt.copy()
    truth["id"] = truth["row_index"].map(lambda idx: f"{well_id}_{int(idx)}")
    merged = pred_df.merge(truth[["id", "TVT"]], on="id", how="left")
    merged["well_id"] = well_id
    return merged


def predict_validation(data_dir: str | Path, model_dir: str | Path, output: str | Path) -> pd.DataFrame:
    model, feature_columns, _ = load_artifacts(model_dir)
    
    metrics_path = Path(model_dir) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path} for validation mode")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    
    if metrics.get("split") == "holdout":
        all_train_ids = metrics.get("train_wells", [])
        valid_ids = metrics.get("valid_wells", [])
    else:
        all_train_ids = metrics.get("wells", [])
        valid_ids = []
        for fold in metrics.get("folds", []):
            valid_ids.extend(fold.get("valid_wells", []))
        valid_ids = sorted(list(set(valid_ids)))
        
    if not valid_ids:
        raise ValueError("No validation wells found in metrics.json")

    print("Loading all train wells for offset matching...")
    # Import load_wells here to avoid circular imports if dataset relies on predict
    from dataset import load_wells
    train_wells = load_wells(data_dir, "train", all_train_ids)

    predictions = Parallel(n_jobs=-2)(
        delayed(predict_single_well)(well_id, data_dir, model, feature_columns, train_wells)
        for well_id in tqdm(valid_ids, desc="Predicting validation wells")
    )
    
    # Filter out None if any
    predictions = [p for p in predictions if p is not None]

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
