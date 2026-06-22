from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import get_test_well_ids, load_well, read_submission_template
from features import (
    ensure_feature_columns,
    make_single_row_features,
    prepare_static_features,
    prepare_typewell_index,
)
from model import load_artifacts


def recursive_predict_well(model, feature_columns: list[str], well) -> tuple[pd.DataFrame, np.ndarray]:
    horizontal = well.horizontal.reset_index(drop=True)
    static_features = prepare_static_features(horizontal)
    typewell_index = prepare_typewell_index(well.typewell)
    tvt_work = horizontal["TVT_input"].to_numpy(dtype=float).copy()

    records: list[dict[str, float | str]] = []
    for idx in range(len(horizontal)):
        if np.isfinite(tvt_work[idx]):
            continue
        features = make_single_row_features(static_features, horizontal, typewell_index, tvt_work, idx)
        features = ensure_feature_columns(features, feature_columns)
        pred = float(model.predict(features)[0])
        tvt_work[idx] = pred
        row_index = int(horizontal.loc[idx, "row_index"])
        records.append({"id": f"{well.well_id}_{row_index}", "tvt": pred})

    return pd.DataFrame(records), tvt_work


def predict_submission(data_dir: str | Path, model_dir: str | Path, output: str | Path) -> pd.DataFrame:
    model, feature_columns, _ = load_artifacts(model_dir)
    template = read_submission_template(data_dir)
    test_ids = get_test_well_ids(data_dir)

    predictions = []
    for well_id in tqdm(test_ids, desc="Predicting test wells"):
        well = load_well(data_dir, "test", well_id)
        pred_df, _ = recursive_predict_well(model, feature_columns, well)
        predictions.append(pred_df)

    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=["id", "tvt"])
    merged = template[["id"]].merge(pred, on="id", how="left", validate="one_to_one")
    if merged["tvt"].isna().any():
        missing = merged.loc[merged["tvt"].isna(), "id"].head(10).tolist()
        raise ValueError(f"Missing predictions for submission ids: {missing}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive inference for ROGII TVT prediction.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission = predict_submission(args.data_dir, args.model_dir, args.output)
    print(f"Saved submission to {args.output}")
    print(f"Rows: {len(submission)}, NaN tvt: {int(submission['tvt'].isna().sum())}")


if __name__ == "__main__":
    main()
