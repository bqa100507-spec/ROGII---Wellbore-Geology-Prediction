from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, train_test_split
from tqdm import tqdm

from dataset import get_test_well_ids, list_well_ids, load_well
from features import build_training_matrix, ensure_feature_columns
from model import get_lightgbm_regressor, save_artifacts
from predict import recursive_predict_well


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eligible_train_ids(data_dir: str | Path, exclude_test_ids: bool = True) -> list[str]:
    ids = list_well_ids(data_dir, "train")
    if exclude_test_ids:
        test_ids = set(get_test_well_ids(data_dir))
        ids = [well_id for well_id in ids if well_id not in test_ids]
    return ids


def build_matrix_for_wells(
    data_dir: str | Path,
    well_ids: list[str],
    scope: str,
    max_rows_per_well: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    groups: list[np.ndarray] = []

    for well_id in tqdm(well_ids, desc="Building features"):
        well = load_well(data_dir, "train", well_id)
        x_well, y_well = build_training_matrix(well.horizontal, well.typewell, scope=scope)
        if max_rows_per_well is not None and len(x_well) > max_rows_per_well:
            x_well = x_well.tail(max_rows_per_well).reset_index(drop=True)
            y_well = y_well.tail(max_rows_per_well).reset_index(drop=True)
        x_parts.append(x_well)
        y_parts.append(y_well)
        groups.append(np.repeat(well_id, len(x_well)))

    if not x_parts:
        raise ValueError("No training wells were selected.")
    x = pd.concat(x_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    group_values = np.concatenate(groups)
    return x, y, group_values


def recursive_validate(
    model,
    feature_columns: list[str],
    data_dir: str | Path,
    well_ids: list[str],
    max_wells: int | None = None,
) -> dict[str, Any]:
    if max_wells is not None and max_wells <= 0:
        return {
            "recursive_rmse": None,
            "rows": 0,
            "wells": [],
            "skipped_wells": well_ids,
            "by_well": {},
        }
    selected_ids = well_ids[:max_wells] if max_wells is not None else well_ids
    rows: list[pd.DataFrame] = []
    for well_id in tqdm(selected_ids, desc="Recursive validation"):
        well = load_well(data_dir, "train", well_id)
        pred_df, _ = recursive_predict_well(model, feature_columns, well)
        truth = well.horizontal[["row_index", "TVT"]].copy()
        truth["id"] = truth["row_index"].map(lambda idx: f"{well_id}_{int(idx)}")
        merged = pred_df.merge(truth[["id", "TVT"]], on="id", how="left")
        merged["well_id"] = well_id
        rows.append(merged)

    if not rows:
        return {"recursive_rmse": None, "rows": 0, "by_well": {}}
    frame = pd.concat(rows, ignore_index=True)
    score = rmse(frame["TVT"], frame["tvt"])
    by_well = {
        well_id: rmse(group["TVT"], group["tvt"])
        for well_id, group in frame.groupby("well_id")
    }
    return {
        "recursive_rmse": score,
        "rows": int(len(frame)),
        "wells": selected_ids,
        "skipped_wells": well_ids[len(selected_ids) :],
        "by_well": by_well,
    }


def fit_lgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    args: argparse.Namespace,
):
    model = get_lightgbm_regressor(
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
    )
    fit_kwargs: dict[str, Any] = {}
    if x_valid is not None and y_valid is not None and len(x_valid):
        try:
            from lightgbm import early_stopping, log_evaluation

            fit_kwargs["eval_set"] = [(x_valid, y_valid)]
            fit_kwargs["eval_metric"] = "rmse"
            fit_kwargs["callbacks"] = [
                early_stopping(args.early_stopping_rounds, verbose=False),
                log_evaluation(args.log_period),
            ]
        except ImportError:
            pass
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def run_holdout(args: argparse.Namespace, all_ids: list[str]) -> tuple[Any, list[str], dict[str, Any]]:
    train_ids, valid_ids = train_test_split(
        all_ids,
        test_size=args.valid_size,
        random_state=args.random_state,
        shuffle=True,
    )
    if args.max_wells is not None:
        train_ids = train_ids[: max(1, int(args.max_wells * (1 - args.valid_size)))]
        valid_ids = valid_ids[: max(1, args.max_wells - len(train_ids))]

    x_train, y_train, _ = build_matrix_for_wells(
        args.data_dir,
        train_ids,
        args.train_scope,
        max_rows_per_well=args.max_train_rows_per_well,
    )
    x_valid, y_valid, _ = build_matrix_for_wells(
        args.data_dir,
        valid_ids,
        args.train_scope,
        max_rows_per_well=args.max_train_rows_per_well,
    )
    feature_columns = list(x_train.columns)
    x_valid = ensure_feature_columns(x_valid, feature_columns)
    model = fit_lgbm(x_train, y_train, x_valid, y_valid, args)

    valid_pred = model.predict(x_valid)
    metrics = {
        "split": "holdout",
        "train_wells": train_ids,
        "valid_wells": valid_ids,
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "teacher_forced_rmse": rmse(y_valid, valid_pred),
        "recursive_validation": recursive_validate(
            model,
            feature_columns,
            args.data_dir,
            valid_ids,
            max_wells=args.recursive_valid_wells,
        ),
    }
    return model, feature_columns, metrics


def run_groupkfold(args: argparse.Namespace, all_ids: list[str]) -> tuple[Any, list[str], dict[str, Any]]:
    ids = all_ids[: args.max_wells] if args.max_wells is not None else all_ids
    x_all, y_all, groups = build_matrix_for_wells(
        args.data_dir,
        ids,
        args.train_scope,
        max_rows_per_well=args.max_train_rows_per_well,
    )
    feature_columns = list(x_all.columns)
    splitter = GroupKFold(n_splits=args.n_splits)
    fold_metrics = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_all, y_all, groups), start=1):
        print(f"Training fold {fold}/{args.n_splits}")
        x_train, y_train = x_all.iloc[train_idx], y_all.iloc[train_idx]
        x_valid, y_valid = x_all.iloc[valid_idx], y_all.iloc[valid_idx]
        model = fit_lgbm(x_train, y_train, x_valid, y_valid, args)
        pred = model.predict(x_valid)
        
        valid_wells = sorted(set(groups[valid_idx]))
        fold_metrics.append(
            {
                "fold": fold,
                "valid_wells": valid_wells,
                "train_rows": len(train_idx),
                "valid_rows": int(len(valid_idx)),
                "teacher_forced_rmse": rmse(y_valid, pred),
            }
        )

    print("Training final model on all selected non-test wells")
    final_model = fit_lgbm(x_all, y_all, None, None, args)
    scores = [item["teacher_forced_rmse"] for item in fold_metrics]
    metrics = {
        "split": "groupkfold",
        "n_splits": args.n_splits,
        "wells": ids,
        "rows": int(len(x_all)),
        "folds": fold_metrics,
        "mean_teacher_forced_rmse": float(np.mean(scores)),
        "std_teacher_forced_rmse": float(np.std(scores)),
    }
    return final_model, feature_columns, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autoregressive LightGBM for ROGII TVT prediction.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--exp_name", type=str, default="lgbm_baseline")
    parser.add_argument("--split", choices=["holdout", "groupkfold"], default="holdout")
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--train_scope", choices=["prediction", "all", "known"], default="prediction")
    parser.add_argument("--include_test_ids", action="store_true", help="Allow leakage-prone training on current test IDs.")
    parser.add_argument("--max_wells", type=int, default=None, help="Debug: limit number of train wells.")
    parser.add_argument("--max_train_rows_per_well", type=int, default=None, help="Debug: keep only the tail rows per well.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)
    parser.add_argument("--log_period", type=int, default=100)
    parser.add_argument(
        "--recursive_valid_wells",
        type=int,
        default=3,
        help="Number of validation wells for slow recursive validation. Use 0 to skip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_ids = eligible_train_ids(args.data_dir, exclude_test_ids=not args.include_test_ids)
    if args.max_wells is not None:
        all_ids = all_ids[: args.max_wells]
    print(f"Selected train wells: {len(all_ids)}")
    print(f"Excluded test IDs: {not args.include_test_ids}")

    if args.split == "holdout":
        model, feature_columns, metrics = run_holdout(args, all_ids)
    else:
        model, feature_columns, metrics = run_groupkfold(args, all_ids)

    config = vars(args)
    config["excluded_test_well_ids"] = [] if args.include_test_ids else get_test_well_ids(args.data_dir)
    model_dir = Path("model") / args.exp_name
    save_artifacts(model, model_dir, feature_columns, config, metrics)
    print(f"Saved model artifacts to {model_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
