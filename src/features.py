from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


BASE_INPUT_COLUMNS = ["MD", "X", "Y", "Z", "GR"]
HISTORY_COLUMNS = ["TVT", "X", "Y", "Z", "GR"]
LAG_STEPS = [1, 2, 3, 5, 10, 20]
ROLLING_WINDOWS = [5, 10, 20]
TYPEWELL_WINDOWS = [5, 10, 20]


@dataclass(frozen=True)
class TypewellIndex:
    tvt: np.ndarray
    gr: np.ndarray


def prepare_typewell_index(typewell: pd.DataFrame) -> TypewellIndex:
    tw = typewell[["TVT", "GR"]].dropna(subset=["TVT"]).sort_values("TVT")
    return TypewellIndex(
        tvt=tw["TVT"].to_numpy(dtype=float),
        gr=tw["GR"].to_numpy(dtype=float),
    )


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_static_features(horizontal: pd.DataFrame) -> pd.DataFrame:
    df = horizontal.copy()
    out = pd.DataFrame(index=df.index)
    out["row_position"] = np.arange(len(df), dtype=float)

    for col in BASE_INPUT_COLUMNS:
        out[col] = _safe_numeric(df[col]) if col in df.columns else np.nan

    out["md_from_start"] = out["MD"] - out["MD"].iloc[0]
    out["dX"] = out["X"].diff()
    out["dY"] = out["Y"].diff()
    out["dZ"] = out["Z"].diff()
    out["dMD"] = out["MD"].diff()
    out["xy_step_distance"] = np.sqrt(out["dX"] ** 2 + out["dY"] ** 2)
    out["step_distance"] = np.sqrt(out["dX"] ** 2 + out["dY"] ** 2 + out["dZ"] ** 2)
    out["path_distance"] = out["step_distance"].fillna(0).cumsum()
    out["dZ_dMD"] = out["dZ"] / out["dMD"].replace(0, np.nan)
    return out


def _nearest_indices(typewell_index: TypewellIndex, expected_tvt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tvt = typewell_index.tvt
    nearest = np.full(len(expected_tvt), -1, dtype=int)
    gap = np.full(len(expected_tvt), np.nan, dtype=float)
    if len(tvt) == 0:
        return nearest, gap

    valid = np.isfinite(expected_tvt)
    pos = np.searchsorted(tvt, expected_tvt[valid], side="left")
    right = np.clip(pos, 0, len(tvt) - 1)
    left = np.clip(pos - 1, 0, len(tvt) - 1)
    use_left = np.abs(expected_tvt[valid] - tvt[left]) <= np.abs(expected_tvt[valid] - tvt[right])
    chosen = np.where(use_left, left, right)
    nearest[np.where(valid)[0]] = chosen
    gap[np.where(valid)[0]] = expected_tvt[valid] - tvt[chosen]
    return nearest, gap


def _typewell_window_stats(typewell_index: TypewellIndex, nearest: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full(len(nearest), np.nan, dtype=float)
    std = np.full(len(nearest), np.nan, dtype=float)
    gr = typewell_index.gr
    if len(gr) == 0:
        return mean, std

    half = window // 2
    for row, idx in enumerate(nearest):
        if idx < 0:
            continue
        start = max(0, idx - half)
        end = min(len(gr), idx + half + 1)
        values = gr[start:end]
        if np.isfinite(values).any():
            mean[row] = np.nanmean(values)
            finite_count = np.isfinite(values).sum()
            std[row] = np.nanstd(values, ddof=1) if finite_count > 1 else np.nan
    return mean, std


def make_typewell_features(
    current_gr: pd.Series,
    expected_tvt: pd.Series,
    typewell_index: TypewellIndex,
) -> pd.DataFrame:
    expected = expected_tvt.to_numpy(dtype=float)
    nearest, gap = _nearest_indices(typewell_index, expected)
    out = pd.DataFrame(index=expected_tvt.index)
    out["expected_tvt"] = expected_tvt

    nearest_gr = np.full(len(expected), np.nan, dtype=float)
    valid = nearest >= 0
    if len(typewell_index.gr):
        nearest_gr[valid] = typewell_index.gr[nearest[valid]]
    out["typewell_gr_at_nearest_tvt"] = nearest_gr
    out["gr_minus_typewell_gr"] = current_gr.to_numpy(dtype=float) - nearest_gr
    out["nearest_typewell_tvt_gap"] = gap

    for window in TYPEWELL_WINDOWS:
        mean, std = _typewell_window_stats(typewell_index, nearest, window)
        out[f"typewell_gr_roll_mean_{window}"] = mean
        out[f"typewell_gr_roll_std_{window}"] = std
    return out


def build_feature_frame(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    tvt_history: pd.Series | np.ndarray,
) -> pd.DataFrame:
    static = prepare_static_features(horizontal)
    history = pd.DataFrame(index=horizontal.index)
    history["TVT"] = pd.Series(tvt_history, index=horizontal.index, dtype=float)
    for col in ["X", "Y", "Z", "GR"]:
        history[col] = _safe_numeric(horizontal[col]) if col in horizontal.columns else np.nan

    parts = [static]
    for col in HISTORY_COLUMNS:
        for lag in LAG_STEPS:
            parts.append(history[col].shift(lag).rename(f"{col}_lag_{lag}"))
        shifted = history[col].shift(1)
        for window in ROLLING_WINDOWS:
            parts.append(shifted.rolling(window, min_periods=1).mean().rename(f"{col}_roll_mean_{window}"))
            parts.append(shifted.rolling(window, min_periods=2).std().rename(f"{col}_roll_std_{window}"))

    typewell_index = prepare_typewell_index(typewell)
    expected_tvt = history["TVT"].shift(1)
    parts.append(make_typewell_features(static["GR"], expected_tvt, typewell_index))
    return pd.concat(parts, axis=1)


def target_mask(horizontal: pd.DataFrame, scope: str = "prediction") -> pd.Series:
    if "TVT" not in horizontal.columns:
        return pd.Series(False, index=horizontal.index)
    mask = horizontal["TVT"].notna()
    if scope == "prediction":
        if "TVT_input" not in horizontal.columns:
            raise ValueError("prediction scope requires TVT_input")
        mask &= horizontal["TVT_input"].isna()
    elif scope == "known":
        if "TVT_input" not in horizontal.columns:
            raise ValueError("known scope requires TVT_input")
        mask &= horizontal["TVT_input"].notna()
    elif scope != "all":
        raise ValueError(f"Unknown train scope: {scope}")
    return mask


def build_training_matrix(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    scope: str = "prediction",
) -> tuple[pd.DataFrame, pd.Series]:
    if "TVT" not in horizontal.columns:
        raise ValueError("Training horizontal data must contain TVT")
    features = build_feature_frame(horizontal, typewell, horizontal["TVT"])
    mask = target_mask(horizontal, scope=scope)
    return features.loc[mask].reset_index(drop=True), horizontal.loc[mask, "TVT"].reset_index(drop=True)


def _rolling_stats(values: np.ndarray, idx: int, window: int) -> tuple[float, float]:
    start = max(0, idx - window)
    hist = values[start:idx]
    if len(hist) == 0 or not np.isfinite(hist).any():
        return np.nan, np.nan
    mean = float(np.nanmean(hist))
    finite_count = int(np.isfinite(hist).sum())
    std = float(np.nanstd(hist, ddof=1)) if finite_count > 1 else np.nan
    return mean, std


def make_single_row_features(
    static_features: pd.DataFrame,
    horizontal: pd.DataFrame,
    typewell_index: TypewellIndex,
    tvt_work: np.ndarray,
    idx: int,
) -> pd.DataFrame:
    row = static_features.iloc[idx].to_dict()
    history_arrays = {
        "TVT": tvt_work.astype(float, copy=False),
        "X": horizontal["X"].to_numpy(dtype=float),
        "Y": horizontal["Y"].to_numpy(dtype=float),
        "Z": horizontal["Z"].to_numpy(dtype=float),
        "GR": horizontal["GR"].to_numpy(dtype=float),
    }

    for col, values in history_arrays.items():
        for lag in LAG_STEPS:
            row[f"{col}_lag_{lag}"] = values[idx - lag] if idx - lag >= 0 else np.nan
        for window in ROLLING_WINDOWS:
            mean, std = _rolling_stats(values, idx, window)
            row[f"{col}_roll_mean_{window}"] = mean
            row[f"{col}_roll_std_{window}"] = std

    expected = tvt_work[idx - 1] if idx > 0 else np.nan
    current_gr = row.get("GR", np.nan)
    tw_features = make_typewell_features(
        pd.Series([current_gr]),
        pd.Series([expected]),
        typewell_index,
    )
    row.update(tw_features.iloc[0].to_dict())
    return pd.DataFrame([row])


def ensure_feature_columns(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    columns = list(feature_columns)
    aligned = frame.copy()
    for col in columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned[columns]
