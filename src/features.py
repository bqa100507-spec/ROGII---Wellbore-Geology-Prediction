from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


BASE_INPUT_COLUMNS = ["MD", "X", "Y", "Z", "GR"]
HISTORY_COLUMNS = ["TVT", "X", "Y", "Z", "GR"]
LAG_STEPS = [1, 2, 3, 5, 10, 20, 50, 100]
ROLLING_WINDOWS = [5, 10, 20, 50, 100]
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
    
    out["GR_diff"] = out["GR"].diff().fillna(0)
    
    out["GR_static_mean_10"] = out["GR"].rolling(10, min_periods=1).mean().fillna(0)
    out["GR_static_std_10"] = out["GR"].rolling(10, min_periods=2).std().fillna(0)
    out["GR_static_mean_50"] = out["GR"].rolling(50, min_periods=1).mean().fillna(0)
    out["GR_static_std_50"] = out["GR"].rolling(50, min_periods=2).std().fillna(0)
    
    out["GR_lookahead_10"] = out["GR"].shift(-10).fillna(0)
    out["GR_lookahead_30"] = out["GR"].shift(-30).fillna(0)
    
    out["GR_lookback_10"] = out["GR"].shift(10).fillna(0)
    out["GR_lookback_30"] = out["GR"].shift(30).fillna(0)
    
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
    is_training: bool = False,
) -> pd.DataFrame:
    static = prepare_static_features(horizontal)
    history = pd.DataFrame(index=horizontal.index)
    tvt_series = pd.Series(tvt_history, index=horizontal.index, dtype=float)
    history["TVT"] = tvt_series.copy()
    if is_training:
        history["TVT"] += np.random.normal(0, 1.0, size=len(history["TVT"]))
        
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
    expected_tvt = tvt_series.shift(1)
    parts.append(make_typewell_features(static["GR"], expected_tvt, typewell_index))
    
    start_tvt = tvt_series.iloc[0]
    current_drift = expected_tvt - start_tvt
    parts.append(current_drift.rename("Current_Drift"))
    
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
    features = build_feature_frame(horizontal, typewell, horizontal["TVT"], is_training=True)
    
    delta_tvt = horizontal["TVT"].diff().fillna(0)
    
    mask = target_mask(horizontal, scope=scope)
    mask &= delta_tvt.notna()
    
    return features.loc[mask].reset_index(drop=True), delta_tvt.loc[mask].reset_index(drop=True)


def _array_stats(values: np.ndarray) -> tuple[float, float]:
    total = 0.0
    count = 0
    for val in values:
        if math.isfinite(val):
            total += val
            count += 1
            
    if count == 0:
        return np.nan, np.nan
        
    mean = total / count
    if count == 1:
        return mean, np.nan
        
    sq_diff = 0.0
    for val in values:
        if math.isfinite(val):
            sq_diff += (val - mean) ** 2
            
    std = math.sqrt(sq_diff / (count - 1))
    return mean, std

def _rolling_stats(values: np.ndarray, idx: int, window: int) -> tuple[float, float]:
    start = max(0, idx - window)
    hist = values[start:idx]
    return _array_stats(hist)


def make_single_row_features(
    static_features: pd.DataFrame,
    horizontal: pd.DataFrame,
    typewell_index: TypewellIndex,
    tvt_work: np.ndarray,
    idx: int,
    start_tvt: float,
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
    row["Current_Drift"] = expected - start_tvt
    
    current_gr = row.get("GR", np.nan)
    tw_features = make_typewell_features(
        pd.Series([current_gr]),
        pd.Series([expected]),
        typewell_index,
    )
    row.update(tw_features.iloc[0].to_dict())
    return pd.DataFrame([row])


class InferenceFeatureBuilder:
    def __init__(self, static_features: pd.DataFrame, horizontal: pd.DataFrame, typewell_index: TypewellIndex, feature_columns: list[str], start_tvt: float):
        self.static_dict = {col: static_features[col].to_numpy() for col in static_features.columns}
        self.history_arrays = {
            "X": horizontal["X"].to_numpy(dtype=float) if "X" in horizontal.columns else np.full(len(horizontal), np.nan),
            "Y": horizontal["Y"].to_numpy(dtype=float) if "Y" in horizontal.columns else np.full(len(horizontal), np.nan),
            "Z": horizontal["Z"].to_numpy(dtype=float) if "Z" in horizontal.columns else np.full(len(horizontal), np.nan),
            "GR": horizontal["GR"].to_numpy(dtype=float) if "GR" in horizontal.columns else np.full(len(horizontal), np.nan),
        }
        self.typewell_index = typewell_index
        self.feature_columns = feature_columns
        self.tw_tvt = typewell_index.tvt
        self.tw_gr = typewell_index.gr
        self.start_tvt = start_tvt
        
    def build_row(self, tvt_work: np.ndarray, idx: int) -> np.ndarray:
        row = {}
        for col, arr in self.static_dict.items():
            row[col] = arr[idx]
            
        self.history_arrays["TVT"] = tvt_work
        
        for col in ["TVT", "X", "Y", "Z", "GR"]:
            values = self.history_arrays[col]
            for lag in LAG_STEPS:
                row[f"{col}_lag_{lag}"] = values[idx - lag] if idx - lag >= 0 else np.nan
            for window in ROLLING_WINDOWS:
                mean, std = _rolling_stats(values, idx, window)
                row[f"{col}_roll_mean_{window}"] = mean
                row[f"{col}_roll_std_{window}"] = std
                
        expected = tvt_work[idx - 1] if idx > 0 else np.nan
        row["Current_Drift"] = expected - self.start_tvt
        
        current_gr = row.get("GR", np.nan)
        
        if np.isfinite(expected) and len(self.tw_tvt) > 0:
            pos = np.searchsorted(self.tw_tvt, expected, side="left")
            right = np.clip(pos, 0, len(self.tw_tvt) - 1)
            left = np.clip(pos - 1, 0, len(self.tw_tvt) - 1)
            if abs(expected - self.tw_tvt[left]) <= abs(expected - self.tw_tvt[right]):
                nearest = left
            else:
                nearest = right
            
            nearest_gr = self.tw_gr[nearest]
            row["expected_tvt"] = expected
            row["typewell_gr_at_nearest_tvt"] = nearest_gr
            row["gr_minus_typewell_gr"] = current_gr - nearest_gr
            row["nearest_typewell_tvt_gap"] = expected - self.tw_tvt[nearest]
            
            for window in TYPEWELL_WINDOWS:
                half = window // 2
                start = max(0, nearest - half)
                end = min(len(self.tw_gr), nearest + half + 1)
                values = self.tw_gr[start:end]
                mean, std = _array_stats(values)
                row[f"typewell_gr_roll_mean_{window}"] = mean
                row[f"typewell_gr_roll_std_{window}"] = std
        else:
            row["expected_tvt"] = expected
            row["typewell_gr_at_nearest_tvt"] = np.nan
            row["gr_minus_typewell_gr"] = np.nan
            row["nearest_typewell_tvt_gap"] = np.nan
            for window in TYPEWELL_WINDOWS:
                row[f"typewell_gr_roll_mean_{window}"] = np.nan
                row[f"typewell_gr_roll_std_{window}"] = np.nan
                
        out = np.zeros((1, len(self.feature_columns)), dtype=float)
        for i, col in enumerate(self.feature_columns):
            out[0, i] = row.get(col, np.nan)
        return out



def ensure_feature_columns(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    columns = list(feature_columns)
    aligned = frame.copy()
    for col in columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned[columns]
