from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import WellData, load_well


TRAIN_ONLY_COLUMNS = {"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}
DIAGNOSTIC_COLUMNS = [
    "id",
    "well_id",
    "row_index",
    "pred_tvt",
    "corrected_tvt",
    "raw_offset",
    "smooth_offset",
    "confidence",
    "gr_cost_before",
    "gr_cost_after",
    "texture_score",
    "uniqueness_score",
    "used_correction",
]


@dataclass(frozen=True)
class PhaseLockConfig:
    enabled: bool = True
    offset_min: float = -6.0
    offset_max: float = 6.0
    offset_step: float = 0.5
    window_size: int = 31
    min_valid_window: int = 8
    min_texture: float = 0.25
    min_uniqueness: float = 0.30
    min_improvement: float = 0.15
    min_confidence: float = 0.75
    alpha: float = 0.15
    ewma_alpha: float = 0.20
    max_abs_correction: float = 1.5
    max_step_change: float = 0.15
    huber_delta: float = 1.5


def parse_prediction_id(prediction_id: str) -> tuple[str, int]:
    text = str(prediction_id)
    if "_" not in text:
        raise ValueError(f"Prediction id must be '<well_id>_<row_index>': {prediction_id!r}")
    well_id, row_text = text.rsplit("_", 1)
    if not well_id or not row_text:
        raise ValueError(f"Prediction id must be '<well_id>_<row_index>': {prediction_id!r}")
    try:
        row_index = int(row_text)
    except ValueError as exc:
        raise ValueError(f"Prediction id has non-integer row_index: {prediction_id!r}") from exc
    if row_index < 0:
        raise ValueError(f"Prediction id has negative row_index: {prediction_id!r}")
    return well_id, row_index


def _offset_grid(config: PhaseLockConfig) -> np.ndarray:
    if config.offset_step <= 0:
        raise ValueError("offset_step must be positive")
    if config.offset_min > config.offset_max:
        raise ValueError("offset_min must be <= offset_max")
    count = int(np.floor((config.offset_max - config.offset_min) / config.offset_step + 0.5)) + 1
    offsets = config.offset_min + np.arange(count, dtype=float) * config.offset_step
    offsets = offsets[offsets <= config.offset_max + 1e-9]
    if not np.any(np.isclose(offsets, 0.0)):
        offsets = np.sort(np.concatenate([offsets, np.array([0.0])]))
    return offsets.astype(float, copy=False)


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return out
    center = float(np.nanmedian(arr[valid]))
    q75, q25 = np.nanpercentile(arr[valid], [75.0, 25.0])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(arr[valid]))
    if not np.isfinite(scale) or scale < 1e-6:
        out[valid] = 0.0
        return out
    out[valid] = (arr[valid] - center) / scale
    return np.clip(out, -8.0, 8.0)


def _prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tw = typewell[["TVT", "GR"]].copy()
    tvt = tw["TVT"].to_numpy(dtype=float)
    gr = _robust_normalize(tw["GR"].to_numpy(dtype=float))
    valid = np.isfinite(tvt) & np.isfinite(gr)
    if valid.sum() < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    tvt = tvt[valid]
    gr = gr[valid]
    order = np.argsort(tvt, kind="mergesort")
    tvt = tvt[order]
    gr = gr[order]
    unique_tvt, unique_idx = np.unique(tvt, return_index=True)
    return unique_tvt.astype(float, copy=False), gr[unique_idx].astype(float, copy=False)


def _huber_loss(residual: np.ndarray, delta: float) -> np.ndarray:
    abs_resid = np.abs(residual)
    quadratic = np.minimum(abs_resid, delta)
    linear = abs_resid - quadratic
    return 0.5 * quadratic * quadratic + delta * linear


def _window_bounds(center: int, length: int, window_size: int) -> tuple[int, int]:
    size = max(1, int(window_size))
    half = size // 2
    lo = max(0, center - half)
    hi = min(length, center + half + 1)
    return lo, hi


def _score_offsets(
    path_tvt: np.ndarray,
    horizontal_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    row_index: int,
    offsets: np.ndarray,
    config: PhaseLockConfig,
) -> tuple[float, float, float, float, float, bool, float]:
    n_rows = len(path_tvt)
    if row_index < 0 or row_index >= n_rows or len(typewell_tvt) < 2:
        return 0.0, 0.0, np.nan, np.nan, 0.0, False, 0.0

    lo, hi = _window_bounds(row_index, n_rows, config.window_size)
    tvt_window = path_tvt[lo:hi]
    gr_window = horizontal_gr[lo:hi]
    valid = np.isfinite(tvt_window) & np.isfinite(gr_window)
    if valid.sum() < config.min_valid_window:
        return 0.0, 0.0, np.nan, np.nan, 0.0, False, 0.0

    tvt_valid = tvt_window[valid]
    gr_valid = gr_window[valid]
    texture_score = float(np.nanstd(gr_valid))
    if not np.isfinite(texture_score):
        texture_score = 0.0

    queries = tvt_valid[None, :] + offsets[:, None]
    interp = np.interp(queries.ravel(), typewell_tvt, typewell_gr, left=np.nan, right=np.nan).reshape(
        len(offsets), len(tvt_valid)
    )
    finite = np.isfinite(interp)
    counts = finite.sum(axis=1)
    residual = interp - gr_valid[None, :]
    loss = _huber_loss(residual, config.huber_delta)
    loss[~finite] = 0.0
    costs = np.full(len(offsets), np.inf, dtype=float)
    enough = counts >= config.min_valid_window
    costs[enough] = loss[enough].sum(axis=1) / counts[enough]

    zero_idx = int(np.argmin(np.abs(offsets)))
    cost_before = float(costs[zero_idx]) if np.isfinite(costs[zero_idx]) else np.nan
    if not np.isfinite(costs).any():
        return 0.0, 0.0, cost_before, np.nan, texture_score, False, 0.0

    best_idx = int(np.nanargmin(costs))
    raw_offset = float(offsets[best_idx])
    cost_after = float(costs[best_idx])
    finite_costs = np.sort(costs[np.isfinite(costs)])
    if len(finite_costs) >= 2:
        second_best = float(finite_costs[1])
        uniqueness_score = max(0.0, (second_best - cost_after) / (abs(cost_after) + 1e-6))
    else:
        uniqueness_score = 0.0

    if np.isfinite(cost_before) and cost_before > 1e-6:
        improvement = max(0.0, (cost_before - cost_after) / (abs(cost_before) + 1e-6))
    else:
        improvement = 0.0

    texture_conf = min(1.0, texture_score / max(config.min_texture * 2.0, 1e-6))
    uniqueness_conf = min(1.0, uniqueness_score / max(config.min_uniqueness, 1e-6))
    improvement_conf = min(1.0, improvement / max(config.min_improvement, 1e-6))
    confidence = float(min(texture_conf, uniqueness_conf, improvement_conf))
    used = (
        texture_score >= config.min_texture
        and uniqueness_score >= config.min_uniqueness
        and improvement >= config.min_improvement
        and confidence >= config.min_confidence
        and abs(raw_offset) > 1e-9
    )
    return raw_offset, confidence, cost_before, cost_after, texture_score, used, uniqueness_score


def _rolling_costs_by_offset(
    path_tvt: np.ndarray,
    horizontal_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    offsets: np.ndarray,
    config: PhaseLockConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(path_tvt)
    if n_rows == 0 or len(typewell_tvt) < 2:
        return np.empty((len(offsets), n_rows), dtype=float), np.zeros(n_rows, dtype=float)

    queries = path_tvt[None, :] + offsets[:, None]
    interp = np.interp(queries.ravel(), typewell_tvt, typewell_gr, left=np.nan, right=np.nan).reshape(
        len(offsets), n_rows
    )
    valid = np.isfinite(interp) & np.isfinite(horizontal_gr)[None, :] & np.isfinite(path_tvt)[None, :]
    residual = interp - horizontal_gr[None, :]
    loss = _huber_loss(residual, config.huber_delta)
    loss[~valid] = 0.0

    half = max(1, int(config.window_size)) // 2
    positions = np.arange(n_rows)
    lo = np.maximum(0, positions - half)
    hi = np.minimum(n_rows, positions + half + 1)

    loss_cumsum = np.concatenate([np.zeros((len(offsets), 1), dtype=float), np.cumsum(loss, axis=1)], axis=1)
    count_cumsum = np.concatenate(
        [np.zeros((len(offsets), 1), dtype=float), np.cumsum(valid.astype(float), axis=1)],
        axis=1,
    )
    loss_sum = loss_cumsum[:, hi] - loss_cumsum[:, lo]
    count_sum = count_cumsum[:, hi] - count_cumsum[:, lo]
    costs = np.full((len(offsets), n_rows), np.inf, dtype=float)
    enough = count_sum >= config.min_valid_window
    costs[enough] = loss_sum[enough] / count_sum[enough]

    texture_valid = np.isfinite(horizontal_gr) & np.isfinite(path_tvt)
    gr_values = np.where(texture_valid, horizontal_gr, 0.0)
    gr_sq_values = np.where(texture_valid, horizontal_gr * horizontal_gr, 0.0)
    gr_count_values = texture_valid.astype(float)
    gr_cumsum = np.concatenate([[0.0], np.cumsum(gr_values)])
    gr_sq_cumsum = np.concatenate([[0.0], np.cumsum(gr_sq_values)])
    gr_count_cumsum = np.concatenate([[0.0], np.cumsum(gr_count_values)])
    gr_sum = gr_cumsum[hi] - gr_cumsum[lo]
    gr_sq_sum = gr_sq_cumsum[hi] - gr_sq_cumsum[lo]
    gr_count = gr_count_cumsum[hi] - gr_count_cumsum[lo]
    texture = np.zeros(n_rows, dtype=float)
    enough_texture = gr_count >= config.min_valid_window
    mean = np.zeros(n_rows, dtype=float)
    mean_sq = np.zeros(n_rows, dtype=float)
    mean[enough_texture] = gr_sum[enough_texture] / gr_count[enough_texture]
    mean_sq[enough_texture] = gr_sq_sum[enough_texture] / gr_count[enough_texture]
    texture[enough_texture] = np.sqrt(np.maximum(0.0, mean_sq[enough_texture] - mean[enough_texture] ** 2))
    return costs, texture


def _score_positions(
    path_tvt: np.ndarray,
    horizontal_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    row_positions: np.ndarray,
    offsets: np.ndarray,
    config: PhaseLockConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_pred = len(row_positions)
    raw_offsets = np.zeros(n_pred, dtype=float)
    confidence = np.zeros(n_pred, dtype=float)
    cost_before = np.full(n_pred, np.nan, dtype=float)
    cost_after = np.full(n_pred, np.nan, dtype=float)
    texture = np.zeros(n_pred, dtype=float)
    used = np.zeros(n_pred, dtype=bool)
    uniqueness = np.zeros(n_pred, dtype=float)
    if n_pred == 0:
        return raw_offsets, confidence, cost_before, cost_after, texture, used, uniqueness

    rolling_costs, texture_all = _rolling_costs_by_offset(
        path_tvt,
        horizontal_gr,
        typewell_tvt,
        typewell_gr,
        offsets,
        config,
    )
    if rolling_costs.size == 0:
        return raw_offsets, confidence, cost_before, cost_after, texture, used, uniqueness

    costs = rolling_costs[:, row_positions]
    texture = texture_all[row_positions]
    zero_idx = int(np.argmin(np.abs(offsets)))
    cost_before = costs[zero_idx, :].astype(float, copy=True)
    finite_any = np.isfinite(costs).any(axis=0)
    best_idx = np.argmin(costs, axis=0)
    pred_idx = np.arange(n_pred)
    raw_offsets[finite_any] = offsets[best_idx[finite_any]]
    cost_after[finite_any] = costs[best_idx[finite_any], pred_idx[finite_any]]

    sorted_costs = np.sort(costs, axis=0)
    second_best = sorted_costs[1, :] if len(offsets) >= 2 else np.full(n_pred, np.inf, dtype=float)
    valid_second = np.isfinite(second_best) & np.isfinite(cost_after)
    uniqueness[valid_second] = np.maximum(
        0.0,
        (second_best[valid_second] - cost_after[valid_second]) / (np.abs(cost_after[valid_second]) + 1e-6),
    )

    valid_improvement = np.isfinite(cost_before) & np.isfinite(cost_after) & (cost_before > 1e-6)
    improvement = np.zeros(n_pred, dtype=float)
    improvement[valid_improvement] = np.maximum(
        0.0,
        (cost_before[valid_improvement] - cost_after[valid_improvement]) / (np.abs(cost_before[valid_improvement]) + 1e-6),
    )

    texture_conf = np.minimum(1.0, texture / max(config.min_texture * 2.0, 1e-6))
    uniqueness_conf = np.minimum(1.0, uniqueness / max(config.min_uniqueness, 1e-6))
    improvement_conf = np.minimum(1.0, improvement / max(config.min_improvement, 1e-6))
    confidence = np.minimum(np.minimum(texture_conf, uniqueness_conf), improvement_conf)
    used = (
        (texture >= config.min_texture)
        & (uniqueness >= config.min_uniqueness)
        & (improvement >= config.min_improvement)
        & (confidence >= config.min_confidence)
        & (np.abs(raw_offsets) > 1e-9)
    )
    return raw_offsets, confidence, cost_before, cost_after, texture, used, uniqueness


def _smooth_corrections(raw_offsets: np.ndarray, confidence: np.ndarray, used: np.ndarray, config: PhaseLockConfig) -> np.ndarray:
    corrections = np.zeros(len(raw_offsets), dtype=float)
    smooth_state = 0.0
    prev_correction = 0.0
    for idx in range(len(raw_offsets)):
        if not used[idx]:
            smooth_state = 0.0
            prev_correction = 0.0
            continue
        target = float(np.clip(raw_offsets[idx], -config.max_abs_correction, config.max_abs_correction))
        smooth_state = config.ewma_alpha * target + (1.0 - config.ewma_alpha) * smooth_state
        desired = config.alpha * float(np.clip(confidence[idx], 0.0, 1.0)) * smooth_state
        desired = float(np.clip(desired, -config.max_abs_correction, config.max_abs_correction))
        step = float(np.clip(desired - prev_correction, -config.max_step_change, config.max_step_change))
        correction = float(np.clip(prev_correction + step, -config.max_abs_correction, config.max_abs_correction))
        corrections[idx] = correction
        prev_correction = correction
    return corrections


def _disabled_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in predictions.itertuples(index=False):
        well_id, row_index = parse_prediction_id(str(row.id))
        pred_tvt = float(row.tvt)
        rows.append(
            {
                "id": str(row.id),
                "well_id": well_id,
                "row_index": int(row_index),
                "pred_tvt": pred_tvt,
                "corrected_tvt": pred_tvt,
                "raw_offset": 0.0,
                "smooth_offset": 0.0,
                "confidence": 0.0,
                "gr_cost_before": np.nan,
                "gr_cost_after": np.nan,
                "texture_score": 0.0,
                "uniqueness_score": 0.0,
                "used_correction": False,
            }
        )
    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def apply_gr_phase_lock_to_well(
    well: WellData,
    predictions: pd.DataFrame,
    config: PhaseLockConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or PhaseLockConfig()
    required = {"id", "tvt"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")

    pred = predictions[["id", "tvt"]].copy()
    pred["id"] = pred["id"].astype(str)
    parsed = pred["id"].map(parse_prediction_id)
    pred["well_id"] = [item[0] for item in parsed]
    pred["row_index"] = [item[1] for item in parsed]
    if (pred["well_id"].astype(str) != str(well.well_id)).any():
        raise ValueError(f"Predictions contain ids outside well {well.well_id}")
    pred = pred.sort_values("row_index", kind="mergesort").reset_index(drop=True)

    if not config.enabled:
        diagnostics = _disabled_diagnostics(pred)
        return pred[["id", "tvt"]].copy(), diagnostics

    forbidden_present = TRAIN_ONLY_COLUMNS.intersection(well.horizontal.columns)
    horizontal = well.horizontal.drop(columns=sorted(forbidden_present), errors="ignore").copy()
    for col in ("row_index", "GR", "TVT_input"):
        if col not in horizontal.columns:
            raise ValueError(f"Horizontal data missing required column: {col}")

    n_rows = len(horizontal)
    path_tvt = horizontal["TVT_input"].to_numpy(dtype=float).copy()
    row_indices = pred["row_index"].to_numpy(dtype=int)
    pred_tvt = pred["tvt"].to_numpy(dtype=float)
    horizontal_row_index = horizontal["row_index"].to_numpy(dtype=int)
    row_to_position = {int(row): int(pos) for pos, row in enumerate(horizontal_row_index)}
    missing_rows = [int(row) for row in row_indices if int(row) not in row_to_position]
    if missing_rows:
        raise ValueError(f"Prediction row_index outside horizontal range for {well.well_id}: {missing_rows[:5]}")
    row_positions = np.array([row_to_position[int(row)] for row in row_indices], dtype=int)
    path_tvt[row_positions] = pred_tvt

    horizontal_gr = _robust_normalize(horizontal["GR"].to_numpy(dtype=float))
    typewell_tvt, typewell_gr = _prepare_typewell(well.typewell)
    offsets = _offset_grid(config)

    raw_offsets, confidence, cost_before, cost_after, texture, used, uniqueness = _score_positions(
        path_tvt,
        horizontal_gr,
        typewell_tvt,
        typewell_gr,
        row_positions,
        offsets,
        config,
    )

    corrections = _smooth_corrections(raw_offsets, confidence, used, config)
    corrected_tvt = pred_tvt + corrections
    finite_corrected = np.isfinite(corrected_tvt)
    corrected_tvt[~finite_corrected] = pred_tvt[~finite_corrected]
    applied = np.isfinite(corrections) & (np.abs(corrections) > 1e-12)

    corrected = pd.DataFrame({"id": pred["id"].to_numpy(), "tvt": corrected_tvt})
    diagnostics = pd.DataFrame(
        {
            "id": pred["id"].to_numpy(),
            "well_id": pred["well_id"].to_numpy(),
            "row_index": row_indices,
            "pred_tvt": pred_tvt,
            "corrected_tvt": corrected_tvt,
            "raw_offset": raw_offsets,
            "smooth_offset": corrections,
            "confidence": confidence,
            "gr_cost_before": cost_before,
            "gr_cost_after": cost_after,
            "texture_score": texture,
            "uniqueness_score": uniqueness,
            "used_correction": applied,
        },
        columns=DIAGNOSTIC_COLUMNS,
    )
    return corrected, diagnostics


def apply_gr_phase_lock(
    predictions: pd.DataFrame,
    wells: dict[str, WellData],
    config: PhaseLockConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or PhaseLockConfig()
    pred = predictions[["id", "tvt"]].copy()
    pred["_input_order"] = np.arange(len(pred), dtype=int)
    parsed = pred["id"].map(parse_prediction_id)
    pred["well_id"] = [item[0] for item in parsed]
    corrected_parts = []
    diagnostic_parts = []
    for well_id in pred["well_id"].drop_duplicates().tolist():
        if well_id not in wells:
            raise KeyError(f"Missing well data for {well_id}")
        group = pred.loc[pred["well_id"] == well_id, ["id", "tvt", "_input_order"]].copy()
        corrected, diagnostics = apply_gr_phase_lock_to_well(wells[well_id], group[["id", "tvt"]], config)
        corrected["_input_order"] = group["_input_order"].to_numpy()
        diagnostics["_input_order"] = group["_input_order"].to_numpy()
        corrected_parts.append(corrected)
        diagnostic_parts.append(diagnostics)

    if corrected_parts:
        corrected_all = pd.concat(corrected_parts, ignore_index=True).sort_values("_input_order", kind="mergesort")
        diagnostics_all = pd.concat(diagnostic_parts, ignore_index=True).sort_values("_input_order", kind="mergesort")
    else:
        corrected_all = pd.DataFrame(columns=["id", "tvt", "_input_order"])
        diagnostics_all = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS + ["_input_order"])
    return corrected_all[["id", "tvt"]].reset_index(drop=True), diagnostics_all[DIAGNOSTIC_COLUMNS].reset_index(drop=True)


def run_gr_phase_lock(
    data_dir: str | Path,
    split: str,
    baseline_path: str | Path,
    output_path: str | Path,
    diagnostics_path: str | Path,
    config: PhaseLockConfig | None = None,
    max_wells: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(baseline_path)
    if "id" not in baseline.columns or "tvt" not in baseline.columns:
        raise ValueError("Baseline prediction must contain columns id,tvt")
    parsed = baseline["id"].astype(str).map(parse_prediction_id)
    well_ids = sorted({item[0] for item in parsed})
    if max_wells is not None:
        if max_wells <= 0:
            raise ValueError("max_wells must be positive when provided")
        well_ids = well_ids[:max_wells]
        keep_ids = {well_id for well_id in well_ids}
        baseline = baseline.loc[[item[0] in keep_ids for item in parsed]].reset_index(drop=True)
    wells = {well_id: load_well(data_dir, split, well_id) for well_id in well_ids}
    corrected, diagnostics = apply_gr_phase_lock(baseline[["id", "tvt"]], wells, config or PhaseLockConfig())

    output_path = Path(output_path)
    diagnostics_path = Path(diagnostics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    return corrected, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative GR phase-locked TVT drift corrector.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--baseline", type=str, required=True, help="Baseline prediction CSV with id,tvt columns.")
    parser.add_argument("--output", type=str, required=True, help="Corrected id,tvt output CSV.")
    parser.add_argument("--diagnostics-output", type=str, required=True, help="GR-lock diagnostics CSV.")
    parser.add_argument("--disable-gr-lock", action="store_true", help="Emit unchanged baseline plus diagnostics.")
    parser.add_argument("--max-wells", type=int, default=None, help="Optional runtime guard for local smoke runs.")
    parser.add_argument("--offset-min", type=float, default=PhaseLockConfig.offset_min)
    parser.add_argument("--offset-max", type=float, default=PhaseLockConfig.offset_max)
    parser.add_argument("--offset-step", type=float, default=PhaseLockConfig.offset_step)
    parser.add_argument("--window-size", type=int, default=PhaseLockConfig.window_size)
    parser.add_argument("--min-valid-window", type=int, default=PhaseLockConfig.min_valid_window)
    parser.add_argument("--min-texture", type=float, default=PhaseLockConfig.min_texture)
    parser.add_argument("--min-uniqueness", type=float, default=PhaseLockConfig.min_uniqueness)
    parser.add_argument("--min-improvement", type=float, default=PhaseLockConfig.min_improvement)
    parser.add_argument("--min-confidence", type=float, default=PhaseLockConfig.min_confidence)
    parser.add_argument("--alpha", type=float, default=PhaseLockConfig.alpha)
    parser.add_argument("--ewma-alpha", type=float, default=PhaseLockConfig.ewma_alpha)
    parser.add_argument("--max-abs-correction", type=float, default=PhaseLockConfig.max_abs_correction)
    parser.add_argument("--max-step-change", type=float, default=PhaseLockConfig.max_step_change)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseLockConfig(
        enabled=not args.disable_gr_lock,
        offset_min=args.offset_min,
        offset_max=args.offset_max,
        offset_step=args.offset_step,
        window_size=args.window_size,
        min_valid_window=args.min_valid_window,
        min_texture=args.min_texture,
        min_uniqueness=args.min_uniqueness,
        min_improvement=args.min_improvement,
        min_confidence=args.min_confidence,
        alpha=args.alpha,
        ewma_alpha=args.ewma_alpha,
        max_abs_correction=args.max_abs_correction,
        max_step_change=args.max_step_change,
    )
    corrected, diagnostics = run_gr_phase_lock(
        data_dir=args.data_dir,
        split=args.split,
        baseline_path=args.baseline,
        output_path=args.output,
        diagnostics_path=args.diagnostics_output,
        config=config,
        max_wells=args.max_wells,
    )
    print(f"Saved corrected predictions to {args.output}")
    print(f"Saved diagnostics to {args.diagnostics_output}")
    print(f"Rows: {len(corrected)}, used corrections: {int(diagnostics['used_correction'].sum())}")


if __name__ == "__main__":
    main()
