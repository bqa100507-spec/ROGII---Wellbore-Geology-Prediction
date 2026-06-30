from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import WellData
from gr_phase_lock import PhaseLockConfig, apply_gr_phase_lock_to_well, parse_prediction_id


def _synthetic_well(flat_gr: bool = False, include_train_only: bool = False) -> tuple[WellData, pd.DataFrame]:
    n_rows = 72
    row_index = np.arange(n_rows)
    tvt_true_path = 100.0 + row_index.astype(float) * 0.5
    typewell_tvt = np.linspace(95.0, 140.0, 180)
    typewell_gr = 90.0 + 18.0 * np.sin(typewell_tvt * 0.45) + 7.0 * np.cos(typewell_tvt * 0.18)
    horizontal_gr = np.interp(tvt_true_path, typewell_tvt, typewell_gr)
    if flat_gr:
        horizontal_gr = np.full(n_rows, 100.0)

    tvt_input = tvt_true_path.copy()
    tvt_input[20:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": row_index.astype(float),
            "X": row_index.astype(float) * 0.1,
            "Y": np.zeros(n_rows),
            "Z": -row_index.astype(float),
            "GR": horizontal_gr,
            "TVT_input": tvt_input,
            "row_index": row_index,
        }
    )
    if include_train_only:
        for idx, col in enumerate(["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]):
            horizontal[col] = (idx + 1) * 1000.0 + row_index.astype(float)

    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    well = WellData("well_a", "train", horizontal, typewell, Path("h.csv"), Path("t.csv"))
    pred_rows = row_index[20:]
    predictions = pd.DataFrame(
        {
            "id": [f"well_a_{int(idx)}" for idx in pred_rows],
            "tvt": tvt_true_path[20:] + 1.5,
        }
    )
    return well, predictions


def _config() -> PhaseLockConfig:
    return PhaseLockConfig(
        offset_min=-3.0,
        offset_max=3.0,
        offset_step=0.5,
        window_size=15,
        min_valid_window=6,
        min_texture=0.10,
        min_uniqueness=0.02,
        min_improvement=0.01,
        min_confidence=0.10,
        max_abs_correction=2.0,
        max_step_change=0.5,
    )


def test_parse_prediction_id_with_underscores():
    assert parse_prediction_id("abc_def_12") == ("abc_def", 12)
    assert parse_prediction_id("well_001") == ("well", 1)
    with pytest.raises(ValueError):
        parse_prediction_id("missingrow")
    with pytest.raises(ValueError):
        parse_prediction_id("abc_notint")


def test_phase_lock_output_is_deterministic_and_finite():
    well, predictions = _synthetic_well()
    corrected_a, diagnostics_a = apply_gr_phase_lock_to_well(well, predictions, _config())
    corrected_b, diagnostics_b = apply_gr_phase_lock_to_well(well, predictions, _config())

    pd.testing.assert_frame_equal(corrected_a, corrected_b)
    pd.testing.assert_frame_equal(diagnostics_a, diagnostics_b)
    assert np.isfinite(corrected_a["tvt"].to_numpy(dtype=float)).all()
    assert np.isfinite(diagnostics_a["corrected_tvt"].to_numpy(dtype=float)).all()


def test_flat_gr_gives_low_confidence_and_no_correction():
    well, predictions = _synthetic_well(flat_gr=True)
    corrected, diagnostics = apply_gr_phase_lock_to_well(well, predictions, _config())

    assert diagnostics["used_correction"].sum() == 0
    assert diagnostics["confidence"].max() < _config().min_confidence
    assert np.allclose(corrected["tvt"].to_numpy(dtype=float), predictions["tvt"].to_numpy(dtype=float))


def test_phase_lock_ignores_train_only_columns():
    well_clean, predictions = _synthetic_well(include_train_only=False)
    well_poisoned, _ = _synthetic_well(include_train_only=True)

    corrected_clean, diagnostics_clean = apply_gr_phase_lock_to_well(well_clean, predictions, _config())
    corrected_poisoned, diagnostics_poisoned = apply_gr_phase_lock_to_well(well_poisoned, predictions, _config())

    pd.testing.assert_frame_equal(corrected_clean, corrected_poisoned)
    pd.testing.assert_frame_equal(diagnostics_clean, diagnostics_poisoned)


def test_disabled_phase_lock_leaves_baseline_unchanged():
    well, predictions = _synthetic_well()
    disabled = PhaseLockConfig(enabled=False)
    corrected, diagnostics = apply_gr_phase_lock_to_well(well, predictions, disabled)

    assert corrected["id"].tolist() == predictions["id"].tolist()
    assert np.allclose(corrected["tvt"].to_numpy(dtype=float), predictions["tvt"].to_numpy(dtype=float))
    assert diagnostics["used_correction"].sum() == 0
    assert np.allclose(diagnostics["smooth_offset"].to_numpy(dtype=float), 0.0)
