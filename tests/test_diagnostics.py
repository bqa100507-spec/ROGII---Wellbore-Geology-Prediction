from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnostics import (
    compute_rmse_by_distance_from_ps,
    compute_rmse_by_well,
    interpolate_typewell_gr_np,
    load_aligned_predictions,
)


def test_load_aligned_predictions_and_rmse_by_well(tmp_path):
    baseline = pd.DataFrame({"id": ["a_0", "a_1", "b_0"], "tvt": [10.0, 12.0, 20.0]})
    experiment = pd.DataFrame({"id": ["a_0", "a_1", "b_0"], "tvt": [10.0, 11.0, 19.0]})
    truth = pd.DataFrame(
        {
            "id": ["a_0", "a_1", "b_0"],
            "TVT": [10.0, 10.0, 20.0],
            "well_id": ["a", "a", "b"],
            "row_index": [0, 1, 0],
        }
    )
    baseline_path = tmp_path / "baseline.csv"
    experiment_path = tmp_path / "experiment.csv"
    baseline.to_csv(baseline_path, index=False)
    experiment.to_csv(experiment_path, index=False)

    aligned, has_experiment = load_aligned_predictions(baseline_path, experiment_path, truth)
    summary = compute_rmse_by_well(aligned)

    assert has_experiment is True
    assert aligned["experiment_tvt"].tolist() == [10.0, 11.0, 19.0]
    assert summary.loc[summary["well_id"] == "a", "delta_rmse"].iloc[0] < 0.0


def test_distance_bins_and_typewell_interpolation():
    aligned = pd.DataFrame(
        {
            "id": [f"a_{i}" for i in range(4)],
            "well_id": ["a"] * 4,
            "row_index": [0, 1, 2, 3],
            "baseline_error": [0.0, 1.0, 2.0, 3.0],
            "experiment_error": [0.0, 0.5, 1.0, 1.5],
        }
    )
    distance = compute_rmse_by_distance_from_ps(aligned, bins=2)
    interp = interpolate_typewell_gr_np(
        np.array([0.0, 10.0, 20.0]),
        np.array([100.0, 120.0, 140.0]),
        np.array([5.0, 15.0, np.nan]),
    )

    assert distance["experiment_rmse"].iloc[0] < distance["baseline_rmse"].iloc[0]
    assert interp.tolist()[:2] == [110.0, 130.0]
    assert np.isnan(interp[2])
