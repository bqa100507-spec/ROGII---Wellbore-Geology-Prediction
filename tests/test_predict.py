from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import WellData
from predict import recursive_predict_well


class SmallDeltaModel:
    def __init__(self):
        self.seen_lags = []

    def predict(self, frame):
        if isinstance(frame, np.ndarray):
            self.seen_lags.extend(frame[:, 0].tolist())
            return np.full(len(frame), 0.1, dtype=float)
        self.seen_lags.extend(frame["TVT_lag_1"].to_numpy(dtype=float).tolist())
        return np.full(len(frame), 0.1, dtype=float)


def test_recursive_prediction_updates_tvt_history():
    horizontal = pd.DataFrame(
        {
            "MD": [1.0, 2.0, 3.0],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, -1.0, -2.0],
            "GR": [100.0, 110.0, 120.0],
            "TVT_input": [10.0, np.nan, np.nan],
            "row_index": [0, 1, 2],
        }
    )
    typewell = pd.DataFrame({"TVT": [10.0, 11.0, 12.0], "GR": [100.0, 101.0, 102.0]})
    well = WellData("abc", "test", horizontal, typewell, Path("h.csv"), Path("t.csv"))

    model = SmallDeltaModel()
    pred, tvt_work = recursive_predict_well(model, ["TVT_lag_1"], well)

    assert pred["id"].tolist() == ["abc_1", "abc_2"]
    assert np.allclose(pred["tvt"].to_numpy(dtype=float), [10.1, 10.2])
    assert np.allclose(tvt_work, [10.0, 10.1, 10.2])
    assert np.allclose(model.seen_lags, [10.0, 10.1])
