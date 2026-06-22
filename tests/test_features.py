from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import build_feature_frame


def test_tvt_lag_and_rolling_do_not_use_current_target():
    horizontal = pd.DataFrame(
        {
            "MD": [1.0, 2.0, 3.0],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, -1.0, -2.0],
            "GR": [100.0, 110.0, 120.0],
            "TVT": [10.0, 20.0, 30.0],
            "TVT_input": [10.0, np.nan, np.nan],
        }
    )
    typewell = pd.DataFrame({"TVT": [9.5, 19.5, 29.5], "GR": [99.0, 109.0, 119.0]})

    features = build_feature_frame(horizontal, typewell, horizontal["TVT"])

    assert np.isnan(features.loc[0, "TVT_lag_1"])
    assert features.loc[1, "TVT_lag_1"] == 10.0
    assert features.loc[1, "TVT_roll_mean_5"] == 10.0
    assert features.loc[2, "TVT_lag_1"] == 20.0
    assert features.loc[2, "TVT_roll_mean_5"] == 15.0
