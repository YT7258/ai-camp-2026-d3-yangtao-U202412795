from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import chronological_split, make_lagged


class SequenceTests(unittest.TestCase):
    def test_chronological_split_keeps_order(self):
        frame = pd.DataFrame({"value": range(20)})
        train, test = chronological_split(frame)
        self.assertLess(train.index.max(), test.index.min())
        self.assertEqual(len(train) + len(test), 20)

    def test_lags_use_past_to_predict_next(self):
        # This sequence checks shifting only. Course evidence comes from the
        # downloaded household measurements.
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=30, freq="h"),
                "global_active_power": list(range(30)),
            }
        )
        lagged = make_lagged(frame)
        first = lagged.iloc[0]
        self.assertEqual(first["lag_24"], 0)
        self.assertEqual(first["lag_1"], 23)
        self.assertEqual(first["target_next"], 25)
        self.assertEqual(first["hour_of_day"], 0)


if __name__ == "__main__":
    unittest.main()
