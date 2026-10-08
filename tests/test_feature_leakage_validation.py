from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.feature_engineering import FeatureEngineering
from app.features.feature_schema import FEATURE_SCHEMA
from app.features.feature_leakage import FeatureLeakageValidator


def _synthetic_frame() -> pd.DataFrame:
    rows = 60
    close = 100.0 + np.arange(rows, dtype=float) * 0.5
    return pd.DataFrame(
        {
            "symbol_token": ["100"] * rows,
            "timeframe": ["FIVE_MINUTE"] * rows,
            "trading_date": pd.date_range("2026-01-01", periods=rows, freq="min").date,
            "candle_time": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": [100.0] * rows,
        }
    )


def test_return_1_uses_current_and_previous_close():
    frame = pd.DataFrame(
        {
            "symbol_token": ["A", "A", "A"],
            "timeframe": ["ONE_MINUTE"] * 3,
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 111.0, 100.0],
            "low": [99.0, 109.0, 98.0],
            "close": [100.0, 110.0, 99.0],
            "volume": [100.0, 100.0, 100.0],
        }
    )
    result = FeatureEngineering().transform(frame)
    assert result["return_1"].tolist() == pytest.approx([0.0, 0.1, -0.1])


def test_return_5_and_return_15_are_backward_looking():
    close = [100.0 + index * 2.0 for index in range(16)]
    frame = pd.DataFrame(
        {
            "symbol_token": ["A"] * 16,
            "timeframe": ["ONE_MINUTE"] * 16,
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [100.0] * 16,
        }
    )
    result = FeatureEngineering().transform(frame)
    assert result.loc[5, "return_5"] == pytest.approx(0.1)
    assert result.loc[15, "return_15"] == pytest.approx(0.3)
    assert result.loc[0, "return_5"] == 0.0
    assert result.loc[0, "return_15"] == 0.0


def test_future_mutation_does_not_change_features_at_timestamp_t():
    frame = _synthetic_frame()
    engineered = FeatureEngineering().transform(frame)
    before = FeatureEngineering().feature_matrix(engineered).loc[20].copy()

    mutated = frame.copy()
    mutated.loc[21:60, "close"] = 1000.0
    mutated.loc[21:60, "high"] = 1000.2
    mutated.loc[21:60, "low"] = 999.8
    mutated.loc[21:60, "volume"] = 1000.0
    after = FeatureEngineering().feature_matrix(FeatureEngineering().transform(mutated)).loc[20]

    pd.testing.assert_series_equal(before, after, check_exact=False, rtol=1e-12, atol=1e-12)


def test_future_target_mutation_does_not_change_features():
    frame = _synthetic_frame()
    engineered = FeatureEngineering().transform(frame)
    engineered["future_close"] = engineered["close"] + 1.0
    engineered["future_return_pct"] = 1.0
    engineered["label"] = "BUY"
    before = FeatureEngineering().feature_matrix(engineered).loc[20].copy()

    target_mutated = engineered.copy()
    target_mutated.loc[20, "future_close"] = 2000.0
    target_mutated.loc[20, "future_return_pct"] = 1900.0
    target_mutated.loc[20, "label"] = "SELL"
    after = FeatureEngineering().feature_matrix(target_mutated).loc[20]

    pd.testing.assert_series_equal(before, after, check_exact=False, rtol=1e-12, atol=1e-12)


def test_feature_matrix_is_exact_schema_and_contains_no_future_columns():
    frame = _synthetic_frame()
    matrix = FeatureEngineering().feature_matrix(FeatureEngineering().transform(frame))
    assert list(matrix.columns) == FEATURE_SCHEMA.columns
    assert matrix.shape[1] == 26
    assert not matrix.isna().any().any()
    assert np.isfinite(matrix.to_numpy(dtype=float)).all()

    validator = FeatureLeakageValidator().validate(matrix)
    assert validator["passed"] is True
    assert validator["futureFeatureColumnsDetected"] == []
    assert validator["targetColumnsInFeatures"] == []
    assert validator["schemaOrderValid"] is True
    assert validator["featureCountValid"] is True


def test_validator_rejects_target_columns_in_feature_matrix():
    frame = _synthetic_frame()
    engineered = FeatureEngineering().transform(frame)
    engineered["future_close"] = 1.0
    matrix = FeatureEngineering().feature_matrix(engineered)
    matrix["future_close"] = 1.0

    validator = FeatureLeakageValidator().validate(matrix)
    assert validator["passed"] is False
    assert validator["targetColumnsInFeatures"] == ["future_close"]


def test_feature_matrix_rejects_invalid_values():
    frame = _synthetic_frame()
    engineered = FeatureEngineering().transform(frame)
    engineered.loc[0, "return_1"] = np.nan
    with pytest.raises(ValueError, match="Feature matrix"):
        FeatureEngineering().feature_matrix(engineered, require_finite=True)
