import pandas as pd
import numpy as np
import pytest

from app.dataset.dataset_generator import DatasetGenerator


def test_target_excluded_from_feature_list():
    dataset = DatasetGenerator()
    assert "future_close" not in dataset.FEATURE_COLUMNS
    assert "future_return_pct" not in dataset.FEATURE_COLUMNS
    assert "label" not in dataset.FEATURE_COLUMNS


def test_v1_feature_contract_contains_exactly_fifteen_columns():
    assert DatasetGenerator.FEATURE_COLUMNS == [
        "return_1",
        "return_5",
        "return_15",
        "price_range_pct",
        "body_size_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "volume_change_pct",
        "rolling_volume_mean",
        "relative_volume",
        "is_doji",
        "is_hammer",
        "is_shooting_star",
        "is_bullish_engulfing",
        "is_bearish_engulfing",
    ]


def test_dataset_generator_feature_count_matches_expected_columns():
    dataset = DatasetGenerator()
    feature_columns = dataset.FEATURE_COLUMNS
    assert isinstance(feature_columns, list)
    assert len(feature_columns) > 0
    assert "return_1" in feature_columns
    assert "is_doji" in feature_columns


def test_finalize_dataset_uses_feature_and_target_intersection():
    dataset = DatasetGenerator()
    rows = 4
    frame = pd.DataFrame({
        "candle_id": range(rows),
        "symbol_token": ["A"] * rows,
        "timeframe": ["FIVE_MINUTE"] * rows,
        "candle_time": pd.date_range("2026-01-01", periods=rows, freq="5min"),
        "close": [100.0] * rows,
        "future_close": [101.0, np.nan, 101.0, 101.0],
        "future_return_pct": [1.0, np.nan, 1.0, 1.0],
        "label": ["BUY"] * rows,
    })
    for column in dataset.FEATURE_COLUMNS:
        frame[column] = 1.0
    frame.loc[2, "return_1"] = np.nan

    finalized, diagnostics = dataset._finalize_dataset(frame)

    assert finalized["candle_id"].tolist() == [0, 3]
    assert diagnostics == {
        "indicator_warmup_rows": 0,
        "feature_invalid_rows": 1,
        "target_skipped_rows": 1,
        "overlap_warmup_and_target_skipped_rows": 0,
    }


def test_finalize_dataset_rejects_infinite_required_features():
    dataset = DatasetGenerator()
    frame = pd.DataFrame({
        "candle_id": [1],
        "symbol_token": ["A"],
        "timeframe": ["FIVE_MINUTE"],
        "candle_time": pd.to_datetime(["2026-01-01"]),
        "close": [100.0],
        "future_close": [101.0],
        "future_return_pct": [1.0],
        "label": ["BUY"],
    })
    for column in dataset.FEATURE_COLUMNS:
        frame[column] = 1.0
    frame.loc[0, "return_5"] = float("inf")

    with pytest.raises(ValueError, match="Dataset is empty after filtering invalid rows"):
        dataset._finalize_dataset(frame)
