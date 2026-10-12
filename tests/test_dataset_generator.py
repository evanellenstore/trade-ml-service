import pandas as pd
import numpy as np
import pytest

from app.dataset.dataset_generator import DatasetGenerator


def test_target_excluded_from_feature_list():
    dataset = DatasetGenerator()
    assert "future_close" not in dataset.FEATURE_COLUMNS
    assert "future_return_pct" not in dataset.FEATURE_COLUMNS
    assert "label" not in dataset.FEATURE_COLUMNS


def test_v3_feature_contract_contains_the_canonical_columns():
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
        "rsi",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_pct",
        "adx",
        "ema_9_distance_pct",
        "ema_20_distance_pct",
        "ema_50_distance_pct",
        "bollinger_position",
        "bollinger_width_pct",
        "is_doji",
        "is_hammer",
        "is_shooting_star",
        "is_bullish_engulfing",
        "is_bearish_engulfing",
        "ema20_slope",
        "ema50_slope",
        "ema200_slope",
        "price_above_ema20",
        "price_above_ema50",
        "price_above_ema200",
        "price_above_vwap",
        "rsi_change_3",
        "rsi_change_5",
        "macd_change",
        "momentum_3",
        "momentum_5",
        "momentum_10",
        "atr_ratio_short_long",
        "volatility_expansion",
        "rolling_std_5",
        "rolling_std_20",
        "distance_from_day_high",
        "distance_from_day_low",
        "breakout_day_high",
        "breakout_day_low",
        "minute_of_day",
        "minute_from_open",
        "session_progress_pct",
        "five_min_rsi",
        "five_min_macd",
        "fifteen_min_rsi",
        "fifteen_min_macd",
        "one_hour_trend",
        "market_regime",
        "volatility_regime",
        "trend_regime",
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
        "feature_warmup_rows": 0,
        "feature_internal_invalid_rows": 1,
        "target_skipped_rows": 1,
        "overlap_warmup_and_target_skipped_rows": 0,
    }


def test_feature_invalid_diagnostics_separate_leading_warmup_from_internal_gaps():
    dataset = DatasetGenerator()
    frame = pd.DataFrame(
        {
            "symbol_token": ["A"] * 5,
            "timeframe": ["ONE_MINUTE"] * 5,
            "candle_time": pd.date_range("2026-01-01 09:15", periods=5, freq="min"),
            "return_1": [np.nan, np.nan, 0.1, np.nan, 0.2],
        }
    )

    report = dataset._feature_validation(frame)
    column_report = report["perFeature"]["return_1"]

    assert column_report["invalidValueCount"] == 3
    assert column_report["warmupRowCount"] == 2
    assert column_report["internalInvalidRowCount"] == 1
    assert column_report["firstValidTimestamp"] == "2026-01-01T09:17:00"


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
