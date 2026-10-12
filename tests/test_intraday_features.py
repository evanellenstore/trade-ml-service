from __future__ import annotations

import pandas as pd

from app.dataset.dataset_generator import DatasetGenerator
from app.features.feature_engineering import FeatureEngineering
from app.features.feature_schema import FEATURE_SCHEMA


REQUESTED_FEATURES = {
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
}


def _candles(count: int = 1600) -> pd.DataFrame:
    times = []
    for day in pd.date_range("2026-01-05", periods=(count // 375) + 2, freq="B"):
        times.extend(pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=375, freq="min"))
        if len(times) >= count:
            break
    times = times[:count]
    close = [100 + index * 0.01 + (index % 13) * 0.03 for index in range(count)]
    return pd.DataFrame(
        {
            "candle_id": range(count),
            "symbol_token": "A",
            "timeframe": "ONE_MINUTE",
            "candle_time": times,
            "open": [value - 0.02 for value in close],
            "high": [value + 0.05 for value in close],
            "low": [value - 0.05 for value in close],
            "close": close,
            "volume": [100 + index % 7 for index in range(count)],
        }
    )


def test_feature_schema_v3_includes_all_requested_features_and_fingerprint_changes() -> None:
    assert FEATURE_SCHEMA.version == "v3"
    assert REQUESTED_FEATURES.issubset(FEATURE_SCHEMA.columns)
    assert len(FEATURE_SCHEMA.columns) == len(set(FEATURE_SCHEMA.columns))

    common = {
        "symbol_token": "A",
        "trading_style": "INTRADAY",
        "timeframe": "ONE_MINUTE",
        "prediction_horizon_bars": 5,
        "buy_threshold_pct": 0.2,
        "sell_threshold_pct": -0.2,
        "feature_columns": FEATURE_SCHEMA.columns,
        "dataset_start_time": "2026-01-01T09:15:00",
        "dataset_end_time": "2026-01-01T15:30:00",
        "dataset_row_count": 100,
    }
    old = DatasetGenerator._dataset_fingerprint(feature_version="v1", **common)
    new = DatasetGenerator._dataset_fingerprint(feature_version=FEATURE_SCHEMA.version, **common)
    assert old != new


def test_engineering_emits_new_features_and_excludes_unavailable_hourly_bucket() -> None:
    source = _candles()
    engineered = FeatureEngineering().transform(source)

    assert REQUESTED_FEATURES.issubset(engineered.columns)
    assert engineered.loc[75, "five_min_rsi"] == engineered.loc[76, "five_min_rsi"]
    assert pd.isna(engineered.loc[1243, "one_hour_trend"])
    assert pd.notna(engineered.loc[1244, "one_hour_trend"])
    assert engineered.loc[15, "minute_of_day"] == 9 * 60 + 30
    assert engineered.loc[15, "minute_from_open"] == 15
    assert engineered.loc[15, "session_progress_pct"] == 15 / 375


def test_multitimeframe_features_ignore_candles_after_prediction_time() -> None:
    original = _candles()
    changed = original.copy()
    changed.loc[1251:, ["open", "high", "low", "close", "volume"]] *= 3

    original_features = FeatureEngineering().transform(original)
    changed_features = FeatureEngineering().transform(changed)

    for column in ("five_min_rsi", "five_min_macd", "fifteen_min_rsi", "fifteen_min_macd", "one_hour_trend"):
        assert original_features.loc[1250, column] == changed_features.loc[1250, column] or (
            pd.isna(original_features.loc[1250, column]) and pd.isna(changed_features.loc[1250, column])
        )


def test_regime_features_are_numeric_and_causal() -> None:
    source = _candles(400)
    original = FeatureEngineering().transform(source)
    mutated = source.copy()
    mutated.loc[350:, ["open", "high", "low", "close", "volume"]] *= 2
    changed = FeatureEngineering().transform(mutated)

    regime_columns = ["market_regime", "volatility_regime", "trend_regime"]
    assert set(original[regime_columns].stack().unique()).issubset({-1.0, 0.0, 1.0})
    pd.testing.assert_series_equal(
        original.loc[349, regime_columns],
        changed.loc[349, regime_columns],
        check_names=False,
    )