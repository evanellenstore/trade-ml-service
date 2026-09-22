import pandas as pd

from app.features.candle_patterns import add_candle_pattern_features


def test_doji_detection():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666"],
        "timeframe": ["ONE_MINUTE", "ONE_MINUTE"],
        "open": [100.0, 100.0],
        "high": [101.0, 101.0],
        "low": [99.0, 99.0],
        "close": [100.0, 99.5],
    })
    result = add_candle_pattern_features(df)
    assert "is_doji" in result.columns
    assert result["is_doji"].iloc[0] == 1


def test_hammer_detection():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666"],
        "timeframe": ["ONE_MINUTE", "ONE_MINUTE"],
        "open": [100.0, 100.0],
        "high": [102.0, 101.0],
        "low": [80.0, 81.0],
        "close": [92.0, 94.0],
    })
    result = add_candle_pattern_features(df)
    assert result["is_hammer"].iloc[0] == 1


def test_bullish_engulfing_detection():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666", "10666"],
        "timeframe": ["ONE_MINUTE", "ONE_MINUTE", "ONE_MINUTE"],
        "open": [100.0, 90.0, 85.0],
        "high": [110.0, 98.0, 112.0],
        "low": [88.0, 80.0, 82.0],
        "close": [90.0, 88.0, 110.0],
    })
    result = add_candle_pattern_features(df)
    assert result["is_bullish_engulfing"].iloc[2] == 1


def test_bearish_engulfing_detection():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666", "10666"],
        "timeframe": ["ONE_MINUTE", "ONE_MINUTE", "ONE_MINUTE"],
        "open": [100.0, 95.0, 110.0],
        "high": [110.0, 108.0, 120.0],
        "low": [90.0, 88.0, 90.0],
        "close": [110.0, 100.0, 90.0],
    })
    result = add_candle_pattern_features(df)
    assert result["is_bearish_engulfing"].iloc[2] == 1
