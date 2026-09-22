import pandas as pd

from app.features.technical_features import add_technical_features


def test_ema_distance_if_supported():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666"],
        "timeframe": ["ONE_MINUTE", "ONE_MINUTE"],
        "trend_ema": [100.0, 102.0],
        "close": [100.0, 101.0],
    })
    result = add_technical_features(df)
    assert "ema_distance" in result.columns
    assert abs(result["ema_distance"].iloc[0] - 0.0) < 1e-9


def test_atr_normalization_if_supported():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666"],
        "timeframe": ["ONE_MINUTE", "ONE_MINUTE"],
        "volatility_atr": [2.0, 3.0],
        "close": [100.0, 100.0],
    })
    result = add_technical_features(df)
    assert "atr_pct" in result.columns
    assert result["atr_pct"].iloc[0] == 0.02
