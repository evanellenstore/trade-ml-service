import pandas as pd

from app.features.price_features import add_price_features


def test_return_1_calculation():
    df = pd.DataFrame({
        "close": [100.0, 110.0, 99.0, 121.0],
        "open": [100.0, 100.0, 110.0, 110.0],
        "high": [101.0, 115.0, 100.0, 125.0],
        "low": [99.0, 108.0, 95.0, 118.0],
        "symbol_token": ["10666", "10666", "10666", "10666"],
        "timeframe": ["ONE_MINUTE", "ONE_MINUTE", "ONE_MINUTE", "ONE_MINUTE"],
    })

    result = add_price_features(df)

    assert "return_1" in result.columns
    assert result["return_1"].iloc[0] == 0.0
    assert abs(result["return_1"].iloc[1] - 0.1) < 1e-9
    assert abs(result["return_1"].iloc[2] - (-0.1)) < 1e-9


def test_return_5_calculation():
    df = pd.DataFrame({
        "close": [100.0, 105.0, 110.0, 100.0, 90.0, 100.0],
        "open": [100.0, 100.0, 100.0, 110.0, 100.0, 95.0],
        "high": [101.0, 106.0, 112.0, 111.0, 102.0, 101.0],
        "low": [99.0, 104.0, 109.0, 96.0, 88.0, 92.0],
        "symbol_token": ["10666"] * 6,
        "timeframe": ["ONE_MINUTE"] * 6,
    })

    result = add_price_features(df)
    assert "return_5" in result.columns
    assert abs(result["return_5"].iloc[5] - 0.0) < 1e-9


def test_price_range_feature():
    df = pd.DataFrame({
        "close": [100.0, 100.0],
        "open": [100.0, 100.0],
        "high": [110.0, 101.0],
        "low": [90.0, 99.0],
        "symbol_token": ["10666", "10666"],
        "timeframe": ["ONE_MINUTE", "ONE_MINUTE"],
    })

    result = add_price_features(df)
    assert "price_range_pct" in result.columns
    assert abs(result["price_range_pct"].iloc[0] - 0.2) < 1e-9
    assert abs(result["price_range_pct"].iloc[1] - 0.02) < 1e-9
