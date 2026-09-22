import pandas as pd

from app.features.volume_features import add_volume_features


def test_volume_feature():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666", "10666"],
        "timeframe": ["ONE_MINUTE"] * 3,
        "volume": [100.0, 200.0, 150.0],
    })
    result = add_volume_features(df)
    assert "volume_change_pct" in result.columns
    assert "rolling_volume_mean" in result.columns
    assert "relative_volume" in result.columns
    assert result["volume_change_pct"].iloc[0] == 0.0
