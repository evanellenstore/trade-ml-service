import numpy as np
import pandas as pd
import pytest

from app.features.technical_features import add_technical_features


def test_ema_distance_if_supported():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666"],
        "timeframe": ["ONE_MINUTE", "ONE_MINUTE"],
        "trend_ema20": [100.0, 102.0],
        "close": [100.0, 101.0],
    })
    result = add_technical_features(df)
    assert "ema_20_distance_pct" in result.columns
    assert abs(result["ema_20_distance_pct"].iloc[0] - 0.0) < 1e-9


def test_atr_normalization_if_supported():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666"],
        "timeframe": ["ONE_MINUTE", "ONE_MINUTE"],
        "volatility_atr": [2.0, 3.0],
        "close": [100.0, 100.0],
    })
    result = add_technical_features(df)
    assert "atr_pct" in result.columns
    assert result["atr_pct"].iloc[0] == 2.0


def test_technical_formulas_match_required_delivery_contract():
    df = pd.DataFrame(
        {
            "symbol_token": ["A", "A"],
            "timeframe": ["ONE_MINUTE", "ONE_MINUTE"],
            "close": [100.0, 100.0],
            "momentum_rsi14": [50.0, 60.0],
            "momentum_macd": [2.0, 3.0],
            "momentum_macdSignal": [1.0, 2.0],
            "volatility_atr": [1.0, 2.0],
            "trend_adx": [20.0, 25.0],
            "trend_ema20": [90.0, 95.0],
            "trend_ema50": [80.0, 85.0],
            "volatility_bb_upper": [110.0, 110.0],
            "volatility_bb_lower": [90.0, 90.0],
            "volatility_bb_middle": [100.0, 100.0],
        }
    )
    result = add_technical_features(df)
    assert result["rsi"].tolist() == [50.0, 60.0]
    assert result["macd_histogram"].tolist() == [1.0, 1.0]
    assert result["atr_pct"].tolist() == [1.0, 2.0]
    assert result["ema_20_distance_pct"].tolist() == pytest.approx([ -10.0, -5.0 ])
    assert result["ema_50_distance_pct"].tolist() == pytest.approx([ -20.0, -15.0 ])
    assert result["bollinger_position"].tolist() == pytest.approx([0.5, 0.5])
    assert result["bollinger_width_pct"].tolist() == pytest.approx([20.0, 20.0])


def test_zero_denominator_handling_is_not_silenced():
    df = pd.DataFrame(
        {
            "symbol_token": ["A"],
            "timeframe": ["ONE_MINUTE"],
            "close": [0.0],
            "momentum_rsi14": [50.0],
            "momentum_macd": [1.0],
            "momentum_macdSignal": [0.5],
            "volatility_atr": [1.0],
            "trend_adx": [20.0],
            "trend_ema20": [0.0],
            "trend_ema50": [0.0],
            "volatility_bb_upper": [0.0],
            "volatility_bb_lower": [0.0],
            "volatility_bb_middle": [0.0],
        }
    )
    result = add_technical_features(df)
    assert np.isnan(result["atr_pct"].iloc[0])
    assert np.isnan(result["ema_20_distance_pct"].iloc[0])
    assert np.isnan(result["bollinger_position"].iloc[0])
    assert np.isnan(result["bollinger_width_pct"].iloc[0])
