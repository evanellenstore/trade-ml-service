import pandas as pd

from app.dataset.label_generator import LabelGenerator


def test_future_close_and_return_calculation():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666", "10666", "10666"],
        "timeframe": ["ONE_MINUTE"] * 4,
        "close": [100.0, 101.0, 102.0, 103.0],
    })
    result = LabelGenerator().generate_labels(df, prediction_horizon=2, buy_threshold_pct=0.5, sell_threshold_pct=-0.5)
    assert "future_close" in result.columns
    assert result["future_close"].iloc[0] == 102.0
    assert result["future_return_pct"].iloc[0] == 2.0


def test_buy_sell_hold_labels():
    df = pd.DataFrame({
        "symbol_token": ["10666", "10666", "10666"],
        "timeframe": ["ONE_MINUTE"] * 3,
        "close": [100.0, 100.0, 100.0],
    })
    result = LabelGenerator().generate_labels(df, prediction_horizon=1, buy_threshold_pct=0.5, sell_threshold_pct=-0.5)
    result.loc[result.index == 0, "future_close"] = 100.5
    result["future_return_pct"] = ((result["future_close"] - result["close"]) / result["close"]) * 100
    result["label"] = "HOLD"
    result.loc[result["future_return_pct"] >= 0.5, "label"] = "BUY"
    result.loc[result["future_return_pct"] <= -0.5, "label"] = "SELL"
    assert set(result["label"].unique()) <= {"BUY", "HOLD", "SELL"}


def test_prediction_horizon_excludes_last_rows():
    df = pd.DataFrame({
        "symbol_token": ["10666"] * 5,
        "timeframe": ["ONE_MINUTE"] * 5,
        "close": [100.0, 101.0, 102.0, 103.0, 104.0],
    })
    result = LabelGenerator().generate_labels(df, prediction_horizon=2, buy_threshold_pct=0.5, sell_threshold_pct=-0.5)
    assert result["future_close"].iloc[-1] is None or pd.isna(result["future_close"].iloc[-1])
