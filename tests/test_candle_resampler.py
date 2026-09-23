import pandas as pd

from app.market_data.candle_resampler import CandleResampler


def minute_candles(times, opens=None):
    opens = opens or list(range(len(times)))
    return pd.DataFrame(
        {
            "symbol_token": ["A"] * len(times),
            "candle_time": pd.to_datetime(times),
            "open": opens,
            "high": [value + 2 for value in opens],
            "low": [value - 1 for value in opens],
            "close": [value + 1 for value in opens],
            "volume": [10] * len(times),
        }
    )


def test_five_minute_aggregation_uses_ohlcv_rules():
    times = pd.date_range("2026-01-02 09:15", periods=10, freq="min")

    result = CandleResampler().resample(minute_candles(times), "FIVE_MINUTE")

    assert result.candles[["open", "high", "low", "close", "volume"]].to_dict("records") == [
        {"open": 0, "high": 6, "low": -1, "close": 5, "volume": 50},
        {"open": 5, "high": 11, "low": 4, "close": 10, "volume": 50},
    ]
    assert result.diagnostics.partial_candle_count == 0
    assert result.diagnostics.dropped_partial_candle_count == 0


def test_five_minute_aggregation_does_not_cross_trading_dates():
    times = list(pd.date_range("2026-01-02 15:28", periods=3, freq="min"))
    times += list(pd.date_range("2026-01-03 09:15", periods=3, freq="min"))

    result = CandleResampler().resample(minute_candles(times), "FIVE_MINUTE")

    assert result.candles.empty
    assert result.diagnostics.partial_candle_count == 3
    assert result.diagnostics.dropped_partial_candle_count == 3


def test_missing_one_minute_bar_drops_strict_partial_candle():
    times = pd.to_datetime([
        "2026-01-02 09:15",
        "2026-01-02 09:16",
        "2026-01-02 09:17",
        "2026-01-02 09:19",
    ])

    result = CandleResampler(partial_mode="STRICT").resample(minute_candles(times), "FIVE_MINUTE")

    assert result.candles.empty
    assert result.diagnostics.partial_candle_count == 1
    assert result.diagnostics.dropped_partial_candle_count == 1


def test_missing_one_minute_bar_can_be_retained_as_partial():
    times = pd.to_datetime([
        "2026-01-02 09:15",
        "2026-01-02 09:16",
        "2026-01-02 09:17",
        "2026-01-02 09:19",
    ])

    result = CandleResampler(partial_mode="ALLOW_PARTIAL").resample(minute_candles(times), "FIVE_MINUTE")

    assert len(result.candles) == 1
    assert bool(result.candles.iloc[0]["partial"])
    assert result.candles.iloc[0]["source_bar_count"] == 4
    assert result.diagnostics.dropped_partial_candle_count == 0
