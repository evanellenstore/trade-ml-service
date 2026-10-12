import pandas as pd

from app.indicators.indicator_calculator import IndicatorCalculator
from app.dataset.dataset_generator import DatasetGenerator
from app.dataset.target_analysis_service import TargetAnalysisService
from app.market_data.market_data_provider import MarketDataProvider


class FakeRepository:
    def __init__(self, raw, joined):
        self.raw = raw
        self.joined = joined
        self.raw_calls = 0
        self.joined_calls = 0

    def fetch_raw_candles(self, symbol_token, start_time=None, end_time=None):
        self.raw_calls += 1
        return self.raw

    def fetch_market_data(self, symbol_token, timeframe, start_time=None, end_time=None):
        self.joined_calls += 1
        return self.joined


def raw_candles(count=40):
    times = []
    for day in pd.date_range("2026-01-02", periods=(count // 375) + 2, freq="B"):
        times.extend(pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=375, freq="min"))
        if len(times) >= count:
            break
    times = times[:count]
    opens = [100 + value for value in range(count)]
    return pd.DataFrame(
        {
            "symbol_token": ["A"] * count,
            "candle_time": times,
            "open": opens,
            "high": [value + 2 for value in opens],
            "low": [value - 1 for value in opens],
            "close": [value + 1 for value in opens],
            "volume": [10] * count,
        }
    )


def test_higher_timeframe_provider_uses_raw_candles_and_calculates_local_indicators():
    repository = FakeRepository(raw_candles(), pd.DataFrame())

    result = MarketDataProvider(repository=repository).get_market_data("A", "FIVE_MINUTE")

    assert repository.raw_calls == 1
    assert repository.joined_calls == 0
    assert set(result.data["timeframe"]) == {"FIVE_MINUTE"}
    assert "momentum_rsi14" in result.data.columns
    assert "volatility_atr" in result.data.columns


def test_one_minute_provider_preserves_existing_joined_indicator_path():
    joined = raw_candles(2)
    joined["momentum_rsi14"] = [50.0, 51.0]
    repository = FakeRepository(pd.DataFrame(), joined)

    result = MarketDataProvider(repository=repository).get_market_data("A", "ONE_MINUTE")

    assert repository.raw_calls == 0
    assert repository.joined_calls == 1
    pd.testing.assert_frame_equal(result.data, joined)


def test_one_minute_provider_fetches_source_candles_when_requested():
    joined = raw_candles(2)
    repository = FakeRepository(raw_candles(10), joined)

    result = MarketDataProvider(repository=repository).get_market_data(
        "A",
        "ONE_MINUTE",
        include_source_candles=True,
    )

    assert repository.raw_calls == 1
    assert result.source_data is repository.raw


def test_indicator_calculator_only_uses_resampled_frame():
    frame = raw_candles(40)
    frame["timeframe"] = "FIVE_MINUTE"
    calculated = IndicatorCalculator().calculate(frame)

    assert calculated["momentum_rsi14"].iloc[-1] == calculated["momentum_rsi14"].iloc[-1]
    assert "momentum_rsi14" not in frame.columns


def test_five_minute_feature_warmup_is_independent_of_target_analysis_counts():
    repository = FakeRepository(raw_candles(6000), pd.DataFrame())
    generator = DatasetGenerator()
    generator.market_data_provider = MarketDataProvider(repository=repository)

    summary = generator.generate_dataset(
        symbol_token="A",
        timeframe="FIVE_MINUTE",
        prediction_horizon_bars=1,
        buy_threshold_pct=0.15,
        sell_threshold_pct=-0.15,
    )
    analysis = TargetAnalysisService(repository=repository).analyze(
        symbol_token="A",
        timeframe="FIVE_MINUTE",
        prediction_horizons_bars=[1],
        thresholds_pct=[0.15],
    )
    threshold = analysis.analysis[0].thresholdAnalysis[0]

    assert summary.datasetRowCount < analysis.analysis[0].validTargetRows
    assert sum(summary.labelDistribution.values()) == summary.datasetRowCount
    assert summary.datasetRowCount <= threshold.buy.count + threshold.hold.count + threshold.sell.count
