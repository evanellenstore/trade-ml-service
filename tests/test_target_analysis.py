import pandas as pd
import pytest

from app.dataset.label_generator import LabelGenerator
from app.dataset.target_analysis_service import TargetAnalysisService
from app.schemas.target_analysis_schema import TargetAnalysisRequest


def candles(closes, symbol="A", timeframe="ONE_MINUTE"):
    return pd.DataFrame(
        {
            "candle_id": range(len(closes)),
            "symbol_token": [symbol] * len(closes),
            "timeframe": [timeframe] * len(closes),
            "candle_time": pd.date_range("2026-01-01", periods=len(closes), freq="min"),
            "close": closes,
        }
    )


class FakeRepository:
    def __init__(self, frame):
        self.frame = frame
        self.calls = 0

    def fetch_market_data(self, symbol_token, timeframe, start_time=None, end_time=None):
        self.calls += 1
        return self.frame


def test_future_returns_horizon_one_and_five_exclude_final_rows_without_mutation():
    source = candles([100, 101, 102, 103, 104, 105, 106])
    original = source.copy(deep=True)
    generator = LabelGenerator()

    horizon_one = generator.calculate_future_returns(source, 1)
    horizon_five = generator.calculate_future_returns(source, 5)

    assert horizon_one["future_close"].iloc[0] == 101
    assert horizon_one["future_return_pct"].iloc[0] == 1
    assert horizon_five["future_close"].iloc[0] == 105
    assert horizon_five["future_return_pct"].iloc[0] == 5
    assert horizon_one["future_close"].tail(1).isna().all()
    assert horizon_five["future_close"].tail(5).isna().all()
    pd.testing.assert_frame_equal(source, original)


def test_threshold_boundaries_and_counts():
    source = candles([100, 101, 100, 99, 99, 99])
    result = TargetAnalysisService(FakeRepository(source)).analyze(
        "A", "ONE_MINUTE", [1], [1]
    )
    threshold = result.analysis[0].thresholdAnalysis[0]

    assert threshold.buy.count == 1
    assert threshold.sell.count == 1
    assert threshold.hold.count == 3
    assert threshold.buy.count + threshold.hold.count + threshold.sell.count == 5
    assert sum((threshold.buy.percentage, threshold.hold.percentage, threshold.sell.percentage)) == pytest.approx(100)
    assert threshold.directional.count == 2


def test_analysis_supports_multiple_horizons_thresholds_and_one_repository_call():
    source = candles([100, 101, 102, 103, 104, 105, 106, 107])
    repository = FakeRepository(source)
    result = TargetAnalysisService(repository).analyze(
        "A", "ONE_MINUTE", [5, 1, 1], [2, 1, 2]
    )

    assert repository.calls == 1
    assert [item.predictionHorizon for item in result.analysis] == [1, 5]
    assert [item.thresholdPct for item in result.analysis[0].thresholdAnalysis] == [1, 2]
    assert result.analysis[0].validTargetRows == 7
    assert result.analysis[0].skippedRows == 1
    assert result.analysis[1].validTargetRows == 3
    assert result.analysis[1].skippedRows == 5
    assert result.analysis[0].returnStatistics.count == 7
    assert result.analysis[0].returnStatistics.p50 > 0


def test_future_returns_never_cross_symbol_or_timeframe_groups():
    first = candles([100, 101, 102], symbol="A", timeframe="ONE_MINUTE")
    second = candles([1000, 1001, 1002], symbol="B", timeframe="ONE_MINUTE")
    third = candles([2000, 2001, 2002], symbol="A", timeframe="FIVE_MINUTE")
    source = pd.concat([first, second, third], ignore_index=True)
    result = LabelGenerator().calculate_future_returns(source, 1)

    grouped = result.set_index(["symbol_token", "timeframe"])
    assert grouped.loc[("A", "ONE_MINUTE"), "future_close"].iloc[1] == 102
    assert pd.isna(grouped.loc[("A", "ONE_MINUTE"), "future_close"].iloc[-1])
    assert grouped.loc[("B", "ONE_MINUTE"), "future_close"].iloc[1] == 1002
    assert pd.isna(grouped.loc[("B", "ONE_MINUTE"), "future_close"].iloc[-1])
    assert grouped.loc[("A", "FIVE_MINUTE"), "future_close"].iloc[1] == 2002
    assert pd.isna(grouped.loc[("A", "FIVE_MINUTE"), "future_close"].iloc[-1])


def test_request_deduplicates_and_sorts_inputs():
    request = TargetAnalysisRequest(
        symbolToken="A",
        timeframe="one_minute",
        predictionHorizons=[30, 5, 30, 15],
        thresholdsPct=[0.5, 0.1, 0.5, 0.2],
    )

    assert request.timeframe == "ONE_MINUTE"
    assert request.predictionHorizons == [5, 15, 30]
    assert request.thresholdsPct == [0.1, 0.2, 0.5]


def test_insufficient_data_is_rejected():
    with pytest.raises(ValueError, match="Insufficient candles"):
        TargetAnalysisService(FakeRepository(candles([100, 101]))).analyze(
            "A", "ONE_MINUTE", [2], [0.5]
        )


def test_future_return_is_not_a_feature():
    from app.dataset.dataset_generator import DatasetGenerator

    assert "future_return_pct" not in DatasetGenerator.FEATURE_COLUMNS