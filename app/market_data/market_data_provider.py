from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from app.indicators.indicator_calculator import IndicatorCalculator
from app.market_data.candle_resampler import CandleResampler, ResampleDiagnostics
from app.market_data.timeframe import Timeframe, definition_for


@dataclass(frozen=True)
class MarketDataDiagnostics:
    source_timeframe: str
    source_row_count: int
    resampled_row_count: int
    partial_candle_count: int
    dropped_partial_candle_count: int
    indicator_warmup_rows: int


@dataclass(frozen=True)
class MarketDataResult:
    data: pd.DataFrame
    diagnostics: MarketDataDiagnostics


class MarketDataProvider:
    """Choose the unchanged database path or dynamic higher-timeframe path."""

    ENABLED_DYNAMIC_TIMEFRAMES = {
        Timeframe.FIVE_MINUTE,
        Timeframe.FIFTEEN_MINUTE,
        Timeframe.THIRTY_MINUTE,
        Timeframe.ONE_HOUR,
        Timeframe.ONE_DAY,
        Timeframe.ONE_WEEK,
    }

    def __init__(
        self,
        repository=None,
        resampler: CandleResampler | None = None,
        indicator_calculator: IndicatorCalculator | None = None,
        indicator_warmup_bars: int = 200,
    ) -> None:
        self.repository = repository
        self.resampler = resampler or CandleResampler()
        self.indicator_calculator = indicator_calculator or IndicatorCalculator()
        self.indicator_warmup_bars = indicator_warmup_bars

    def get_market_data(
        self,
        symbol_token: str,
        timeframe: Timeframe | str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> MarketDataResult:
        normalized_timeframe = Timeframe.normalize(timeframe)
        repository = self._repository()
        if normalized_timeframe == Timeframe.ONE_MINUTE:
            data = repository.fetch_market_data(symbol_token, normalized_timeframe.value, start_time, end_time)
            return MarketDataResult(
                data=data,
                diagnostics=MarketDataDiagnostics(
                    source_timeframe=Timeframe.ONE_MINUTE.value,
                    source_row_count=len(data),
                    resampled_row_count=len(data),
                    partial_candle_count=0,
                    dropped_partial_candle_count=0,
                    indicator_warmup_rows=0,
                ),
            )

        if normalized_timeframe not in self.ENABLED_DYNAMIC_TIMEFRAMES:
            raise ValueError(
                f"Dynamic timeframe support is not enabled for {normalized_timeframe.value}; "
                "supported values: FIVE_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE, "
                "ONE_HOUR, ONE_DAY, ONE_WEEK"
            )

        definition = definition_for(normalized_timeframe)
        source_start = start_time
        if source_start is not None:
            lookback_minutes = (definition.source_minutes or 1440) * self.indicator_warmup_bars
            source_start = source_start - timedelta(minutes=lookback_minutes)
        raw = repository.fetch_raw_candles(symbol_token, source_start, end_time)
        resampled = self.resampler.resample(raw, normalized_timeframe)
        calculated = self.indicator_calculator.calculate(resampled.candles)
        warmup_rows = self.indicator_calculator.warmup_row_count(calculated)
        if start_time is not None:
            calculated = calculated.loc[calculated["candle_time"] >= pd.Timestamp(start_time)].copy()
        if end_time is not None:
            calculated = calculated.loc[calculated["candle_time"] <= pd.Timestamp(end_time)].copy()
        return MarketDataResult(
            data=calculated.reset_index(drop=True),
            diagnostics=MarketDataDiagnostics(
                source_timeframe=Timeframe.ONE_MINUTE.value,
                source_row_count=len(raw),
                resampled_row_count=resampled.diagnostics.resampled_row_count,
                partial_candle_count=resampled.diagnostics.partial_candle_count,
                dropped_partial_candle_count=resampled.diagnostics.dropped_partial_candle_count,
                indicator_warmup_rows=warmup_rows,
            ),
        )

    def _repository(self):
        if self.repository is None:
            from app.repository.market_data_repository import MarketDataRepository

            self.repository = MarketDataRepository()
        return self.repository
