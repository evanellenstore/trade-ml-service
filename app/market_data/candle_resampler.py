from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from app.market_data.timeframe import Timeframe, definition_for


@dataclass(frozen=True)
class ResampleDiagnostics:
    source_row_count: int
    resampled_row_count: int
    partial_candle_count: int
    dropped_partial_candle_count: int


@dataclass(frozen=True)
class ResampleResult:
    candles: pd.DataFrame
    diagnostics: ResampleDiagnostics


class CandleResampler:
    """Aggregate raw one-minute OHLCV candles without fabricating missing bars."""

    REQUIRED_COLUMNS = {
        "symbol_token",
        "candle_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    def __init__(self, session_start: time = time(9, 15), partial_mode: str = "STRICT") -> None:
        normalized_mode = partial_mode.upper()
        if normalized_mode not in {"STRICT", "ALLOW_PARTIAL"}:
            raise ValueError("partial_mode must be STRICT or ALLOW_PARTIAL")
        self.session_start = session_start
        self.partial_mode = normalized_mode

    def resample(self, df: pd.DataFrame, timeframe: Timeframe | str) -> ResampleResult:
        target = Timeframe.normalize(timeframe)
        definition = definition_for(target)
        if target == Timeframe.ONE_MINUTE:
            return ResampleResult(
                candles=df.copy(),
                diagnostics=ResampleDiagnostics(len(df), len(df), 0, 0),
            )
        self._validate_input(df)
        source = df.copy()
        source["candle_time"] = pd.to_datetime(source["candle_time"])
        source = source.sort_values(["symbol_token", "candle_time"]).reset_index(drop=True)
        source["trading_date"] = source["candle_time"].dt.date

        if target == Timeframe.ONE_DAY:
            source["_bucket"] = source["trading_date"]
            group_columns = ["symbol_token", "_bucket"]
            expected_count = None
        elif target == Timeframe.ONE_WEEK:
            source["_bucket"] = source["candle_time"].dt.to_period("W-MON").astype(str)
            group_columns = ["symbol_token", "_bucket"]
            expected_count = None
        else:
            minutes = (
                source["candle_time"].dt.hour * 60
                + source["candle_time"].dt.minute
                - (self.session_start.hour * 60 + self.session_start.minute)
            )
            source["_bucket"] = (minutes // definition.source_minutes).astype(int)
            group_columns = ["symbol_token", "trading_date", "_bucket"]
            expected_count = definition.source_minutes

        grouped = source.groupby(group_columns, sort=True, dropna=False)
        aggregates = grouped.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            candle_time=("candle_time", "first"),
            source_bar_count=("candle_time", "size"),
        ).reset_index()
        if target in {Timeframe.ONE_DAY, Timeframe.ONE_WEEK}:
            aggregates["trading_date"] = pd.to_datetime(aggregates["candle_time"]).dt.date
        aggregates["partial"] = (
            aggregates["source_bar_count"] < expected_count if expected_count else False
        )
        partial_count = int(aggregates["partial"].sum())
        dropped_count = partial_count if self.partial_mode == "STRICT" else 0
        if self.partial_mode == "STRICT":
            aggregates = aggregates.loc[~aggregates["partial"]].copy()

        aggregates["timeframe"] = target.value
        aggregates["candle_id"] = (
            aggregates["symbol_token"].astype(str)
            + "|"
            + target.value
            + "|"
            + aggregates["candle_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        )
        columns = [
            "candle_id",
            "symbol_token",
            "timeframe",
            "candle_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trading_date",
            "partial",
            "source_bar_count",
        ]
        result = aggregates[columns].sort_values(["symbol_token", "candle_time"]).reset_index(drop=True)
        return ResampleResult(
            candles=result,
            diagnostics=ResampleDiagnostics(
                source_row_count=len(source),
                resampled_row_count=len(result),
                partial_candle_count=partial_count,
                dropped_partial_candle_count=dropped_count,
            ),
        )

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = sorted(self.REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(f"Raw candle data missing required columns: {missing}")
        if df.empty:
            return
        for column in ("open", "high", "low", "close", "volume"):
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"{column} must be numeric")
