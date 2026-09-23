from __future__ import annotations

from datetime import datetime
import math
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.trading_style import TradingStyle


VALID_TIMEFRAMES = {
    "ONE_MINUTE",
    "THREE_MINUTE",
    "FIVE_MINUTE",
    "FIFTEEN_MINUTE",
    "THIRTY_MINUTE",
    "ONE_HOUR",
    "FOUR_HOUR",
    "ONE_DAY",
}


class TargetAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    symbolToken: str = Field(..., min_length=1)
    tradingStyle: TradingStyle = Field(default=TradingStyle.INTRADAY)
    timeframe: str = Field(..., min_length=1)
    predictionHorizonsBars: Optional[list[int]] = Field(default=None, min_length=1, max_length=20)
    thresholdsPct: list[float] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Positive percentage thresholds used for BUY/HOLD/SELL classification.",
    )
    startTime: Optional[datetime] = None
    endTime: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def require_horizons(cls, data):
        if isinstance(data, dict) and "predictionHorizonsBars" not in data:
            raise ValueError("predictionHorizonsBars is required")
        return data

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in VALID_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {value}")
        return normalized

    @field_validator("predictionHorizonsBars")
    @classmethod
    def validate_horizons(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("predictionHorizonsBars must contain only values greater than zero")
        return sorted(set(values))

    @field_validator("thresholdsPct")
    @classmethod
    def validate_thresholds(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("thresholdsPct must contain only positive values")
        return sorted(set(values))

    @field_validator("endTime")
    @classmethod
    def validate_time_range(cls, value: Optional[datetime], info) -> Optional[datetime]:
        start_time = info.data.get("startTime")
        if value is not None and start_time is not None and value < start_time:
            raise ValueError("endTime must be greater than or equal to startTime")
        return value


class DistributionStatistics(BaseModel):
    count: int
    min: float
    p01: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    max: float
    mean: float
    std: float


class ClassMetric(BaseModel):
    count: int
    percentage: float


class ThresholdAnalysis(BaseModel):
    thresholdPct: float
    buy: ClassMetric
    hold: ClassMetric
    sell: ClassMetric
    directional: ClassMetric
    classImbalanceRatio: Optional[float]
    majorityClassPct: float
    minorityClassPct: float


class HorizonAnalysis(BaseModel):
    predictionHorizonBars: int
    validTargetRows: int
    skippedRows: int
    returnStatistics: DistributionStatistics
    thresholdAnalysis: list[ThresholdAnalysis]

class TargetAnalysisData(BaseModel):
    symbolToken: str
    tradingStyle: str
    timeframe: str
    sourceRowCount: int
    analysis: list[HorizonAnalysis]


class TargetAnalysisResponse(BaseModel):
    success: bool
    message: str
    data: TargetAnalysisData