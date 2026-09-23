from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.trading_style import TradingStyle
from app.market_data.timeframe import Timeframe


class DatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    symbolToken: str = Field(..., min_length=1)
    tradingStyle: TradingStyle = Field(default=TradingStyle.INTRADAY)
    timeframe: str = Field(..., min_length=1)
    predictionHorizonBars: Optional[int] = Field(default=None, ge=1)
    buyThresholdPct: float = Field(default=0.5)
    sellThresholdPct: float = Field(default=-0.5)
    startTime: Optional[datetime] = None
    endTime: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def apply_default_horizon(cls, data):
        if isinstance(data, dict) and "predictionHorizonBars" not in data:
            data["predictionHorizonBars"] = 15
        return data

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        return Timeframe.normalize(value).value

    @field_validator("buyThresholdPct")
    @classmethod
    def validate_buy_threshold(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("buyThresholdPct must be positive")
        return value

    @field_validator("sellThresholdPct")
    @classmethod
    def validate_sell_threshold(cls, value: float) -> float:
        if value >= 0:
            raise ValueError("sellThresholdPct must be negative")
        return value


class DatasetSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbolToken: str
    tradingStyle: str
    timeframe: str
    predictionHorizonBars: int
    sourceTimeframe: str = "ONE_MINUTE"
    sourceRowCount: int
    resampledRowCount: int = 0
    partialCandleCount: int = 0
    droppedPartialCandleCount: int = 0
    indicatorWarmupRows: int = 0
    datasetRowCount: int
    skippedRowCount: int
    featureCount: int
    featureVersion: str = "v1"
    labelDistribution: dict[str, int]

class DatasetGenerationResponse(BaseModel):
    success: bool
    message: str
    data: DatasetSummary
