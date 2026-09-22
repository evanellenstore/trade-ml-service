from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class DatasetRequest(BaseModel):
    symbolToken: str = Field(..., min_length=1)
    timeframe: str = Field(..., min_length=1)
    predictionHorizon: int = Field(default=15, ge=1)
    buyThresholdPct: float = Field(default=0.5)
    sellThresholdPct: float = Field(default=-0.5)
    startTime: datetime | None = None
    endTime: datetime | None = None

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        valid = {
            "ONE_MINUTE",
            "THREE_MINUTE",
            "FIVE_MINUTE",
            "FIFTEEN_MINUTE",
            "THIRTY_MINUTE",
            "ONE_HOUR",
            "FOUR_HOUR",
            "ONE_DAY",
        }
        if value.upper() not in valid:
            raise ValueError(f"Unsupported timeframe: {value}")
        return value.upper()

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
    symbolToken: str
    timeframe: str
    predictionHorizon: int
    sourceRowCount: int
    datasetRowCount: int
    skippedRowCount: int
    featureCount: int
    labelDistribution: dict[str, int]


class DatasetGenerationResponse(BaseModel):
    success: bool
    message: str
    data: DatasetSummary
