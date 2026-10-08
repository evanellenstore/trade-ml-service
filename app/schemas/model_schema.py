from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.trading_style import TradingStyle


class ValidationStrategy(str, Enum):
    HOLDOUT = "HOLDOUT"
    WALK_FORWARD = "WALK_FORWARD"


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    foldCount: int = Field(default=4, ge=1)
    validationWindowSessions: int = Field(default=60, ge=1)
    expandingWindow: bool = True
    sessionAligned: bool = True
    purgeEnabled: bool = True
    embargoBars: int = Field(default=0, ge=0)
    minimumTrainingSessions: Optional[int] = Field(default=None, ge=1)


class ModelTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    symbolToken: str = Field(..., min_length=1)
    tradingStyle: TradingStyle = TradingStyle.INTRADAY
    timeframe: str = Field(..., min_length=1)
    predictionHorizonBars: int = Field(default=15, ge=1)
    buyThresholdPct: float = Field(default=0.5, gt=0)
    sellThresholdPct: float = Field(default=-0.5, lt=0)
    startTime: Optional[datetime] = None
    endTime: Optional[datetime] = None
    trainRatio: float = Field(default=0.70, gt=0.0)
    validationRatio: float = Field(default=0.15, gt=0.0)
    testRatio: float = Field(default=0.15, gt=0.0)
    purgeEnabled: bool = True
    embargoBars: int = Field(default=0, ge=0)
    validationStrategy: ValidationStrategy = ValidationStrategy.HOLDOUT
    walkForward: Optional[WalkForwardConfig] = None

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        return value.upper()


class ModelTrainingResponse(BaseModel):
    success: bool
    message: str
    data: dict


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    symbolToken: str = Field(..., min_length=1)
    timeframe: str = Field(..., min_length=1)
    predictionHorizonBars: int = Field(default=15, ge=1)
    tradingStyle: TradingStyle = TradingStyle.INTRADAY
