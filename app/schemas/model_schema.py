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


class OOFCombinedThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidenceThreshold: float = Field(..., ge=0.0, le=1.0)
    marginThreshold: float = Field(..., ge=0.0, le=1.0)


class OOFAnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidenceThresholds: list[float] = Field(
        default_factory=lambda: [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        min_length=1,
    )
    marginThresholds: list[float] = Field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.25],
        min_length=1,
    )
    combinedThresholds: list[OOFCombinedThreshold] = Field(
        default_factory=lambda: [
            OOFCombinedThreshold(confidenceThreshold=0.60, marginThreshold=0.10),
            OOFCombinedThreshold(confidenceThreshold=0.65, marginThreshold=0.15),
            OOFCombinedThreshold(confidenceThreshold=0.70, marginThreshold=0.20),
        ]
    )
    brokerage: float = Field(default=0.0, ge=0.0)
    transactionCostPct: float = Field(default=0.0, ge=0.0)
    slippagePct: float = Field(default=0.0, ge=0.0)
    sttPct: float = Field(default=0.0, ge=0.0)
    exchangeFeePct: float = Field(default=0.0, ge=0.0)
    capitalPerTrade: float = Field(default=100000.0, gt=0.0)
    minimumTradeCount: int = Field(default=30, ge=1)
    maximumDrawdownPct: float = Field(default=25.0, gt=0.0)

    @field_validator("confidenceThresholds", "marginThresholds")
    @classmethod
    def validate_thresholds(cls, values: list[float]) -> list[float]:
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("OOF thresholds must be between 0.0 and 1.0")
        if len(set(values)) != len(values):
            raise ValueError("OOF thresholds must be unique")
        return values


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
    oofAnalysis: OOFAnalysisConfig = Field(default_factory=OOFAnalysisConfig)

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
