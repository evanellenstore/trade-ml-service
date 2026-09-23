from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModelMetadata:
    tradingStyle: str
    timeframe: str
    predictionHorizonBars: int
    buyThresholdPct: float
    sellThresholdPct: float
    featureVersion: str = "v1"
    algorithm: str = "pending"
    trainingStartTime: datetime | None = None
    trainingEndTime: datetime | None = None
    extra: dict[str, object] = field(default_factory=dict)
