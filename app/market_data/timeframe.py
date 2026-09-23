from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Timeframe(str, Enum):
    ONE_MINUTE = "ONE_MINUTE"
    THREE_MINUTE = "THREE_MINUTE"
    FIVE_MINUTE = "FIVE_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    THIRTY_MINUTE = "THIRTY_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    FOUR_HOUR = "FOUR_HOUR"
    ONE_DAY = "ONE_DAY"
    ONE_WEEK = "ONE_WEEK"

    @classmethod
    def normalize(cls, value: "Timeframe | str") -> "Timeframe":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).upper())
        except ValueError as exc:
            valid = ", ".join(item.value for item in cls)
            raise ValueError(f"Unsupported timeframe: {value}. Valid values: {valid}") from exc


@dataclass(frozen=True)
class TimeframeDefinition:
    value: Timeframe
    pandas_rule: str | None
    source_minutes: int | None
    intraday: bool


TIMEFRAME_DEFINITIONS = {
    Timeframe.ONE_MINUTE: TimeframeDefinition(Timeframe.ONE_MINUTE, None, 1, True),
    Timeframe.THREE_MINUTE: TimeframeDefinition(Timeframe.THREE_MINUTE, "3min", 3, True),
    Timeframe.FIVE_MINUTE: TimeframeDefinition(Timeframe.FIVE_MINUTE, "5min", 5, True),
    Timeframe.FIFTEEN_MINUTE: TimeframeDefinition(Timeframe.FIFTEEN_MINUTE, "15min", 15, True),
    Timeframe.THIRTY_MINUTE: TimeframeDefinition(Timeframe.THIRTY_MINUTE, "30min", 30, True),
    Timeframe.ONE_HOUR: TimeframeDefinition(Timeframe.ONE_HOUR, "1h", 60, True),
    Timeframe.FOUR_HOUR: TimeframeDefinition(Timeframe.FOUR_HOUR, "4h", 240, True),
    Timeframe.ONE_DAY: TimeframeDefinition(Timeframe.ONE_DAY, "1D", None, False),
    Timeframe.ONE_WEEK: TimeframeDefinition(Timeframe.ONE_WEEK, "1W", None, False),
}


def definition_for(value: Timeframe | str) -> TimeframeDefinition:
    return TIMEFRAME_DEFINITIONS[Timeframe.normalize(value)]
