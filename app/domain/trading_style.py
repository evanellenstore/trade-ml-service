from __future__ import annotations

from enum import Enum


class TradingStyle(str, Enum):
    INTRADAY = "INTRADAY"
    SWING = "SWING"
    LONG_TERM = "LONG_TERM"

    @classmethod
    def normalize(cls, value: "TradingStyle | str | None") -> "TradingStyle":
        if value is None:
            raise ValueError("tradingStyle is required")
        if isinstance(value, cls):
            return value
        normalized = str(value).upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            valid_values = ", ".join(style.value for style in cls)
            raise ValueError(f"Unsupported tradingStyle: {value}. Valid values: {valid_values}") from exc
