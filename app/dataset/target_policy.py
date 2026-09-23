from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from app.domain.trading_style import TradingStyle


class TargetPolicy(ABC):
    trading_style: ClassVar[TradingStyle]

    def apply(self, df: pd.DataFrame, prediction_horizon_bars: int) -> pd.DataFrame:
        if prediction_horizon_bars <= 0:
            raise ValueError("predictionHorizonBars must be greater than zero")

        result = self._prepare_frame(df)
        future_price = self._compute_future_price(result, prediction_horizon_bars)
        result["future_close"] = future_price
        result["future_return_pct"] = (
            (result["future_close"] - result["close"]) / result["close"].replace(0, float("nan"))
        ) * 100.0
        return result

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        required_columns = {"symbol_token", "timeframe", "close"}
        missing = sorted(required_columns - set(df.columns))
        if missing:
            raise ValueError(f"Missing required target columns: {missing}")

        result = df.copy()
        if "candle_time" in result.columns:
            result = result.sort_values(["symbol_token", "timeframe", "candle_time"]).reset_index(drop=True)
            result["trading_date"] = pd.to_datetime(result["candle_time"]).dt.date
        else:
            result = result.sort_values(["symbol_token", "timeframe"]).reset_index(drop=True)
            result["trading_date"] = pd.NA
        return result

    @abstractmethod
    def group_columns(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def _compute_future_price(self, df: pd.DataFrame, prediction_horizon_bars: int) -> pd.Series:
        raise NotImplementedError


class TargetPolicyFactory:
    _policies: dict[TradingStyle, TargetPolicy] = {}

    @classmethod
    def create(cls, trading_style: TradingStyle | str) -> TargetPolicy:
        normalized = TradingStyle.normalize(trading_style)
        if normalized not in cls._policies:
            from app.dataset.intraday_target_policy import IntradayTargetPolicy
            from app.dataset.long_term_target_policy import LongTermTargetPolicy
            from app.dataset.swing_target_policy import SwingTargetPolicy

            cls._policies = {
                TradingStyle.INTRADAY: IntradayTargetPolicy(),
                TradingStyle.SWING: SwingTargetPolicy(),
                TradingStyle.LONG_TERM: LongTermTargetPolicy(),
            }
        return cls._policies[normalized]
