from __future__ import annotations

import pandas as pd

from app.dataset.target_policy import TargetPolicyFactory
from app.domain.trading_style import TradingStyle


class LabelGenerator:
    """Generate future-return labels from the next N candles using a trading-style-aware target policy."""

    def calculate_future_returns(
        self,
        df: pd.DataFrame,
        prediction_horizon_bars: int,
        trading_style: TradingStyle | str = TradingStyle.SWING,
    ) -> pd.DataFrame:
        self._validate_horizon(prediction_horizon_bars)
        result = TargetPolicyFactory.create(trading_style).apply(df, prediction_horizon_bars)
        return result

    def generate_labels(
        self,
        df: pd.DataFrame,
        prediction_horizon_bars: int,
        buy_threshold_pct: float = 0.5,
        sell_threshold_pct: float = -0.5,
        trading_style: TradingStyle | str = TradingStyle.SWING,
    ) -> pd.DataFrame:
        self._validate_horizon(prediction_horizon_bars)
        if buy_threshold_pct <= 0:
            raise ValueError("buyThresholdPct must be positive")
        if sell_threshold_pct >= 0:
            raise ValueError("sellThresholdPct must be negative")

        result = self.calculate_future_returns(
            df,
            prediction_horizon_bars=prediction_horizon_bars,
            trading_style=trading_style,
        )

        result["label"] = "HOLD"
        result.loc[result["future_return_pct"] >= buy_threshold_pct, "label"] = "BUY"
        result.loc[result["future_return_pct"] <= sell_threshold_pct, "label"] = "SELL"
        return result

    @staticmethod
    def _validate_horizon(horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("predictionHorizonBars must be greater than zero")
