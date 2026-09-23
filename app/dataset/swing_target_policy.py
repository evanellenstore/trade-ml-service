from __future__ import annotations

import pandas as pd

from app.dataset.target_policy import TargetPolicy
from app.domain.trading_style import TradingStyle


class SwingTargetPolicy(TargetPolicy):
    trading_style = TradingStyle.SWING

    def group_columns(self) -> list[str]:
        return ["symbol_token", "timeframe"]

    def _compute_future_price(self, df: pd.DataFrame, prediction_horizon_bars: int) -> pd.Series:
        return df.groupby(["symbol_token", "timeframe"], dropna=False)["close"].shift(-prediction_horizon_bars)
