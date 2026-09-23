from __future__ import annotations

import pandas as pd

from app.dataset.target_policy import TargetPolicy
from app.domain.trading_style import TradingStyle


class IntradayTargetPolicy(TargetPolicy):
    trading_style = TradingStyle.INTRADAY

    def group_columns(self) -> list[str]:
        return ["symbol_token", "timeframe", "trading_date"]

    def _compute_future_price(self, df: pd.DataFrame, prediction_horizon_bars: int) -> pd.Series:
        group_cols = self.group_columns()
        if any(column not in df.columns for column in group_cols):
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")

        return df.groupby(group_cols, dropna=False)["close"].shift(-prediction_horizon_bars)
