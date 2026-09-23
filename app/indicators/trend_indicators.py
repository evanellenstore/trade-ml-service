from __future__ import annotations

import pandas as pd


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    groups = result.groupby(["symbol_token", "timeframe"], dropna=False, sort=False)["close"]
    for period in (20, 50, 100, 200):
        result[f"trend_ema{period}"] = groups.transform(
            lambda values, period=period: values.ewm(span=period, adjust=False, min_periods=period).mean()
        )
    result["trend_ema"] = result["trend_ema20"]
    return result
