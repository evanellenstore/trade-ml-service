from __future__ import annotations

import numpy as np
import pandas as pd


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    groups = result.groupby(["symbol_token", "timeframe"], dropna=False, sort=False)
    delta = groups["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.groupby([result["symbol_token"], result["timeframe"]], sort=False).transform(
        lambda values: values.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    average_loss = losses.groupby([result["symbol_token"], result["timeframe"]], sort=False).transform(
        lambda values: values.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result["momentum_rsi14"] = 100 - (100 / (1 + relative_strength))
    result.loc[(average_loss == 0) & (average_gain > 0), "momentum_rsi14"] = 100.0

    ema12 = groups["close"].transform(lambda values: values.ewm(span=12, adjust=False, min_periods=12).mean())
    ema26 = groups["close"].transform(lambda values: values.ewm(span=26, adjust=False, min_periods=26).mean())
    result["momentum_macd"] = ema12 - ema26
    result["momentum_macd_signal"] = result.groupby(
        ["symbol_token", "timeframe"], dropna=False, sort=False
    )["momentum_macd"].transform(lambda values: values.ewm(span=9, adjust=False, min_periods=9).mean())
    result["momentum_macd_histogram"] = result["momentum_macd"] - result["momentum_macd_signal"]
    return result
