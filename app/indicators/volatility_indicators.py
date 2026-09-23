from __future__ import annotations

import numpy as np
import pandas as pd


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    groups = result.groupby(["symbol_token", "timeframe"], dropna=False, sort=False)
    previous_close = groups["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["volatility_atr"] = true_range.groupby(
        [result["symbol_token"], result["timeframe"]], sort=False
    ).transform(lambda values: values.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean())

    rolling = groups["close"].rolling(window=20, min_periods=20)
    middle = rolling.mean().reset_index(level=[0, 1], drop=True)
    standard_deviation = rolling.std(ddof=0).reset_index(level=[0, 1], drop=True)
    result["volatility_bb_middle"] = middle.to_numpy()
    result["volatility_bb_lower"] = (middle - 2 * standard_deviation).to_numpy()
    result["volatility_bb_upper"] = (middle + 2 * standard_deviation).to_numpy()
    result["volatility_bb_width"] = (
        result["volatility_bb_upper"] - result["volatility_bb_lower"]
    ) / result["volatility_bb_middle"].replace(0, np.nan)
    result["volatility_percentb"] = (
        result["close"] - result["volatility_bb_lower"]
    ) / (result["volatility_bb_upper"] - result["volatility_bb_lower"]).replace(0, np.nan)
    return result
