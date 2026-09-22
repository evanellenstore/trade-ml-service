from __future__ import annotations

import numpy as np
import pandas as pd


def add_candle_pattern_features(
    df: pd.DataFrame,
    doji_body_ratio: float = 0.1,
    hammer_wick_ratio: float = 1.0,
    shooting_star_wick_ratio: float = 1.0,
) -> pd.DataFrame:
    """Compute simple candle pattern features without using future candles."""
    result = df.copy()
    if result.empty:
        result["is_doji"] = pd.Series(dtype=int)
        result["is_hammer"] = pd.Series(dtype=int)
        result["is_shooting_star"] = pd.Series(dtype=int)
        result["is_bullish_engulfing"] = pd.Series(dtype=int)
        result["is_bearish_engulfing"] = pd.Series(dtype=int)
        return result

    open_price = result["open"].replace(0, np.nan)
    high = result["high"].replace(0, np.nan)
    low = result["low"].replace(0, np.nan)
    close = result["close"].replace(0, np.nan)

    body_size = np.abs(close - open_price)
    total_range = (high - low).replace(0, np.nan)
    body_ratio = body_size / total_range
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low

    prev_open = result.groupby(["symbol_token", "timeframe"], dropna=False)["open"].shift(1)
    prev_close = result.groupby(["symbol_token", "timeframe"], dropna=False)["close"].shift(1)

    result["is_doji"] = ((body_ratio <= doji_body_ratio) & (total_range > 0)).astype(int)
    result["is_hammer"] = (
        (close < open_price)
        & (lower_wick >= hammer_wick_ratio * body_size)
        & (body_size > 0)
        & (total_range > 0)
    ).astype(int)
    result["is_shooting_star"] = (
        (close > open_price)
        & (upper_wick >= shooting_star_wick_ratio * body_size)
        & (body_size > 0)
        & (total_range > 0)
    ).astype(int)

    bullish_engulf = (
        (close > open_price)
        & (prev_close.notna())
        & (prev_close < prev_open)
        & (close >= prev_open)
        & (open_price <= prev_close)
    )
    bearish_engulf = (
        (close < open_price)
        & (prev_close.notna())
        & (prev_close > prev_open)
        & (close <= prev_open)
        & (open_price >= prev_close)
    )
    result["is_bullish_engulfing"] = bullish_engulf.astype(int)
    result["is_bearish_engulfing"] = bearish_engulf.astype(int)

    return result
