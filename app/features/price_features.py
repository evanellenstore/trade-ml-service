from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds price-derived features using only current and historical values."""
    result = df.copy()
    if result.empty:
        result["return_1"] = pd.Series(dtype=float)
        result["return_5"] = pd.Series(dtype=float)
        result["return_15"] = pd.Series(dtype=float)
        result["price_range_pct"] = pd.Series(dtype=float)
        result["body_size_pct"] = pd.Series(dtype=float)
        result["upper_wick_pct"] = pd.Series(dtype=float)
        result["lower_wick_pct"] = pd.Series(dtype=float)
        return result

    result["return_1"] = (
        result.groupby(["symbol_token", "timeframe"], dropna=False)["close"]
        .pct_change(1)
        .fillna(0.0)
    )
    result["return_5"] = (
        result.groupby(["symbol_token", "timeframe"], dropna=False)["close"]
        .pct_change(5)
        .fillna(0.0)
    )
    result["return_15"] = (
        result.groupby(["symbol_token", "timeframe"], dropna=False)["close"]
        .pct_change(15)
        .fillna(0.0)
    )

    close = result["close"].replace(0, np.nan)
    high = result["high"].replace(0, np.nan)
    low = result["low"].replace(0, np.nan)
    open_price = result["open"].replace(0, np.nan)

    result["price_range_pct"] = _safe_divide(high - low, close)
    result["body_size_pct"] = _safe_divide(np.abs(close - open_price), close)
    result["upper_wick_pct"] = _safe_divide(high - np.maximum(open_price, close), close)
    result["lower_wick_pct"] = _safe_divide(np.minimum(open_price, close) - low, close)

    result["return_1"] = result["return_1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["return_5"] = result["return_5"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["return_15"] = result["return_15"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["price_range_pct"] = result["price_range_pct"].replace([np.inf, -np.inf], np.nan)
    result["body_size_pct"] = result["body_size_pct"].replace([np.inf, -np.inf], np.nan)
    result["upper_wick_pct"] = result["upper_wick_pct"].replace([np.inf, -np.inf], np.nan)
    result["lower_wick_pct"] = result["lower_wick_pct"].replace([np.inf, -np.inf], np.nan)

    return result
