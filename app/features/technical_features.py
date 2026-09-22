from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ML-oriented transformations from already-computed Java indicator columns when present."""
    result = df.copy()
    if result.empty:
        return result

    if "trend_ema" in result.columns and "close" in result.columns:
        result["ema_distance"] = (result["trend_ema"] - result["close"]) / result["close"].replace(0, np.nan)
    if "trend_ema20" in result.columns and "trend_ema50" in result.columns and "close" in result.columns:
        result["ema_20_50_distance"] = (result["trend_ema20"] - result["trend_ema50"]) / result["close"].replace(0, np.nan)
    if "volatility_atr" in result.columns and "close" in result.columns:
        result["atr_pct"] = result["volatility_atr"] / result["close"].replace(0, np.nan)
    if "momentum_macd" in result.columns and "close" in result.columns:
        result["macd_normalized"] = result["momentum_macd"] / result["close"].replace(0, np.nan)
    if "momentum_rsi14" in result.columns:
        result["rsi_change"] = result.groupby(["symbol_token", "timeframe"], dropna=False)["momentum_rsi14"].diff(1)

    for column in list(result.columns):
        if column.endswith("_pct") and result[column].dtype.kind in "fc":
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)

    return result
