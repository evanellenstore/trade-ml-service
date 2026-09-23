from __future__ import annotations

import pandas as pd

from app.indicators.momentum_indicators import add_momentum_indicators
from app.indicators.trend_indicators import add_trend_indicators
from app.indicators.volatility_indicators import add_volatility_indicators


class IndicatorCalculator:
    """Calculate timeframe-local indicators from chronologically ordered OHLCV candles."""

    INDICATOR_COLUMNS = (
        "momentum_rsi14",
        "momentum_macd",
        "momentum_macd_signal",
        "momentum_macd_histogram",
        "trend_ema",
        "trend_ema20",
        "trend_ema50",
        "trend_ema100",
        "trend_ema200",
        "volatility_atr",
        "volatility_bb_lower",
        "volatility_bb_middle",
        "volatility_bb_upper",
        "volatility_bb_width",
        "volatility_percentb",
    )

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.sort_values(["symbol_token", "timeframe", "candle_time"]).reset_index(drop=True).copy()
        result = add_momentum_indicators(result)
        result = add_trend_indicators(result)
        result = add_volatility_indicators(result)
        return result

    def warmup_row_count(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        return int(df[list(self.INDICATOR_COLUMNS)].isna().any(axis=1).sum())
