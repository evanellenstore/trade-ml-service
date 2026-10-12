from __future__ import annotations


class FeatureSchema:
    """Canonical model feature contract used for generation and prediction."""

    version = "v3"
    columns = [
        "return_1",
        "return_5",
        "return_15",
        "price_range_pct",
        "body_size_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "volume_change_pct",
        "rolling_volume_mean",
        "relative_volume",
        "rsi",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_pct",
        "adx",
        "ema_9_distance_pct",
        "ema_20_distance_pct",
        "ema_50_distance_pct",
        "bollinger_position",
        "bollinger_width_pct",
        "is_doji",
        "is_hammer",
        "is_shooting_star",
        "is_bullish_engulfing",
        "is_bearish_engulfing",
        "ema20_slope",
        "ema50_slope",
        "ema200_slope",
        "price_above_ema20",
        "price_above_ema50",
        "price_above_ema200",
        "price_above_vwap",
        "rsi_change_3",
        "rsi_change_5",
        "macd_change",
        "momentum_3",
        "momentum_5",
        "momentum_10",
        "atr_ratio_short_long",
        "volatility_expansion",
        "rolling_std_5",
        "rolling_std_20",
        "distance_from_day_high",
        "distance_from_day_low",
        "breakout_day_high",
        "breakout_day_low",
        "minute_of_day",
        "minute_from_open",
        "session_progress_pct",
        "five_min_rsi",
        "five_min_macd",
        "fifteen_min_rsi",
        "fifteen_min_macd",
        "one_hour_trend",
        "market_regime",
        "volatility_regime",
        "trend_regime",
    ]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls.columns)

    @classmethod
    def as_dict(cls) -> dict[str, object]:
        return {"version": cls.version, "count": len(cls.columns), "features": list(cls.columns)}


FEATURE_SCHEMA = FeatureSchema()
FEATURE_COUNT = len(FEATURE_SCHEMA.columns)
