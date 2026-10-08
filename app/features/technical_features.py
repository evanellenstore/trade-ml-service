from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _compute_grouped_rsi(values: pd.Series) -> pd.Series:
    delta = values.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi.where((average_loss != 0) | (average_gain > 0), 100.0)


def _compute_grouped_macd(values: pd.Series) -> pd.Series:
    ema12 = values.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = values.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd, macd_signal, macd - macd_signal


def _compute_grouped_atr(values: pd.Series, previous_close: pd.Series) -> pd.Series:
    true_range = pd.concat(
        [
            values - values.shift(1),
            (values - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map the actual Java backtest indicator columns onto the canonical ML feature schema."""
    result = df.copy()
    if result.empty:
        return result

    close_raw = result["close"] if "close" in result.columns else pd.Series(np.nan, index=result.index)
    close = close_raw.replace(0, np.nan)
    groups = result.groupby(["symbol_token", "timeframe"], dropna=False, sort=False)

    if "momentum_rsi14" in result.columns:
        result["rsi"] = result["momentum_rsi14"]
    else:
        result["rsi"] = groups["close"].transform(_compute_grouped_rsi)
    result["rsi"] = result["rsi"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "momentum_macd" in result.columns:
        result["macd"] = result["momentum_macd"]
    else:
        result["macd"] = groups["close"].transform(lambda values: values.ewm(span=12, adjust=False, min_periods=12).mean() - values.ewm(span=26, adjust=False, min_periods=26).mean())
    if "momentum_macdSignal" in result.columns:
        result["macd_signal"] = result["momentum_macdSignal"]
    elif "momentum_macd_signal" in result.columns:
        result["macd_signal"] = result["momentum_macd_signal"]
    else:
        result["macd_signal"] = result["macd"].groupby([result["symbol_token"], result["timeframe"]], sort=False).transform(lambda values: values.ewm(span=9, adjust=False, min_periods=9).mean())
    result["macd"] = result["macd"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["macd_signal"] = result["macd_signal"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "momentum_macd_histogram" in result.columns:
        result["macd_histogram"] = result["momentum_macd_histogram"]
    elif "momentum_macdHistogram" in result.columns:
        result["macd_histogram"] = result["momentum_macdHistogram"]
    else:
        result["macd_histogram"] = result["macd"] - result["macd_signal"]
    result["macd_histogram"] = result["macd_histogram"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "volatility_atr" in result.columns and "close" in result.columns:
        atr_pct = _safe_ratio(result["volatility_atr"], close) * 100.0
        atr_pct = atr_pct.replace([np.inf, -np.inf], np.nan).where(close_raw != 0, np.nan)
        result["atr_pct"] = atr_pct.fillna(0.0).where(close_raw != 0, np.nan)
    else:
        previous_close = groups["close"].shift(1)
        atr = _compute_grouped_atr(result["close"], previous_close)
        atr_pct = _safe_ratio(atr, close) * 100.0
        atr_pct = atr_pct.replace([np.inf, -np.inf], np.nan).where(close_raw != 0, np.nan)
        result["atr_pct"] = atr_pct.fillna(0.0).where(close_raw != 0, np.nan)

    if "trend_adx" in result.columns:
        result["adx"] = result["trend_adx"]
    else:
        result["adx"] = 0.0

    ema_9_source = result["trend_ema"] if "trend_ema" in result.columns else groups["close"].transform(lambda values: values.ewm(span=9, adjust=False, min_periods=9).mean())
    ema_9_distance = _safe_ratio(ema_9_source - close, close) * 100.0
    ema_9_distance = ema_9_distance.replace([np.inf, -np.inf], np.nan).where(close_raw != 0, np.nan)
    result["ema_9_distance_pct"] = ema_9_distance.fillna(0.0).where(close_raw != 0, np.nan)

    if "trend_ema20" in result.columns:
        ema_20 = result["trend_ema20"]
    else:
        ema_20 = groups["close"].transform(lambda values: values.ewm(span=20, adjust=False, min_periods=20).mean())
    ema_20_distance = _safe_ratio(ema_20 - close, close) * 100.0
    ema_20_distance = ema_20_distance.replace([np.inf, -np.inf], np.nan).where(close_raw != 0, np.nan)
    result["ema_20_distance_pct"] = ema_20_distance.fillna(0.0).where(close_raw != 0, np.nan)

    if "trend_ema50" in result.columns:
        ema_50 = result["trend_ema50"]
    else:
        ema_50 = groups["close"].transform(lambda values: values.ewm(span=50, adjust=False, min_periods=50).mean())
    ema_50_distance = _safe_ratio(ema_50 - close, close) * 100.0
    ema_50_distance = ema_50_distance.replace([np.inf, -np.inf], np.nan).where(close_raw != 0, np.nan)
    result["ema_50_distance_pct"] = ema_50_distance.fillna(0.0).where(close_raw != 0, np.nan)

    upper = result["volatility_bb_upper"] if "volatility_bb_upper" in result.columns else None
    lower = result["volatility_bb_lower"] if "volatility_bb_lower" in result.columns else None
    middle = result["volatility_bb_middle"] if "volatility_bb_middle" in result.columns else None
    if upper is None or lower is None or middle is None:
        rolling = groups["close"].transform(lambda values: values.rolling(window=20, min_periods=20).mean())
        std = groups["close"].transform(lambda values: values.rolling(window=20, min_periods=20).std(ddof=0))
        upper = rolling + (2 * std)
        lower = rolling - (2 * std)
        middle = rolling
        result["volatility_bb_upper"] = upper
        result["volatility_bb_lower"] = lower
        result["volatility_bb_middle"] = middle

    width = result["volatility_bb_upper"] - result["volatility_bb_lower"]
    bollinger_position = np.where(width == 0, np.nan, (close - result["volatility_bb_lower"]) / width)
    bollinger_position = pd.Series(bollinger_position, index=result.index)
    bollinger_position = bollinger_position.replace([np.inf, -np.inf], np.nan).where(close_raw != 0, np.nan)
    result["bollinger_position"] = bollinger_position.fillna(0.0).where(close_raw != 0, np.nan)

    bollinger_width_pct = np.where(result["volatility_bb_middle"] == 0, np.nan, ((result["volatility_bb_upper"] - result["volatility_bb_lower"]) / result["volatility_bb_middle"]) * 100.0)
    bollinger_width_pct = pd.Series(bollinger_width_pct, index=result.index)
    bollinger_width_pct = bollinger_width_pct.replace([np.inf, -np.inf], np.nan).where(result["volatility_bb_middle"] != 0, np.nan)
    result["bollinger_width_pct"] = bollinger_width_pct.fillna(0.0).where(result["volatility_bb_middle"] != 0, np.nan)

    for column in list(result.columns):
        if column.endswith("_pct") and result[column].dtype.kind in "fc":
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)

    return result
