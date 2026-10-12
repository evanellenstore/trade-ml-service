from __future__ import annotations

import numpy as np
import pandas as pd


SESSION_OPEN_MINUTES = 9 * 60 + 15
SESSION_LENGTH_MINUTES = 375
TIMEFRAME_MINUTES = {
    "ONE_MINUTE": 1,
    "THREE_MINUTE": 3,
    "FIVE_MINUTE": 5,
    "FIFTEEN_MINUTE": 15,
    "THIRTY_MINUTE": 30,
    "ONE_HOUR": 60,
    "FOUR_HOUR": 240,
    "ONE_DAY": 1440,
}


def _rsi(values: pd.Series) -> pd.Series:
    delta = values.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result = 100.0 - (100.0 / (1.0 + relative_strength))
    result = result.mask(average_loss.eq(0) & average_gain.gt(0), 100.0)
    result = result.mask(average_loss.eq(0) & average_gain.eq(0), 50.0)
    return result


def _macd(values: pd.Series) -> pd.Series:
    fast = values.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = values.ewm(span=26, adjust=False, min_periods=26).mean()
    return fast - slow


def _multi_timeframe_features(
    df: pd.DataFrame,
    source_candles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = df.copy()
    names = (
        "five_min_rsi",
        "five_min_macd",
        "fifteen_min_rsi",
        "fifteen_min_macd",
        "one_hour_trend",
    )
    for name in names:
        result[name] = np.nan
    required = {"symbol_token", "timeframe", "candle_time", "close", "high", "low", "volume"}
    if result.empty or not required.issubset(result.columns):
        return result

    work = result.copy()
    work["_feature_position"] = np.arange(len(work))
    work["candle_time"] = pd.to_datetime(work["candle_time"])
    source_work = source_candles.copy() if source_candles is not None else work
    if source_work.empty or not {"symbol_token", "candle_time", "close", "high", "low", "volume"}.issubset(source_work.columns):
        return result
    source_work["candle_time"] = pd.to_datetime(source_work["candle_time"])
    for symbol, symbol_rows in work.groupby("symbol_token", sort=False, dropna=False):
        base_timeframe = str(symbol_rows["timeframe"].iloc[0]).upper()
        base_minutes = TIMEFRAME_MINUTES.get(base_timeframe)
        if base_minutes is None:
            continue
        prediction_times = symbol_rows["candle_time"] + pd.to_timedelta(base_minutes, unit="min")
        for target_minutes, prefix in ((5, "five_min"), (15, "fifteen_min"), (60, "one_hour")):
            if source_candles is None and (base_minutes > target_minutes or target_minutes % base_minutes):
                continue
            source_minutes = 1 if source_candles is not None else base_minutes
            expected_bars = target_minutes // source_minutes
            available_source = source_work.loc[source_work["symbol_token"] == symbol]
            source = available_source[["candle_time", "open", "high", "low", "close", "volume"]].copy()
            source["_trading_date"] = source["candle_time"].dt.date
            minutes = source["candle_time"].dt.hour * 60 + source["candle_time"].dt.minute
            source["_bucket"] = ((minutes - SESSION_OPEN_MINUTES) // target_minutes).astype(int)
            grouped = source.groupby(["_trading_date", "_bucket"], sort=True, dropna=False)
            aggregates = grouped.agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
                candle_time=("candle_time", "first"),
                last_source_time=("candle_time", "last"),
                source_count=("candle_time", "size"),
            ).reset_index()
            aggregates = aggregates.loc[aggregates["source_count"] == expected_bars].copy()
            if aggregates.empty:
                continue

            # A full row count is not enough when a source candle is missing.
            valid_buckets = []
            for _, bucket in grouped:
                if len(bucket) != expected_bars:
                    continue
                times = bucket["candle_time"].sort_values()
                if expected_bars == 1 or times.diff().dropna().eq(pd.Timedelta(minutes=source_minutes)).all():
                    valid_buckets.append((times.iloc[0], times.iloc[-1]))
            valid_starts = {start for start, _ in valid_buckets}
            aggregates = aggregates.loc[aggregates["candle_time"].isin(valid_starts)].copy()
            if aggregates.empty:
                continue

            aggregates = aggregates.sort_values("last_source_time").reset_index(drop=True)
            aggregates["available_at"] = aggregates["last_source_time"] + pd.to_timedelta(source_minutes, unit="min")
            aggregates["rsi"] = _rsi(aggregates["close"])
            aggregates["macd"] = _macd(aggregates["close"])
            ema20 = aggregates["close"].ewm(span=20, adjust=False, min_periods=20).mean()
            aggregates["trend"] = np.sign(aggregates["close"] - ema20)

            left = pd.DataFrame(
                {
                    "_feature_position": symbol_rows["_feature_position"].to_numpy(),
                    "prediction_time": prediction_times.to_numpy(),
                }
            ).sort_values("prediction_time")
            right_columns = ["available_at", "rsi", "macd", "trend"]
            aligned = pd.merge_asof(
                left,
                aggregates[right_columns].sort_values("available_at"),
                left_on="prediction_time",
                right_on="available_at",
                direction="backward",
                allow_exact_matches=True,
            )
            positions = aligned["_feature_position"].to_numpy(dtype=int)
            if prefix == "five_min":
                result.loc[result.index[positions], "five_min_rsi"] = aligned["rsi"].to_numpy()
                result.loc[result.index[positions], "five_min_macd"] = aligned["macd"].to_numpy()
            elif prefix == "fifteen_min":
                result.loc[result.index[positions], "fifteen_min_rsi"] = aligned["rsi"].to_numpy()
                result.loc[result.index[positions], "fifteen_min_macd"] = aligned["macd"].to_numpy()
            else:
                result.loc[result.index[positions], "one_hour_trend"] = aligned["trend"].to_numpy()
    return result


def add_intraday_features(
    df: pd.DataFrame,
    source_candles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add causal trend, momentum, volatility, session, and timeframe features."""
    result = df.copy()
    if result.empty:
        return result

    group_columns = [column for column in ("symbol_token", "timeframe") if column in result.columns]
    groups = result.groupby(group_columns, dropna=False, sort=False) if group_columns else None
    close = result["close"]

    for period in (20, 50, 200):
        source_column = f"trend_ema{period}"
        if source_column in result.columns:
            ema = result[source_column]
        elif groups is not None:
            ema = groups["close"].transform(
                lambda values, span=period: values.ewm(span=span, adjust=False, min_periods=span).mean()
            )
        else:
            ema = close.ewm(span=period, adjust=False, min_periods=period).mean()
        previous_ema = ema.groupby([result[column] for column in group_columns], dropna=False, sort=False).shift(1) if group_columns else ema.shift(1)
        result[f"ema{period}_slope"] = (ema - previous_ema).divide(previous_ema.replace(0, np.nan))
        result[f"price_above_ema{period}"] = close.ge(ema).where(ema.notna()).astype(float)

    if "rsi" in result.columns:
        for periods in (3, 5):
            result[f"rsi_change_{periods}"] = result.groupby(group_columns, dropna=False, sort=False)["rsi"].diff(periods) if group_columns else result["rsi"].diff(periods)
    else:
        result["rsi_change_3"] = np.nan
        result["rsi_change_5"] = np.nan
    result["macd_change"] = result.groupby(group_columns, dropna=False, sort=False)["macd"].diff() if group_columns and "macd" in result.columns else (result["macd"].diff() if "macd" in result.columns else np.nan)
    for periods in (3, 5, 10):
        result[f"momentum_{periods}"] = (
            result.groupby(group_columns, dropna=False, sort=False)["close"].pct_change(periods)
            if group_columns
            else close.pct_change(periods)
        )

    close_returns = result.groupby(group_columns, dropna=False, sort=False)["close"].pct_change() if group_columns else close.pct_change()
    for window in (5, 20):
        result[f"rolling_std_{window}"] = close_returns.groupby([result[column] for column in group_columns], dropna=False, sort=False).transform(
            lambda values, size=window: values.rolling(size, min_periods=size).std(ddof=0)
        ) if group_columns else close_returns.rolling(window, min_periods=window).std(ddof=0)

    if {"high", "low"}.issubset(result.columns):
        previous_close = result.groupby(group_columns, dropna=False, sort=False)["close"].shift(1) if group_columns else close.shift(1)
        true_range = pd.concat(
            [result["high"] - result["low"], (result["high"] - previous_close).abs(), (result["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr_short = true_range.groupby([result[column] for column in group_columns], dropna=False, sort=False).transform(
            lambda values: values.rolling(5, min_periods=5).mean()
        ) if group_columns else true_range.rolling(5, min_periods=5).mean()
        atr_long = true_range.groupby([result[column] for column in group_columns], dropna=False, sort=False).transform(
            lambda values: values.rolling(20, min_periods=20).mean()
        ) if group_columns else true_range.rolling(20, min_periods=20).mean()
        result["atr_ratio_short_long"] = atr_short.divide(atr_long.replace(0, np.nan))
    else:
        result["atr_ratio_short_long"] = np.nan
    result["volatility_expansion"] = result["rolling_std_5"].divide(result["rolling_std_20"].replace(0, np.nan))

    if "candle_time" in result.columns:
        candle_time = pd.to_datetime(result["candle_time"])
        minute_of_day = candle_time.dt.hour * 60 + candle_time.dt.minute
        result["minute_of_day"] = minute_of_day.astype(float)
        result["minute_from_open"] = (minute_of_day - SESSION_OPEN_MINUTES).astype(float)
        result["session_progress_pct"] = result["minute_from_open"].clip(lower=0, upper=SESSION_LENGTH_MINUTES) / SESSION_LENGTH_MINUTES
        session_keys = [column for column in ("symbol_token", "timeframe") if column in result.columns]
        dates = candle_time.dt.date
        session_keys.append("_feature_session_date")
        result["_feature_session_date"] = dates
        session_groups = result.groupby(session_keys, dropna=False, sort=False)
        if "volume" in result.columns:
            typical_price = (result["high"] + result["low"] + close) / 3 if {"high", "low"}.issubset(result.columns) else close
            cumulative_value = (typical_price * result["volume"]).groupby(
                [result[column] for column in session_keys], dropna=False, sort=False
            ).cumsum()
            cumulative_volume = result["volume"].groupby(
                [result[column] for column in session_keys], dropna=False, sort=False
            ).cumsum()
            vwap = cumulative_value.divide(cumulative_volume.replace(0, np.nan))
            result["price_above_vwap"] = close.ge(vwap).where(vwap.notna()).astype(float)
        else:
            result["price_above_vwap"] = np.nan
        running_high = session_groups["high"].cummax()
        running_low = session_groups["low"].cummin()
        result["distance_from_day_high"] = (close - running_high).divide(running_high.replace(0, np.nan))
        result["distance_from_day_low"] = (close - running_low).divide(running_low.replace(0, np.nan))
        previous_high = running_high.groupby([result[column] for column in session_keys], dropna=False, sort=False).shift(1)
        previous_low = running_low.groupby([result[column] for column in session_keys], dropna=False, sort=False).shift(1)
        result["breakout_day_high"] = result["high"].gt(previous_high).where(previous_high.notna(), False).astype(float)
        result["breakout_day_low"] = result["low"].lt(previous_low).where(previous_low.notna(), False).astype(float)
        result = result.drop(columns="_feature_session_date")
    else:
        for name in ("minute_of_day", "minute_from_open", "session_progress_pct", "price_above_vwap", "distance_from_day_high", "distance_from_day_low", "breakout_day_high", "breakout_day_low"):
            result[name] = np.nan

    result = _multi_timeframe_features(result, source_candles)
    result = _add_regime_features(result)
    return result.replace([np.inf, -np.inf], np.nan)


def _add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    groups = [column for column in ("symbol_token", "timeframe") if column in result.columns]
    grouped_close = result.groupby(groups, dropna=False, sort=False)["close"] if groups else None

    def transform(function):
        return grouped_close.transform(function) if grouped_close is not None else function(result["close"])

    returns = transform(lambda values: values.pct_change())
    if "candle_time" in result.columns and groups:
        dates = pd.to_datetime(result["candle_time"]).dt.date
        prior_dates = dates.groupby([result[column] for column in groups], sort=False, dropna=False).shift(1)
        returns = returns.where(dates.eq(prior_dates), 0.0)
    absolute_returns = returns.abs()
    path_length = absolute_returns.groupby(
        [result[column] for column in groups], dropna=False, sort=False
    ).transform(lambda values: values.rolling(20, min_periods=20).sum()) if groups else absolute_returns.rolling(20, min_periods=20).sum()
    cumulative_return = (1.0 + returns).groupby(
        [result[column] for column in groups], dropna=False, sort=False
    ).transform(lambda values: values.rolling(20, min_periods=20).apply(np.prod, raw=True) - 1.0) if groups else (1.0 + returns).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1.0
    displacement = cumulative_return.abs()
    efficiency = displacement.divide(path_length.replace(0, np.nan))
    result["market_regime"] = np.where(efficiency >= 0.35, 1.0, 0.0)
    result.loc[efficiency.isna(), "market_regime"] = np.nan

    volatility_expansion = result["volatility_expansion"]
    result["volatility_regime"] = np.select(
        [volatility_expansion >= 1.25, volatility_expansion <= 0.80],
        [1.0, -1.0],
        default=0.0,
    )
    result.loc[volatility_expansion.isna(), "volatility_regime"] = np.nan

    ema20 = result.get("trend_ema20")
    ema50 = result.get("trend_ema50")
    if ema20 is None:
        ema20 = transform(lambda values: values.ewm(span=20, adjust=False, min_periods=20).mean())
    if ema50 is None:
        ema50 = transform(lambda values: values.ewm(span=50, adjust=False, min_periods=50).mean())
    spread = (ema20 - ema50).divide(result["close"].abs().replace(0, np.nan))
    result["trend_regime"] = np.select([spread > 0.0005, spread < -0.0005], [1.0, -1.0], default=0.0)
    result.loc[spread.isna(), "trend_regime"] = np.nan
    return result