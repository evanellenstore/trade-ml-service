from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.config.database import SessionLocal

logger = logging.getLogger(__name__)


class MarketDataRepository:
    """Repository for loading candle and indicator data from the existing MySQL tables."""

    def fetch_raw_candles(
        self,
        symbol_token: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if not symbol_token or not symbol_token.strip():
            raise ValueError("symbolToken is required")

        query = """
            SELECT
                m.id AS candle_id,
                m.symbol_token AS symbol_token,
                m.timeframe AS timeframe,
                m.candle_time AS candle_time,
                m.open AS open,
                m.high AS high,
                m.low AS low,
                m.close AS close,
                m.volume AS volume
            FROM market_candles_backtest m
            WHERE m.symbol_token = :symbol_token
              AND m.timeframe = 'ONE_MINUTE'
        """
        params: dict[str, object] = {"symbol_token": symbol_token}
        if start_time is not None:
            query += " AND m.candle_time >= :start_time"
            params["start_time"] = start_time
        if end_time is not None:
            query += " AND m.candle_time <= :end_time"
            params["end_time"] = end_time
        query += " ORDER BY m.symbol_token, m.candle_time ASC"
        if limit is not None:
            query += " LIMIT :limit"
            params["limit"] = limit

        with SessionLocal() as session:
            result = session.execute(text(query), params)
            rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=[
                "candle_id", "symbol_token", "timeframe", "candle_time",
                "open", "high", "low", "close", "volume",
            ])
        df = pd.DataFrame(rows, columns=result.keys())
        df["candle_time"] = pd.to_datetime(df["candle_time"])
        return df

    def fetch_market_data(
        self,
        symbol_token: str,
        timeframe: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if not symbol_token or not symbol_token.strip():
            raise ValueError("symbolToken is required")

        query = """
            SELECT
                m.id AS candle_id,
                m.symbol_token AS symbol_token,
                m.timeframe AS timeframe,
                m.candle_time AS candle_time,
                m.open AS open,
                m.high AS high,
                m.low AS low,
                m.close AS close,
                m.volume AS volume,
                i.id AS indicator_id,
                i.created_at AS indicator_created_at,
                i.momentum_cci,
                i.momentum_macd,
                i.momentum_macd_histogram,
                i.momentum_macd_signal,
                i.momentum_roc,
                i.momentum_rsi14,
                i.momentum_stochasticd,
                i.momentum_stochastick,
                i.origin,
                i.pivot,
                i.resistance1,
                i.resistance2,
                i.run_id,
                i.support1,
                i.support2,
                i.symbol,
                i.trend_adx,
                i.trend_ema,
                i.trend_ema100,
                i.trend_ema20,
                i.trend_ema200,
                i.trend_ema50,
                i.trend_minus_di,
                i.trend_plus_di,
                i.trend_supertrend,
                i.volatility_atr,
                i.volatility_bb_lower,
                i.volatility_bb_middle,
                i.volatility_bb_upper,
                i.volatility_bb_width,
                i.volatility_percentb,
                i.volume_cmf,
                i.volume_mfi,
                i.volume_obv,
                i.volume_vwap
            FROM market_candles_backtest m
            INNER JOIN market_indicators_backtest i
                ON m.id = i.candle_id
            WHERE m.symbol_token = :symbol_token
              AND m.timeframe = :timeframe
        """

        params: dict[str, object] = {"symbol_token": symbol_token, "timeframe": timeframe}

        if start_time is not None:
            query += " AND m.candle_time >= :start_time"
            params["start_time"] = start_time
        if end_time is not None:
            query += " AND m.candle_time <= :end_time"
            params["end_time"] = end_time

        query += " ORDER BY m.symbol_token, m.timeframe, m.candle_time ASC"

        if limit is not None:
            query += " LIMIT :limit"
            params["limit"] = limit

        with SessionLocal() as session:
            result = session.execute(text(query), params)
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame(columns=[
                "candle_id",
                "symbol_token",
                "timeframe",
                "candle_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ])

        df = pd.DataFrame(rows, columns=result.keys())
        if "candle_time" in df.columns:
            df["candle_time"] = pd.to_datetime(df["candle_time"])
        return df
