from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from app.dataset.label_generator import LabelGenerator
from app.domain.trading_style import TradingStyle
from app.market_data.market_data_provider import MarketDataProvider
from app.schemas.target_analysis_schema import (
    ClassMetric,
    DistributionStatistics,
    HorizonAnalysis,
    TargetAnalysisData,
    ThresholdAnalysis,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.repository.market_data_repository import MarketDataRepository


class TargetAnalysisService:
    def __init__(self, repository: "MarketDataRepository | None" = None) -> None:
        if repository is None:
            from app.repository.market_data_repository import MarketDataRepository

            repository = MarketDataRepository()
        self.market_data_provider = MarketDataProvider(repository=repository)
        self.label_generator = LabelGenerator()

    def analyze(
        self,
        symbol_token: str,
        timeframe: str,
        prediction_horizons_bars: list[int],
        thresholds_pct: list[float] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        trading_style: TradingStyle | str = TradingStyle.INTRADAY,
    ) -> TargetAnalysisData:
        prediction_horizons_bars = sorted(set(prediction_horizons_bars))

        if not prediction_horizons_bars or any(horizon <= 0 for horizon in prediction_horizons_bars):
            raise ValueError("predictionHorizonsBars must contain only values greater than zero")
        if thresholds_pct is None or not thresholds_pct or any(not math.isfinite(threshold) or threshold <= 0 for threshold in thresholds_pct):
            raise ValueError("thresholdsPct must contain only positive values")
        thresholds_pct = sorted(set(thresholds_pct))
        trading_style = TradingStyle.normalize(trading_style)

        source_df = self.market_data_provider.get_market_data(
            symbol_token, timeframe, start_time, end_time
        ).data
        source_row_count = len(source_df)
        if source_row_count == 0:
            raise ValueError("No market data found for the requested symbol and timeframe")
        if "close" not in source_df.columns or not pd.api.types.is_numeric_dtype(source_df["close"]):
            raise ValueError("close must be numeric")
        if not np.isfinite(source_df["close"].to_numpy(dtype=float)).all():
            raise ValueError("close contains invalid numbers")

        logger.info(
            "Target analysis started: symbolToken=%s tradingStyle=%s timeframe=%s sourceRowCount=%s horizons=%s thresholdsPct=%s",
            symbol_token,
            trading_style.value,
            timeframe,
            source_row_count,
            prediction_horizons_bars,
            thresholds_pct,
        )

        analyses = []
        for horizon in prediction_horizons_bars:
            if horizon >= source_row_count:
                raise ValueError(
                    f"Insufficient candles for prediction horizon {horizon}; at least {horizon + 1} candles are required"
                )

            labeled_df = self.label_generator.calculate_future_returns(
                source_df,
                prediction_horizon_bars=horizon,
                trading_style=trading_style,
            )
            target_rows = labeled_df["future_close"].notna()
            invalid_returns = labeled_df.loc[target_rows, "future_return_pct"].replace(
                [np.inf, -np.inf], np.nan
            ).isna()
            if invalid_returns.any():
                raise ValueError(f"Invalid future return values detected for horizon {horizon}")
            valid_returns = labeled_df.loc[target_rows, "future_return_pct"]
            valid_target_rows = len(valid_returns)
            skipped_rows = source_row_count - valid_target_rows
            if valid_target_rows + skipped_rows != source_row_count:
                raise ValueError(f"Target row validation failed for horizon {horizon}")

            statistics = self._statistics(valid_returns)
            threshold_analysis = [
                self._threshold_analysis(valid_returns, threshold)
                for threshold in thresholds_pct
            ]
            logger.info(
                "Target analysis horizon: symbolToken=%s tradingStyle=%s timeframe=%s horizon=%s validTargetRows=%s skippedRows=%s mean=%s std=%s",
                symbol_token,
                trading_style.value,
                timeframe,
                horizon,
                valid_target_rows,
                skipped_rows,
                statistics.mean,
                statistics.std,
            )
            analyses.append(
                HorizonAnalysis(
                    predictionHorizonBars=horizon,
                    validTargetRows=valid_target_rows,
                    skippedRows=skipped_rows,
                    returnStatistics=statistics,
                    thresholdAnalysis=threshold_analysis,
                )
            )

        return TargetAnalysisData(
            symbolToken=symbol_token,
            tradingStyle=trading_style.value,
            timeframe=timeframe,
            sourceRowCount=source_row_count,
            analysis=analyses,
        )

    @staticmethod
    def _statistics(values: pd.Series) -> DistributionStatistics:
        if values.empty:
            return DistributionStatistics(**{field: 0 for field in DistributionStatistics.model_fields})
        percentiles = values.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        return DistributionStatistics(
            count=len(values),
            min=round(float(values.min()), 6),
            p01=round(float(percentiles.loc[0.01]), 6),
            p05=round(float(percentiles.loc[0.05]), 6),
            p10=round(float(percentiles.loc[0.10]), 6),
            p25=round(float(percentiles.loc[0.25]), 6),
            p50=round(float(percentiles.loc[0.50]), 6),
            p75=round(float(percentiles.loc[0.75]), 6),
            p90=round(float(percentiles.loc[0.90]), 6),
            p95=round(float(percentiles.loc[0.95]), 6),
            p99=round(float(percentiles.loc[0.99]), 6),
            max=round(float(values.max()), 6),
            mean=round(float(values.mean()), 6),
            std=round(float(values.std()), 6) if len(values) > 1 else 0.0,
        )

    @staticmethod
    def _threshold_analysis(values: pd.Series, threshold: float) -> ThresholdAnalysis:
        buy_count = int((values >= threshold).sum())
        sell_count = int((values <= -threshold).sum())
        hold_count = int(len(values) - buy_count - sell_count)
        total = len(values)
        percentages = [count / total * 100.0 for count in (buy_count, hold_count, sell_count)] if total else [0.0] * 3
        majority = max(percentages)
        minority = min(percentages)
        return ThresholdAnalysis(
            thresholdPct=threshold,
            buy=ClassMetric(count=buy_count, percentage=round(percentages[0], 6)),
            hold=ClassMetric(count=hold_count, percentage=round(percentages[1], 6)),
            sell=ClassMetric(count=sell_count, percentage=round(percentages[2], 6)),
            directional=ClassMetric(
                count=buy_count + sell_count,
                percentage=round((buy_count + sell_count) / total * 100.0, 6) if total else 0.0,
            ),
            classImbalanceRatio=round(majority / minority, 6) if minority else None,
            majorityClassPct=round(majority, 6),
            minorityClassPct=round(minority, 6),
        )