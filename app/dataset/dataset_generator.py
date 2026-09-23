from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd

from app.dataset.label_generator import LabelGenerator
from app.dataset.target_policy import TargetPolicyFactory
from app.domain.trading_style import TradingStyle
from app.features.feature_config import FEATURE_VERSION
from app.features.feature_engineering import FeatureEngineering
from app.indicators.indicator_calculator import IndicatorCalculator
from app.market_data.market_data_provider import MarketDataProvider
from app.schemas.dataset_schema import DatasetSummary

logger = logging.getLogger(__name__)


class DatasetGenerator:
    FEATURE_COLUMNS = [
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
        "is_doji",
        "is_hammer",
        "is_shooting_star",
        "is_bullish_engulfing",
        "is_bearish_engulfing",
    ]

    def __init__(self) -> None:
        self.market_data_provider = MarketDataProvider()
        self.feature_engineering = FeatureEngineering()
        self.label_generator = LabelGenerator()

    def generate_dataset(self,symbol_token: str,timeframe: str, prediction_horizon_bars: int,buy_threshold_pct: float = 0.5,
        sell_threshold_pct: float = -0.5,trading_style: TradingStyle | str = TradingStyle.INTRADAY,
        start_time: datetime | None = None,end_time: datetime | None = None,) -> DatasetSummary:
        
        """Generate a dataset using bars from the requested market timeframe.

        ONE_MINUTE uses the existing database candle and indicator path.
        Higher timeframes are prepared by MarketDataProvider from raw
        one-minute candles. The prediction horizon always counts bars in the
        requested timeframe, not minutes.
        """
        
        started_at = time.time()
        normalized_style = TradingStyle.normalize(trading_style)
        if prediction_horizon_bars <= 0:
            raise ValueError("predictionHorizonBars must be greater than zero")

        # The provider keeps ONE_MINUTE on the existing database-indicator path
        # and prepares higher-timeframe candles and indicators in memory.
        market_data = self.market_data_provider.get_market_data(symbol_token, timeframe, start_time, end_time)
        source_df = market_data.data

        if source_df.empty:
            raise ValueError("No market data found for the requested symbol and timeframe")

        source_df = self._validate_and_order(source_df)
        feature_df = self.feature_engineering.transform(source_df)

        # Calculate targets before filtering feature warm-up rows so the
        # horizon always refers to actual market bars in the requested timeframe.

        labeled_df = self.label_generator.generate_labels(feature_df,prediction_horizon_bars=prediction_horizon_bars,buy_threshold_pct=buy_threshold_pct,sell_threshold_pct=sell_threshold_pct,trading_style=normalized_style)

        # Filter out rows with incomplete features or invalid targets. The final
        # dataset is guaranteed to have all declared v1 features and valid labels.
        cleaned_df, filtering_diagnostics = self._finalize_dataset(labeled_df)

        label_distribution = {
            "BUY": int((cleaned_df["label"] == "BUY").sum()),
            "HOLD": int((cleaned_df["label"] == "HOLD").sum()),
            "SELL": int((cleaned_df["label"] == "SELL").sum()),
        }

        summary = DatasetSummary(symbolToken=symbol_token,tradingStyle=normalized_style.value,timeframe=timeframe,predictionHorizonBars=prediction_horizon_bars,
            sourceTimeframe=market_data.diagnostics.source_timeframe,
            sourceRowCount=int(market_data.diagnostics.source_row_count),resampledRowCount=market_data.diagnostics.resampled_row_count,partialCandleCount=market_data.diagnostics.partial_candle_count,
            droppedPartialCandleCount=market_data.diagnostics.dropped_partial_candle_count,
            indicatorWarmupRows=filtering_diagnostics["indicator_warmup_rows"],datasetRowCount=int(len(cleaned_df)),skippedRowCount=filtering_diagnostics["target_skipped_rows"],
            featureCount=len(self.FEATURE_COLUMNS),featureVersion=FEATURE_VERSION,labelDistribution=label_distribution,
        )

        logger.info("Dataset generation complete: symbolToken=%s tradingStyle=%s timeframe=%s predictionHorizonBars=%s sourceRowCount=%s featureCount=%s warmupRowsRemoved=%s invalidRowsRemoved=%s finalDatasetCount=%s BUY=%s HOLD=%s SELL=%s executionTime=%.2fs",symbol_token,normalized_style.value,timeframe,prediction_horizon_bars,len(source_df),len(self.FEATURE_COLUMNS),filtering_diagnostics["feature_invalid_rows"],filtering_diagnostics["target_skipped_rows"],len(cleaned_df),label_distribution["BUY"],label_distribution["HOLD"],label_distribution["SELL"],time.time() - started_at)
        return summary

    def _validate_and_order(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Dataset is empty")
        if "candle_id" not in df.columns or df["candle_id"].isnull().any():
            raise ValueError("Missing candle_id values detected")
        if "close" not in df.columns or df["close"].isnull().any():
            raise ValueError("Missing close values detected")
        if not pd.api.types.is_numeric_dtype(df["close"]):
            raise ValueError("close must be numeric")
        if not np_all_finite(df["close"]):
            raise ValueError("close contains invalid numbers")
        if df.duplicated(subset=["candle_id"]).any():
            raise ValueError("Duplicate candle_id records detected")

        df = df.sort_values(["symbol_token", "timeframe", "candle_time"]).reset_index(drop=True)
        return df

    def _finalize_dataset(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """Select rows with both complete v1 features and valid targets.

        Target columns are created before this method so removing feature
        warm-up rows cannot change bar-distance semantics. Final rows are
        selected with ``feature_ready AND target_valid``; metadata and unused
        indicator columns do not decide feature readiness.
        """
        required_columns = ["candle_id", "symbol_token", "timeframe", "candle_time", "close"]
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        feature_columns = list(self.FEATURE_COLUMNS)
        missing_features = [column for column in feature_columns if column not in df.columns]
        if missing_features:
            raise ValueError(f"Dataset missing required feature columns: {missing_features}")

        # Readiness is based only on the declared v1 model inputs. Auxiliary
        # indicator or metadata columns must not remove otherwise valid rows.
        feature_ready_mask = self._finite_columns_mask(df, feature_columns)
        indicator_feature_columns = [
            column
            for column in feature_columns
            if column in IndicatorCalculator.INDICATOR_COLUMNS
        ]
        indicator_warmup_mask = (
            ~self._finite_columns_mask(df, indicator_feature_columns)
            if indicator_feature_columns
            else pd.Series(False, index=df.index)
        )
        # Target validity is independent from feature readiness; both masks
        # must be true before a row can enter the training dataset.
        target_valid_mask = self._finite_columns_mask(
            df,
            ["close", "future_close", "future_return_pct", "label"],
        )
        final_valid_mask = feature_ready_mask & target_valid_mask
        final_df = df.loc[final_valid_mask].copy()

        if final_df.empty:
            raise ValueError("Dataset is empty after filtering invalid rows")

        metadata_columns = {"candle_id", "symbol_token", "timeframe", "candle_time", "trading_date"}
        final_columns = [
            column for column in required_columns + feature_columns + ["future_close", "future_return_pct", "label"]
            if column not in metadata_columns or column in final_df.columns
        ]
        final_df = final_df[final_columns].copy()
        feature_matrix = final_df[feature_columns]
        if not feature_matrix.apply(pd.api.types.is_numeric_dtype).all():
            raise ValueError("Dataset contains non-numeric feature columns")
        if not feature_matrix.replace([float("inf"), float("-inf")], float("nan")).notna().all().all():
            raise ValueError("Dataset contains invalid feature values")
        if final_df["label"].isna().any():
            raise ValueError("Dataset contains null labels")
        # Keep overlapping exclusions visible for diagnostics instead of
        # deriving dataset size by subtracting independent counts.
        diagnostics = {
            "indicator_warmup_rows": int(indicator_warmup_mask.sum()),
            "feature_invalid_rows": int((~feature_ready_mask).sum()),
            "target_skipped_rows": int((~target_valid_mask).sum()),
            "overlap_warmup_and_target_skipped_rows": int(
                ((~feature_ready_mask) & (~target_valid_mask)).sum()
            ),
        }
        return final_df.reset_index(drop=True), diagnostics

    @staticmethod
    def _finite_columns_mask(df: pd.DataFrame, columns: list[str]) -> pd.Series:
        values = df[columns].notna().all(axis=1)
        for column in columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                values &= pd.Series(
                    df[column].replace([float("inf"), float("-inf")], float("nan")),
                    index=df.index,
                ).notna()
        return values

    @staticmethod
    def resolve_target_policy(trading_style: TradingStyle | str):
        return TargetPolicyFactory.create(trading_style)


def np_all_finite(series: pd.Series) -> bool:
    return pd.Series(series).replace([float("inf"), float("-inf")], float("nan")).notna().all()
