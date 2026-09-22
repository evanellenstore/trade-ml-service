from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd

from app.dataset.label_generator import LabelGenerator
from app.features.feature_engineering import FeatureEngineering
from app.repository.market_data_repository import MarketDataRepository
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
        self.repository = MarketDataRepository()
        self.feature_engineering = FeatureEngineering()
        self.label_generator = LabelGenerator()

    def generate_dataset(
        self,
        symbol_token: str,
        timeframe: str,
        prediction_horizon: int,
        buy_threshold_pct: float,
        sell_threshold_pct: float,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> DatasetSummary:
        started_at = time.time()
        source_df = self.repository.fetch_market_data(symbol_token, timeframe, start_time, end_time)

        if source_df.empty:
            raise ValueError("No market data found for the requested symbol and timeframe")

        source_df = self._validate_and_order(source_df)
        feature_df = self.feature_engineering.transform(source_df)

        labeled_df = self.label_generator.generate_labels(
            feature_df,
            prediction_horizon=prediction_horizon,
            buy_threshold_pct=buy_threshold_pct,
            sell_threshold_pct=sell_threshold_pct,
        )

        cleaned_df = self._finalize_dataset(labeled_df, prediction_horizon)

        label_distribution = {
            "BUY": int((cleaned_df["label"] == "BUY").sum()),
            "HOLD": int((cleaned_df["label"] == "HOLD").sum()),
            "SELL": int((cleaned_df["label"] == "SELL").sum()),
        }

        summary = DatasetSummary(
            symbolToken=symbol_token,
            timeframe=timeframe,
            predictionHorizon=prediction_horizon,
            sourceRowCount=int(len(source_df)),
            datasetRowCount=int(len(cleaned_df)),
            skippedRowCount=int(len(source_df) - len(cleaned_df)),
            featureCount=len(self.FEATURE_COLUMNS),
            labelDistribution=label_distribution,
        )

        logger.info(
            "Dataset generation complete: symbolToken=%s timeframe=%s predictionHorizon=%s sourceRowCount=%s featureCount=%s warmupRowsRemoved=%s invalidRowsRemoved=%s finalDatasetCount=%s BUY=%s HOLD=%s SELL=%s executionTime=%.2fs",
            symbol_token,
            timeframe,
            prediction_horizon,
            len(source_df),
            len(self.FEATURE_COLUMNS),
            int(len(source_df) - len(cleaned_df)),
            0,
            len(cleaned_df),
            label_distribution["BUY"],
            label_distribution["HOLD"],
            label_distribution["SELL"],
            time.time() - started_at,
        )
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

    def _finalize_dataset(self, df: pd.DataFrame, prediction_horizon: int) -> pd.DataFrame:
        required_columns = ["candle_id", "symbol_token", "timeframe", "candle_time", "close"]
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        final_df = df.copy()
        final_df = final_df.dropna(subset=["close", "future_close", "future_return_pct"]).copy()
        final_df = final_df[final_df["future_close"].notna()].copy()

        final_df = final_df[final_df["future_return_pct"].notna()].copy()
        final_df = final_df[final_df["label"].notna()].copy()

        final_df = final_df[~final_df["label"].isna()].copy()
        if final_df.empty:
            raise ValueError("Dataset is empty after filtering invalid rows")

        feature_columns = [column for column in self.FEATURE_COLUMNS if column in final_df.columns]
        if not feature_columns:
            raise ValueError("No valid feature columns found after engineering")

        final_df = final_df[required_columns + feature_columns + ["future_close", "future_return_pct", "label"]].copy()
        return final_df.reset_index(drop=True)


def np_all_finite(series: pd.Series) -> bool:
    return pd.Series(series).replace([float("inf"), float("-inf")], float("nan")).notna().all()
