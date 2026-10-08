from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import datetime

import pandas as pd

from app.dataset.label_generator import LabelGenerator
from app.dataset.target_policy import TargetPolicyFactory
from app.domain.trading_style import TradingStyle
from app.features.feature_config import FEATURE_VERSION
from app.features.feature_engineering import FeatureEngineering
from app.features.feature_schema import FEATURE_SCHEMA
from app.indicators.indicator_calculator import IndicatorCalculator
from app.market_data.market_data_provider import MarketDataProvider
from app.schemas.dataset_schema import DatasetSummary

logger = logging.getLogger(__name__)


class DatasetGenerator:
    FEATURE_SCHEMA = FEATURE_SCHEMA
    FEATURE_COLUMNS = list(FEATURE_SCHEMA.columns)

    def __init__(self) -> None:
        self.market_data_provider = MarketDataProvider()
        self.feature_engineering = FeatureEngineering()
        self.label_generator = LabelGenerator()
        self.supervised_rows: pd.DataFrame | None = None

    def generate_dataset(self,symbol_token: str,timeframe: str, prediction_horizon_bars: int,buy_threshold_pct: float = 0.5,
        sell_threshold_pct: float = -0.5,trading_style: TradingStyle | str = TradingStyle.INTRADAY,
        start_time: datetime | None = None,end_time: datetime | None = None,) -> DatasetSummary:
        
      
        
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
        labeled_df = self._add_supervised_metadata(
            labeled_df,
            prediction_horizon_bars=prediction_horizon_bars,
            trading_style=normalized_style,
        )

        # Filter out rows with incomplete features or invalid targets. The final
        # dataset is guaranteed to have all declared v1 features and valid labels.
        cleaned_df, filtering_diagnostics = self._finalize_dataset(labeled_df)
        self.supervised_rows = cleaned_df

        label_distribution = {
            "BUY": int((cleaned_df["label"] == "BUY").sum()),
            "HOLD": int((cleaned_df["label"] == "HOLD").sum()),
            "SELL": int((cleaned_df["label"] == "SELL").sum()),
        }
        skip_reasons = self._primary_skip_reasons(
            labeled_df,
            prediction_horizon_bars=prediction_horizon_bars,
            trading_style=normalized_style,
        )
        feature_schema = self.FEATURE_SCHEMA.as_dict()
        dataset_start_time = cleaned_df["candle_time"].min().isoformat() if not cleaned_df.empty else None
        dataset_end_time = cleaned_df["candle_time"].max().isoformat() if not cleaned_df.empty else None
        dataset_fingerprint = self._dataset_fingerprint(
            symbol_token=symbol_token,
            trading_style=normalized_style,
            timeframe=timeframe,
            prediction_horizon_bars=prediction_horizon_bars,
            buy_threshold_pct=buy_threshold_pct,
            sell_threshold_pct=sell_threshold_pct,
            feature_version=FEATURE_VERSION,
            feature_columns=list(self.FEATURE_COLUMNS),
            dataset_start_time=dataset_start_time,
            dataset_end_time=dataset_end_time,
            dataset_row_count=int(len(cleaned_df)),
        )
        feature_validation = self._feature_validation_audit(
            labeled_df,
            cleaned_df,
            filtering_diagnostics,
        )
        training_eligibility = self._training_eligibility(
            dataset_row_count=int(len(cleaned_df)),
            feature_count=len(self.FEATURE_COLUMNS),
            feature_validation=feature_validation,
            label_distribution=label_distribution,
            skip_reasons=skip_reasons,
        )

        summary = DatasetSummary(
            symbolToken=symbol_token,
            tradingStyle=normalized_style.value,
            timeframe=timeframe,
            predictionHorizonBars=prediction_horizon_bars,
            sourceTimeframe=market_data.diagnostics.source_timeframe,
            sourceRowCount=int(market_data.diagnostics.source_row_count),
            resampledRowCount=market_data.diagnostics.resampled_row_count,
            partialCandleCount=market_data.diagnostics.partial_candle_count,
            droppedPartialCandleCount=market_data.diagnostics.dropped_partial_candle_count,
            indicatorWarmupRows=filtering_diagnostics["indicator_warmup_rows"],
            datasetRowCount=int(len(cleaned_df)),
            skippedRowCount=filtering_diagnostics["target_skipped_rows"],
            featureCount=len(self.FEATURE_COLUMNS),
            featureVersion=FEATURE_VERSION,
            labelDistribution=label_distribution,
            featureSchema=feature_schema,
            skipReasons=skip_reasons,
            trainingEligibility=training_eligibility,
            featureValidation=feature_validation,
            datasetFingerprint=dataset_fingerprint,
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
    def _add_supervised_metadata(
        df: pd.DataFrame,
        *,
        prediction_horizon_bars: int,
        trading_style: TradingStyle | str,
    ) -> pd.DataFrame:
        result = df.copy()
        result["featureTimestamp"] = pd.to_datetime(result["candle_time"])
        group_columns = ["symbol_token", "timeframe"]
        if TradingStyle.normalize(trading_style) == TradingStyle.INTRADAY and "trading_date" in result.columns:
            group_columns.append("trading_date")
        target_times = result.groupby(group_columns, dropna=False)["candle_time"].shift(
            -prediction_horizon_bars
        )
        result["targetEndTimestamp"] = pd.to_datetime(target_times)
        return result

    def _label_rows(self, df: pd.DataFrame, buy_threshold_pct: float, sell_threshold_pct: float) -> pd.DataFrame:
        result = df.copy()
        result["label"] = "HOLD"
        result.loc[result["future_return_pct"] >= buy_threshold_pct, "label"] = "BUY"
        result.loc[result["future_return_pct"] <= sell_threshold_pct, "label"] = "SELL"
        return result

    def _primary_skip_reasons(
        self,
        df: pd.DataFrame,
        prediction_horizon_bars: int,
        trading_style: TradingStyle | str = TradingStyle.INTRADAY,
    ) -> dict[str, int]:
        target_invalid = (
            df["future_close"].isna() | df["future_return_pct"].isna()
            if "future_close" in df.columns and "future_return_pct" in df.columns
            else pd.Series(False, index=df.index)
        )
        if not target_invalid.any():
            return {}

        normalized_style = TradingStyle.normalize(trading_style)
        group_columns = ["symbol_token", "timeframe"]
        if normalized_style == TradingStyle.INTRADAY and "trading_date" in df.columns:
            group_columns.append("trading_date")
        group_sizes = df.groupby(group_columns, dropna=False).size()
        insufficient_rows = pd.Series(False, index=df.index)
        if set(group_columns).issubset(df.columns):
            session_sizes = df.groupby(group_columns, dropna=False).transform("size")
            insufficient_rows = target_invalid & (session_sizes < prediction_horizon_bars)
        cross_session_rows = target_invalid & ~insufficient_rows
        return {
            "crossSessionHorizon": int(cross_session_rows.sum()),
            "insufficientFutureBars": int(insufficient_rows.sum()),
        }

    def _feature_validation(self, df: pd.DataFrame) -> dict[str, int]:
        feature_columns = list(self.FEATURE_COLUMNS)
        if not feature_columns:
            return {
                "rowsWithInvalidFeatures": 0,
                "nullValueCount": 0,
                "nanValueCount": 0,
                "positiveInfinityCount": 0,
                "negativeInfinityCount": 0,
            }
        available = [column for column in feature_columns if column in df.columns]
        values = df[available]
        null_value_count = int(
            values.apply(
                lambda column: column.map(
                    lambda value: value is None or value is pd.NA
                )
            ).sum().sum()
        )
        nan_value_count = int(
            values.apply(
                lambda column: column.map(
                    lambda value: isinstance(value, float) and math.isnan(value)
                )
            ).sum().sum()
        )
        positive_infinity_count = int((values == float("inf")).sum().sum())
        negative_infinity_count = int((values == float("-inf")).sum().sum())
        invalid_rows = (
            values.isna().any(axis=1)
            | (values.eq(float("inf")) | values.eq(float("-inf"))).any(axis=1)
        )
        return {
            "rowsWithInvalidFeatures": int(invalid_rows.sum()),
            "nullValueCount": null_value_count,
            "nanValueCount": nan_value_count,
            "positiveInfinityCount": positive_infinity_count,
            "negativeInfinityCount": negative_infinity_count,
        }

    def _feature_validation_audit(
        self,
        source_df: pd.DataFrame,
        final_df: pd.DataFrame,
        diagnostics: dict[str, int],
    ) -> dict[str, object]:
        before = self._feature_validation(source_df)
        after = self._feature_validation(final_df)
        feature_warmup_rows = 0
        if all(column in source_df.columns for column in ["volume", "relative_volume", "rolling_volume_mean"]):
            feature_warmup_rows = int(
                source_df["relative_volume"].isna().mul(source_df["volume"].eq(0)).sum()
            )
        return {
            "beforeFiltering": before,
            "afterFiltering": after,
            "featureWarmupRows": feature_warmup_rows,
        }

    @staticmethod
    def _dataset_fingerprint(
        symbol_token: str,
        trading_style: TradingStyle | str,
        timeframe: str,
        prediction_horizon_bars: int,
        buy_threshold_pct: float,
        sell_threshold_pct: float,
        feature_version: str,
        feature_columns: list[str],
        dataset_start_time: str | None,
        dataset_end_time: str | None,
        dataset_row_count: int,
    ) -> str:
        canonical_payload = {
            "symbolToken": symbol_token,
            "tradingStyle": TradingStyle.normalize(trading_style).value,
            "timeframe": timeframe,
            "predictionHorizonBars": prediction_horizon_bars,
            "buyThresholdPct": buy_threshold_pct,
            "sellThresholdPct": sell_threshold_pct,
            "featureVersion": feature_version,
            "featureColumns": list(feature_columns),
            "datasetStartTime": dataset_start_time,
            "datasetEndTime": dataset_end_time,
            "datasetRowCount": dataset_row_count,
        }
        serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _training_eligibility(
        dataset_row_count: int,
        feature_count: int,
        feature_validation: dict[str, object],
        label_distribution: dict[str, int],
        skip_reasons: dict[str, int],
    ) -> dict[str, object]:
        reasons = []
        if dataset_row_count == 0:
            reasons.append("EMPTY_DATASET")
        if feature_count == 0:
            reasons.append("NO_FEATURES")
        if feature_validation["afterFiltering"]["rowsWithInvalidFeatures"] > 0:
            reasons.append("INVALID_FINAL_FEATURES")
        if sum(label_distribution.values()) != dataset_row_count:
            reasons.append("LABEL_COUNT_MISMATCH")
        if skip_reasons.get("insufficientFutureBars", 0) > 0:
            reasons.append("INSUFFICIENT_FUTURE_BARS")
        return {"eligible": not reasons, "reasons": reasons}

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
