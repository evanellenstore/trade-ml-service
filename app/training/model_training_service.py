from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.settings import settings
from app.dataset.dataset_generator import DatasetGenerator
from app.dataset.split.dataset_split_config import DatasetSplitConfig
from app.dataset.split.purged_chronological_dataset_splitter import PurgedChronologicalDatasetSplitter
from app.domain.trading_style import TradingStyle
from app.features.feature_schema import FEATURE_SCHEMA
from app.model.model_registry import model_path_for
from app.training.xgboost_trainer import XGBoostTrainer


class ModelTrainingService:
    """Generate real rows, chronologically split them, train, and persist a model."""

    def __init__(
        self,
        *,
        generator: DatasetGenerator | None = None,
        splitter: PurgedChronologicalDatasetSplitter | None = None,
        trainer: XGBoostTrainer | None = None,
        model_directory: str | None = None,
    ) -> None:
        self.model_directory = model_directory or settings.model_directory
        self.generator = generator or DatasetGenerator()
        self.splitter = splitter or PurgedChronologicalDatasetSplitter()
        self.trainer = trainer or XGBoostTrainer(
            model_path=model_path_for(
                TradingStyle.INTRADAY,
                self.model_directory,
            )
        )

    def train(
        self,
        *,
        symbol_token: str,
        timeframe: str,
        prediction_horizon_bars: int,
        buy_threshold_pct: float,
        sell_threshold_pct: float,
        trading_style: TradingStyle | str = TradingStyle.INTRADAY,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_enabled: bool = True,
        embargo_bars: int = 0,
    ) -> dict[str, Any]:
        dataset_summary = self.generator.generate_dataset(
            symbol_token=symbol_token,
            timeframe=timeframe,
            prediction_horizon_bars=prediction_horizon_bars,
            buy_threshold_pct=buy_threshold_pct,
            sell_threshold_pct=sell_threshold_pct,
            trading_style=trading_style,
            start_time=start_time,
            end_time=end_time,
        )
        
        print(f"---------- Dataset summary: {dataset_summary}")  # Debugging line to print the dataset summary
        
        rows = self.generator.supervised_rows
        if rows is None or rows.empty:
            raise ValueError("Dataset generation did not produce training rows")

        split_config = DatasetSplitConfig(
            trainRatio=train_ratio,
            validationRatio=validation_ratio,
            testRatio=test_ratio,
            purgeEnabled=purge_enabled,
            embargoBars=embargo_bars,
        )
        print(f"---------- Split config: {split_config}")  # Debugging line to print the split config
        
        split_result = self.splitter.split(
            rows,
            split_config,
            dataset_fingerprint=dataset_summary.datasetFingerprint or "",
            prediction_horizon_bars=prediction_horizon_bars,
            trading_style=trading_style,
        )

        print(f"---------- Split result: {split_result}")  # Debugging line to print the split result
        train_df, validation_df, test_df = self.splitter.split_frames(
            rows,
            split_config,
            dataset_fingerprint=dataset_summary.datasetFingerprint or "",
            prediction_horizon_bars=prediction_horizon_bars,
            trading_style=trading_style,
        )
        
        print(f"---------- Train DataFrame: {train_df}")  # Debugging line to print the train DataFrame
        print(f"---------- Validation DataFrame: {validation_df}")  # Debugging line to print the validation DataFrame
        print(f"---------- Test DataFrame: {test_df}")
        
        if train_df.empty or validation_df.empty or test_df.empty:
            raise ValueError("Chronological split must contain train, validation, and test rows")

        normalized_style = TradingStyle.normalize(trading_style)
        style_model_path = model_path_for(normalized_style, self.model_directory)
        if self.trainer.model_path != style_model_path:
            self.trainer = XGBoostTrainer(model_path=style_model_path)

        feature_columns = list(FEATURE_SCHEMA.columns)
        trainer_result = self.trainer.train(
            train_df=train_df,
            validation_df=validation_df,
            feature_columns=feature_columns,
            label_column="label",
        )
        return {
            "dataset": dataset_summary.model_dump(mode="json"),
            "split": split_result.model_dump(mode="json"),
            "training": trainer_result,
            "feature_columns": feature_columns,
        }
