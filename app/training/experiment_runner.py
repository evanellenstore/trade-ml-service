from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.dataset.dataset_generator import DatasetGenerator
from app.dataset.split.dataset_split_config import DatasetSplitConfig
from app.dataset.split.purged_chronological_dataset_splitter import PurgedChronologicalDatasetSplitter
from app.domain.trading_style import TradingStyle
from app.features.feature_schema import FEATURE_SCHEMA
from app.training.model_evaluator import ModelEvaluator
from app.training.xgboost_trainer import XGBoostTrainer


DEFAULT_EXPERIMENTS = (
    {
        "experimentId": "A",
        "predictionHorizonBars": 10,
        "buyThresholdPct": 0.10,
        "sellThresholdPct": -0.10,
    },
    {
        "experimentId": "B",
        "predictionHorizonBars": 20,
        "buyThresholdPct": 0.10,
        "sellThresholdPct": -0.10,
    },
    {
        "experimentId": "C",
        "predictionHorizonBars": 30,
        "buyThresholdPct": 0.15,
        "sellThresholdPct": -0.15,
    },
    {
        "experimentId": "D",
        "predictionHorizonBars": 60,
        "buyThresholdPct": 0.20,
        "sellThresholdPct": -0.20,
    },
)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    prediction_horizon_bars: int
    buy_threshold_pct: float
    sell_threshold_pct: float
    timeframe: str = "ONE_MINUTE"
    trading_style: TradingStyle | str = TradingStyle.INTRADAY


class ExperimentRunner:
    """Run independent target/model experiments without touching the TEST set."""

    def __init__(
        self,
        *,
        generator: DatasetGenerator | None = None,
        splitter: PurgedChronologicalDatasetSplitter | None = None,
        trainer_factory: Any | None = None,
    ) -> None:
        self.generator = generator or DatasetGenerator()
        self.splitter = splitter or PurgedChronologicalDatasetSplitter()
        self.trainer_factory = trainer_factory or XGBoostTrainer

    @staticmethod
    def _canonical_label_order() -> list[str]:
        return ["SELL", "HOLD", "BUY"]

    @staticmethod
    def _safe_label_distribution(df: pd.DataFrame) -> dict[str, int]:
        labels = df["label"].astype(str)
        return {
            "BUY": int((labels == "BUY").sum()),
            "HOLD": int((labels == "HOLD").sum()),
            "SELL": int((labels == "SELL").sum()),
        }

    @staticmethod
    def _validation_period(df: pd.DataFrame) -> dict[str, str | None]:
        if df.empty:
            return {"start": None, "end": None}
        return {
            "start": df["candle_time"].min().isoformat(),
            "end": df["candle_time"].max().isoformat(),
        }

    @staticmethod
    def _confidence_analysis_for_validation(
        y_true: pd.Series,
        probabilities: Any,
        *,
        labels: list[str],
    ) -> list[dict[str, Any]]:
        analysis = ModelEvaluator.confidence_analysis(
            y_true=y_true.astype(str),
            probabilities=probabilities,
            labels=labels,
            thresholds=[0.40, 0.50, 0.60, 0.70, 0.80],
        )
        predicted_labels = [labels[int(np.argmax(row))] for row in probabilities]
        predicted_series = pd.Series(predicted_labels, dtype=object)
        for entry in analysis:
            entry["buyPredictedCount"] = int((predicted_series == "BUY").sum())
            entry["sellPredictedCount"] = int((predicted_series == "SELL").sum())
        return analysis

    def _run_single_experiment(
        self,
        *,
        symbol_token: str,
        config: ExperimentConfig,
        split_mode: str = "PER_DATASET_RATIO",
    ) -> dict[str, Any]:
        dataset_summary = self.generator.generate_dataset(
            symbol_token=symbol_token,
            timeframe=config.timeframe,
            prediction_horizon_bars=config.prediction_horizon_bars,
            buy_threshold_pct=config.buy_threshold_pct,
            sell_threshold_pct=config.sell_threshold_pct,
            trading_style=config.trading_style,
        )
        rows = self.generator.supervised_rows
        if rows is None or rows.empty:
            raise ValueError(f"No rows generated for experiment {config.experiment_id}")

        split_config = DatasetSplitConfig(
            trainRatio=0.70,
            validationRatio=0.15,
            testRatio=0.15,
            purgeEnabled=True,
            embargoBars=0,
        )
        split_result = self.splitter.split(
            rows,
            split_config,
            dataset_fingerprint=dataset_summary.datasetFingerprint or "",
            prediction_horizon_bars=config.prediction_horizon_bars,
            trading_style=config.trading_style,
        )
        train_df, validation_df, test_df = self.splitter.split_frames(
            rows,
            split_config,
            dataset_fingerprint=dataset_summary.datasetFingerprint or "",
            prediction_horizon_bars=config.prediction_horizon_bars,
            trading_style=config.trading_style,
        )

        if train_df.empty or validation_df.empty:
            raise ValueError(f"Experiment {config.experiment_id} produced empty train/validation data")

        feature_columns = list(FEATURE_SCHEMA.columns)
        trainer = self.trainer_factory(model_path=f"/tmp/{config.experiment_id}_model.joblib")
        search_result = trainer.search(
            train_df=train_df,
            validation_df=validation_df,
            feature_columns=feature_columns,
            label_column="label",
            candidate_count=12,
        )
        best_validation_metrics = search_result["bestValidationMetrics"]
        best_hyperparameters = search_result["bestHyperparameters"]

        probabilities = None
        model = trainer._fit_candidate(
            X_train=trainer.validate_feature_matrix(train_df, feature_columns),
            X_validation=trainer.validate_feature_matrix(validation_df, feature_columns),
            y_train=train_df["label"].astype(str),
            y_validation=validation_df["label"].astype(str),
            hyperparameters={**best_hyperparameters, "n_estimators": 200, "early_stopping_rounds": 20},
            label_order=self._canonical_label_order(),
        )["probabilities"]
        probabilities = model

        validation_labels = validation_df["label"].astype(str)
        majority_baseline = ModelEvaluator.majority_class_baseline(validation_labels)
        stratified_baseline = ModelEvaluator.stratified_baseline(validation_labels, random_state=42)
        confidence_analysis = self._confidence_analysis_for_validation(
            validation_labels,
            probabilities,
            labels=self._canonical_label_order(),
        )
        for entry in confidence_analysis:
            predicted = pd.Series([self._canonical_label_order()[int(np.argmax(row))] for row in probabilities], dtype=object)
            entry["buyPredictedCount"] = int((predicted == "BUY").sum())
            entry["sellPredictedCount"] = int((predicted == "SELL").sum())

        feature_importance = []
        best_model = trainer._fit_candidate(
            X_train=trainer.validate_feature_matrix(train_df, feature_columns),
            X_validation=trainer.validate_feature_matrix(validation_df, feature_columns),
            y_train=train_df["label"].astype(str),
            y_validation=validation_df["label"].astype(str),
            hyperparameters={**best_hyperparameters, "n_estimators": 200, "early_stopping_rounds": 20},
            label_order=self._canonical_label_order(),
        )["trainedModel"]
        if hasattr(best_model, "feature_importances_"):
            feature_importance = ModelEvaluator.feature_importance_summary(best_model, feature_columns)

        result = {
            "experimentId": config.experiment_id,
            "predictionHorizonBars": config.prediction_horizon_bars,
            "buyThresholdPct": config.buy_threshold_pct,
            "sellThresholdPct": config.sell_threshold_pct,
            "datasetFingerprint": dataset_summary.datasetFingerprint,
            "splitFingerprint": split_result.splitFingerprint,
            "datasetRowCount": int(dataset_summary.datasetRowCount),
            "trainRows": int(len(train_df)),
            "validationRows": int(len(validation_df)),
            "testRows": int(len(test_df)),
            "validationPeriod": self._validation_period(validation_df),
            "labelDistribution": self._safe_label_distribution(validation_df),
            "majorityBaseline": majority_baseline,
            "stratifiedBaseline": stratified_baseline,
            "bestHyperparameters": best_hyperparameters,
            "validationMetrics": best_validation_metrics,
            "classMetrics": best_validation_metrics.get("classMetrics", {}),
            "confidenceAnalysis": confidence_analysis,
            "featureImportance": feature_importance,
            "testEvaluated": False,
            "testLocked": True,
            "splitMode": split_mode,
            "featureCount": len(feature_columns),
            "featureSchema": list(feature_columns),
        }
        return result

    def run_all(
        self,
        *,
        symbol_token: str,
        mode: str = "COMMON_CALENDAR_WINDOW",
    ) -> dict[str, Any]:
        experiments = [
            ExperimentConfig(
                experiment_id=item["experimentId"],
                prediction_horizon_bars=item["predictionHorizonBars"],
                buy_threshold_pct=item["buyThresholdPct"],
                sell_threshold_pct=item["sellThresholdPct"],
                timeframe="ONE_MINUTE",
                trading_style=TradingStyle.INTRADAY,
            )
            for item in DEFAULT_EXPERIMENTS
        ]
        results = [self._run_single_experiment(symbol_token=symbol_token, config=config, split_mode=mode) for config in experiments]
        leaderboard = self._finalize_leaderboard(results)
        selection = self.select_best_experiment(results)
        return {
            "results": results,
            "leaderboard": leaderboard,
            "selected": selection,
        }

    @staticmethod
    def _finalize_leaderboard(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(
            results,
            key=lambda item: (
                float(item["validationMetrics"].get("macroF1", 0.0)),
                float(item["validationMetrics"].get("balancedAccuracy", 0.0)),
                float(item["validationMetrics"].get("accuracy", 0.0)),
            ),
            reverse=True,
        )
        for idx, item in enumerate(ranked, start=1):
            item["rank"] = idx
        return ranked

    @staticmethod
    def select_best_experiment(results: list[dict[str, Any]]) -> dict[str, Any]:
        ranked = sorted(
            results,
            key=lambda item: float(item["validationMetrics"].get("macroF1", 0.0)),
            reverse=True,
        )
        best = ranked[0]
        if best["validationMetrics"].get("macroF1", 0.0) >= 0.35:
            quality_status = "STRONG_CANDIDATE"
        elif best["validationMetrics"].get("macroF1", 0.0) >= 0.25:
            quality_status = "WEAK_CANDIDATE"
        else:
            quality_status = "LOW_PREDICTIVE_SIGNAL"
        return {
            "selectedExperimentId": best["experimentId"],
            "selectedHorizon": best["predictionHorizonBars"],
            "selectedThreshold": {
                "buyThresholdPct": best["buyThresholdPct"],
                "sellThresholdPct": best["sellThresholdPct"],
            },
            "selectedHyperparameters": best["bestHyperparameters"],
            "selectionMetric": "macroF1",
            "selectionMetricValue": float(best["validationMetrics"].get("macroF1", 0.0)),
            "qualityStatus": quality_status,
            "experiment": best,
        }
