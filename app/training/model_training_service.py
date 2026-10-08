from __future__ import annotations

import json
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
from app.model.model_registry import artifact_dir_for, model_id_for, model_path_for
from app.schemas.model_schema import ValidationStrategy, WalkForwardConfig
from app.training.lightgbm_trainer import LightGBMTrainer
from app.training.model_evaluator import ModelEvaluator
from app.training.walk_forward import WalkForwardConfig as WalkForwardSpec, WalkForwardTrainer
from app.training.xgboost_trainer import XGBoostTrainer


class ModelTrainingService:
    """Generate real rows, chronologically split them, run validation-only model selection, and persist metadata."""

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

    @staticmethod
    def _feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        X = df.loc[:, feature_columns].copy()
        X = X.replace([float("inf"), float("-inf")], 0.0)
        X = X.fillna(0.0)
        return X.astype(float)

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
        validation_strategy: ValidationStrategy | str = ValidationStrategy.HOLDOUT,
        walk_forward: WalkForwardConfig | WalkForwardSpec | None = None,
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

        rows = self.generator.supervised_rows
        if rows is None or rows.empty:
            raise ValueError("Dataset generation did not produce training rows")

        normalized_strategy = ValidationStrategy(validation_strategy) if not isinstance(validation_strategy, ValidationStrategy) else validation_strategy
        feature_columns = list(FEATURE_SCHEMA.columns)

        if normalized_strategy == ValidationStrategy.WALK_FORWARD:
            config = walk_forward or WalkForwardConfig(
                foldCount=4,
                validationWindowSessions=60,
                expandingWindow=True,
                sessionAligned=True,
                purgeEnabled=purge_enabled,
                embargoBars=embargo_bars,
            )
            if isinstance(config, WalkForwardConfig):
                wf_config = config
            else:
                wf_config = WalkForwardConfig(
                    foldCount=getattr(config, "foldCount", 4),
                    validationWindowSessions=getattr(config, "validationWindowSessions", 60),
                    expandingWindow=getattr(config, "expandingWindow", True),
                    sessionAligned=getattr(config, "sessionAligned", True),
                    purgeEnabled=getattr(config, "purgeEnabled", purge_enabled),
                    embargoBars=getattr(config, "embargoBars", embargo_bars),
                    minimumTrainingSessions=getattr(config, "minimumTrainingSessions", None),
                )
            if "trading_date" not in rows.columns:
                raise ValueError("Walk-forward validation requires trading_date metadata")
            split_config = DatasetSplitConfig(
                trainRatio=train_ratio,
                validationRatio=validation_ratio,
                testRatio=test_ratio,
                purgeEnabled=purge_enabled,
                embargoBars=embargo_bars,
            )
            trainer = WalkForwardTrainer()
            candidate = {
                "max_depth": 4,
                "learning_rate": 0.03,
                "min_child_weight": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "gamma": 0.05,
                "reg_alpha": 0.01,
                "reg_lambda": 2.0,
                "n_estimators": 200,
                "early_stopping_rounds": 20,
            }
            evaluation = trainer.evaluate_candidate(
                rows=rows,
                config=wf_config,
                split_config=split_config,
                dataset_fingerprint=dataset_summary.datasetFingerprint or "",
                prediction_horizon_bars=prediction_horizon_bars,
                trading_style=trading_style,
                feature_columns=feature_columns,
                trainer_factory=XGBoostTrainer,
                candidate_params=candidate,
            )
            normalized_style = TradingStyle.normalize(trading_style)
            walk_forward_model_id = (
                f"{symbol_token}_{normalized_style.value}_{timeframe.upper()}_"
                f"{prediction_horizon_bars}_XGBOOST_{FEATURE_SCHEMA.version}_"
                f"{evaluation['walkForwardFingerprint'][:16]}"
            )
            return {
                "status": "WALK_FORWARD_VALIDATED",
                "validationStrategy": "WALK_FORWARD",
                "testEvaluated": False,
                "testLocked": True,
                "modelId": walk_forward_model_id,
                "selectedHyperparameters": candidate,
                "walkForward": {
                    "foldCount": evaluation["foldCount"],
                    "expandingWindow": wf_config.expandingWindow,
                    "validationWindowSessions": wf_config.validationWindowSessions,
                    "folds": evaluation["folds"],
                    "aggregate": evaluation["aggregate"],
                    "lockedTest": evaluation["lockedTest"],
                    "developmentData": evaluation["developmentData"],
                    "walkForwardCoverage": evaluation["walkForwardCoverage"],
                    "walkForwardFingerprint": evaluation["walkForwardFingerprint"],
                    "excludedBoundaryRows": evaluation["excludedBoundaryRows"],
                    "rowReconciliationValid": evaluation["rowReconciliationValid"],
                },
                "dataset": dataset_summary.model_dump(mode="json"),
                "feature_columns": feature_columns,
                "testLocked": True,
                "testEvaluated": False,
                "lifecycleStatus": "WALK_FORWARD_VALIDATED",
            }

        split_config = DatasetSplitConfig(
            trainRatio=train_ratio,
            validationRatio=validation_ratio,
            testRatio=test_ratio,
            purgeEnabled=purge_enabled,
            embargoBars=embargo_bars,
        )
        split_result = self.splitter.split(
            rows,
            split_config,
            dataset_fingerprint=dataset_summary.datasetFingerprint or "",
            prediction_horizon_bars=prediction_horizon_bars,
            trading_style=trading_style,
        )
        train_df, validation_df, test_df = self.splitter.split_frames(
            rows,
            split_config,
            dataset_fingerprint=dataset_summary.datasetFingerprint or "",
            prediction_horizon_bars=prediction_horizon_bars,
            trading_style=trading_style,
        )
        if train_df.empty or validation_df.empty or test_df.empty:
            raise ValueError("Chronological split must contain train, validation, and test rows")

        X_train = self._feature_matrix(train_df, feature_columns)
        X_validation = self._feature_matrix(validation_df, feature_columns)
        if X_train.shape != (len(train_df), 26):
            raise ValueError(f"X_train shape mismatch: expected {(len(train_df), 26)}, received {X_train.shape}")
        if X_validation.shape != (len(validation_df), 26):
            raise ValueError(f"X_validation shape mismatch: expected {(len(validation_df), 26)}, received {X_validation.shape}")

        majority_baseline = ModelEvaluator.majority_class_baseline(validation_df["label"].astype(str))
        stratified_baseline = ModelEvaluator.stratified_baseline(validation_df["label"].astype(str), random_state=42)

        normalized_style = TradingStyle.normalize(trading_style)
        model_id = model_id_for(
            symbol_token=symbol_token,
            trading_style=normalized_style,
            timeframe=timeframe,
            prediction_horizon_bars=prediction_horizon_bars,
            algorithm="XGBOOST",
            feature_version=FEATURE_SCHEMA.version,
            dataset_fingerprint=dataset_summary.datasetFingerprint or "",
            feature_count=26,
        )
        artifact_path = artifact_dir_for(
            symbol_token=symbol_token,
            trading_style=normalized_style,
            timeframe=timeframe,
            prediction_horizon_bars=prediction_horizon_bars,
            algorithm="XGBOOST",
            feature_version=FEATURE_SCHEMA.version,
            dataset_fingerprint=dataset_summary.datasetFingerprint or "",
            feature_count=26,
            model_directory=self.model_directory,
        )
        xgb_model_path = artifact_path / "model.joblib"
        self.trainer = XGBoostTrainer(model_path=xgb_model_path, random_state=42)
        xgb_search = self.trainer.search(
            train_df=train_df,
            validation_df=validation_df,
            feature_columns=feature_columns,
            label_column="label",
            candidate_count=12,
        )
        best_xgb = xgb_search["bestValidationMetrics"]
        best_xgb_params = xgb_search["bestHyperparameters"]

        lightgbm_result = {"available": False, "status": "LIGHTGBM_UNAVAILABLE"}
        if LightGBMTrainer.available:
            lgb_path = artifact_path.with_name("lightgbm_model.joblib")
            lgb_trainer = LightGBMTrainer(model_path=lgb_path, random_state=42)
            lightgbm_result = lgb_trainer.train(
                train_df=train_df,
                validation_df=validation_df,
                feature_columns=feature_columns,
                label_column="label",
            )

        selected_candidate = {
            "algorithm": "XGBOOST",
            "hyperparameters": best_xgb_params,
            "validationMetrics": best_xgb,
        }
        if LightGBMTrainer.available and lightgbm_result.get("validation_metrics", {}).get("macroF1", 0.0) > best_xgb.get("macroF1", 0.0):
            selected_candidate = {
                "algorithm": "LIGHTGBM",
                "hyperparameters": lightgbm_result["hyperparameters"],
                "validationMetrics": lightgbm_result["validation_metrics"],
            }

        incumbent = None
        incumbent_metadata_path = self.trainer.model_path.with_suffix(".joblib.metadata.json")
        if incumbent_metadata_path.exists():
            try:
                incumbent = json.loads(incumbent_metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                incumbent = None

        incumbent_candidate = incumbent if incumbent else None
        if incumbent_candidate is not None:
            incumbent_candidate = {
                "model_path": incumbent_candidate.get("model_path", str(self.trainer.model_path)),
                "validationMetrics": incumbent_candidate.get("validation_metrics") or incumbent_candidate.get("validationMetrics", {}),
            }

        challenger_result = {
            "model_path": str(xgb_model_path),
            "validationMetrics": xgb_search["bestValidationMetrics"],
        }
        promotion = ModelEvaluator.promote_candidate_if_better(
            incumbent_candidate,
            challenger_result,
            minimum_promotion_delta=0.0,
        )

        selection_metric_value = float(promotion["selectedModel"]["validationMetrics"].get("macroF1", 0.0))
        lift_vs_majority = selection_metric_value - float(majority_baseline["macroF1"])
        return {
            "status": "VALIDATED",
            "testEvaluated": False,
            "testLocked": True,
            "modelId": model_id,
            "artifactPath": str(xgb_model_path),
            "metadataPath": str(xgb_model_path.with_suffix(".joblib.metadata.json")),
            "baseline": {
                "majority": majority_baseline,
                "stratified": stratified_baseline,
            },
            "search": {
                "xgboostCandidatesEvaluated": xgb_search["candidatesEvaluated"],
                "lightgbmCandidatesEvaluated": 1 if LightGBMTrainer.available else 0,
            },
            "bestXgboost": {
                "hyperparameters": best_xgb_params,
                "validationMetrics": best_xgb,
            },
            "bestLightgbm": lightgbm_result,
            "incumbent": promotion["incumbent"],
            "challenger": promotion["challenger"],
            "promotionDecision": promotion["promotionDecision"],
            "promotionReason": promotion["promotionReason"],
            "selectedModel": {
                "modelId": model_id,
                "algorithm": "XGBOOST",
                "selectionMetric": "macroF1",
                "selectionMetricValue": selection_metric_value,
                "validationMetrics": promotion["selectedModel"]["validationMetrics"],
            },
            "lift": {
                "accuracyVsMajority": float(promotion["selectedModel"]["validationMetrics"].get("accuracy", 0.0)) - float(majority_baseline["accuracy"]),
                "macroF1VsMajority": lift_vs_majority,
            },
            "dataset": dataset_summary.model_dump(mode="json"),
            "split": split_result.model_dump(mode="json"),
            "feature_columns": feature_columns,
            "training": {
                "model_path": str(xgb_model_path),
                "xgboostSearch": xgb_search,
                "labelEncoding": {"canonicalClassOrder": ["SELL", "HOLD", "BUY"]},
                "canonicalClassOrder": ["SELL", "HOLD", "BUY"],
            },
        }
