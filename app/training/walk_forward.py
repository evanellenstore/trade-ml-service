from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.domain.trading_style import TradingStyle
from app.dataset.split.dataset_split_config import DatasetSplitConfig
from app.dataset.split.purged_chronological_dataset_splitter import PurgedChronologicalDatasetSplitter
from app.features.feature_schema import FEATURE_SCHEMA
from app.training.model_evaluator import ModelEvaluator
from app.training.xgboost_trainer import XGBoostTrainer


CANONICAL_CLASS_ORDER = ["SELL", "HOLD", "BUY"]


@dataclass(frozen=True)
class WalkForwardConfig:
    foldCount: int = 4
    validationWindowSessions: int = 60
    expandingWindow: bool = True
    sessionAligned: bool = True
    purgeEnabled: bool = True
    embargoBars: int = 0
    minimumTrainingSessions: int | None = None


class WalkForwardSplitter:
    """Create expanding, non-overlapping session-aware folds from development data only."""

    @staticmethod
    def reserve_test_rows(
        rows: pd.DataFrame,
        *,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        development_rows, test_rows, _ = WalkForwardSplitter.reserve_test_partition(
            rows,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        return development_rows, test_rows

    @staticmethod
    def reserve_test_partition(
        rows: pd.DataFrame,
        *,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        purge_enabled: bool = True,
        embargo_bars: int = 0,
        trading_style: TradingStyle | str = TradingStyle.INTRADAY,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Use HOLDOUT's deterministic split to lock TEST before constructing folds."""
        if rows.empty:
            raise ValueError("Dataset is empty for walk-forward splitting")
        split_config = DatasetSplitConfig(
            trainRatio=train_ratio,
            validationRatio=validation_ratio,
            testRatio=test_ratio,
            purgeEnabled=purge_enabled,
            embargoBars=embargo_bars,
        )
        splitter = PurgedChronologicalDatasetSplitter()
        _, _, holdout_test = splitter.split_frames(
            rows,
            split_config,
            dataset_fingerprint="walk-forward-test-reservation",
            trading_style=trading_style,
        )
        if holdout_test.empty:
            raise ValueError("WALK_FORWARD_TEST_RESERVATION_EMPTY")

        prepared = splitter._prepare_rows(rows, trading_style)
        boundaries = splitter._determine_boundaries(prepared, split_config, trading_style)
        first_test_session = prepared.iloc[boundaries["validation_end"] + 1]["trading_date"]
        test_sessions = set(prepared.loc[prepared["trading_date"] >= first_test_session, "trading_date"])
        reserved_test_rows = prepared.loc[prepared["trading_date"].isin(test_sessions)].copy()
        test_ids = set(reserved_test_rows["candle_id"].astype(str))
        development_candidates = prepared.loc[~prepared["trading_date"].isin(test_sessions)].copy()
        test_start = pd.to_datetime(reserved_test_rows["featureTimestamp"]).min()
        boundary_overlap = development_candidates["targetEndTimestamp"] >= test_start
        excluded_rows = development_candidates.loc[boundary_overlap].copy()
        development_rows = development_candidates.loc[~boundary_overlap].copy()

        development_ids = set(development_rows["candle_id"].astype(str))
        input_ids = set(prepared["candle_id"].astype(str))
        excluded_ids = input_ids - development_ids - test_ids
        if development_ids & test_ids:
            raise ValueError("WALK_FORWARD_TEST_OVERLAP: reserved TEST intersects development data")
        if development_rows.empty or test_sessions.intersection(development_rows["trading_date"]):
            raise ValueError("WALK_FORWARD_TEST_OVERLAP: a TEST session is present in development data")
        if development_rows["featureTimestamp"].max() >= test_start:
            raise ValueError("WALK_FORWARD_TEST_OVERLAP: development timestamp reaches TEST")
        if development_rows["targetEndTimestamp"].max() >= test_start:
            raise ValueError("WALK_FORWARD_TEST_OVERLAP: development target interval reaches TEST")

        metadata = {
            "testRowIds": test_ids,
            "testSessions": test_sessions,
            "excludedRows": int(len(excluded_ids)),
            "lockedTest": {
                "rowCount": int(len(reserved_test_rows)),
                "sessionCount": int(reserved_test_rows["trading_date"].nunique()),
                "startTime": reserved_test_rows["featureTimestamp"].min().isoformat(),
                "endTime": reserved_test_rows["featureTimestamp"].max().isoformat(),
                "firstCandleId": str(reserved_test_rows.iloc[0]["candle_id"]),
                "lastCandleId": str(reserved_test_rows.iloc[-1]["candle_id"]),
                "evaluated": False,
                "locked": True,
            },
            "developmentData": {
                "rowCount": int(len(development_rows)),
                "sessionCount": int(development_rows["trading_date"].nunique()),
                "startTime": development_rows["featureTimestamp"].min().isoformat(),
                "endTime": development_rows["featureTimestamp"].max().isoformat(),
            },
            "testStartTimestamp": test_start,
            "inputRowCount": int(len(prepared)),
            "rowReconciliationValid": development_ids.isdisjoint(test_ids)
            and development_ids.isdisjoint(excluded_ids)
            and test_ids.isdisjoint(excluded_ids)
            and len(development_ids | test_ids | excluded_ids) == len(input_ids),
        }
        if not metadata["rowReconciliationValid"]:
            raise ValueError("WALK_FORWARD_TEST_RESERVATION_INVALID: input rows do not reconcile")
        return development_rows.reset_index(drop=True), reserved_test_rows.reset_index(drop=True), metadata

    @staticmethod
    def _compute_session_boundaries(rows: pd.DataFrame) -> list[pd.Timestamp]:
        if "trading_date" not in rows.columns or rows["trading_date"].isna().all():
            return []
        unique_sessions = sorted(pd.Series(rows["trading_date"]).dropna().unique())
        return [pd.Timestamp(session) for session in unique_sessions]

    @staticmethod
    def _session_index_map(rows: pd.DataFrame) -> dict[pd.Timestamp, int]:
        sessions = WalkForwardSplitter._compute_session_boundaries(rows)
        return {session: idx for idx, session in enumerate(sessions)}

    @staticmethod
    def _validate_fold(
        *,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        feature_columns: list[str],
        validation_window_sessions: int,
        config: WalkForwardConfig,
    ) -> dict[str, Any]:
        if train_df.empty or validation_df.empty:
            raise ValueError("Each fold must contain train and validation data")
        if train_df["featureTimestamp"].is_monotonic_increasing is False:
            raise ValueError("Train fold is not chronological")
        if validation_df["featureTimestamp"].is_monotonic_increasing is False:
            raise ValueError("Validation fold is not chronological")
        if len(set(train_df["candle_id"]).intersection(set(validation_df["candle_id"]))) > 0:
            raise ValueError("Validation row appears in training data for the same fold")
        if list(train_df.columns) and list(validation_df.columns):
            train_features = [column for column in feature_columns if column in train_df.columns]
            validation_features = [column for column in feature_columns if column in validation_df.columns]
            if train_features != feature_columns or validation_features != feature_columns:
                raise ValueError("Feature schema mismatch in walk-forward fold")
        if train_df["targetEndTimestamp"].max() >= validation_df["featureTimestamp"].min():
            raise ValueError("Train target overlap with validation window detected")
        if validation_window_sessions <= 0:
            raise ValueError("Validation window session count must be positive")
        if config.sessionAligned:
            if "trading_date" in train_df.columns and "trading_date" in validation_df.columns:
                if train_df["trading_date"].max() >= validation_df["trading_date"].min():
                    raise ValueError("Fold boundary is not session aligned")
        return {
            "chronologicalOrderValid": True,
            "noDuplicateRows": True,
            "trainValidationLabelOverlap": False,
            "featureSchemaValid": True,
            "rowReconciliationValid": True,
            "trainEndLessThanValidationStart": train_df["featureTimestamp"].max() < validation_df["featureTimestamp"].min(),
        }

    def create_folds(
        self,
        development_rows: pd.DataFrame,
        config: WalkForwardConfig,
    ) -> list[dict[str, Any]]:
        if development_rows.empty:
            raise ValueError("Development data is empty for walk-forward splitting")
        if "trading_date" not in development_rows.columns:
            raise ValueError("Development rows must include trading_date for session-aware walk-forward validation")

        session_order = sorted(pd.Series(development_rows["trading_date"]).dropna().unique())
        total_sessions = len(session_order)
        required_training_window = config.minimumTrainingSessions or config.validationWindowSessions
        required_sessions = required_training_window + (config.validationWindowSessions * config.foldCount)
        if total_sessions < required_sessions:
            raise ValueError("INSUFFICIENT_HISTORY_FOR_WALK_FORWARD")

        feature_columns = list(FEATURE_SCHEMA.columns)
        fold_summaries: list[dict[str, Any]] = []

        initial_training_sessions = total_sessions - (config.foldCount * config.validationWindowSessions)
        if initial_training_sessions < (config.minimumTrainingSessions or 1):
            raise ValueError("INSUFFICIENT_HISTORY_FOR_WALK_FORWARD")

        for fold_number in range(1, config.foldCount + 1):
            validation_start_session = initial_training_sessions + ((fold_number - 1) * config.validationWindowSessions)
            validation_end_session = validation_start_session + config.validationWindowSessions
            if validation_end_session > total_sessions:
                raise ValueError("INSUFFICIENT_HISTORY_FOR_WALK_FORWARD")

            train_mask = development_rows["trading_date"].isin(session_order[:validation_start_session])
            validation_mask = development_rows["trading_date"].isin(
                session_order[validation_start_session:validation_end_session]
            )
            train_df = development_rows.loc[train_mask].copy().reset_index(drop=True)
            validation_df = development_rows.loc[validation_mask].copy().reset_index(drop=True)

            if train_df.empty or validation_df.empty:
                raise ValueError(f"Fold {fold_number} produced empty partition")

            if config.purgeEnabled:
                validation_start_time = validation_df["featureTimestamp"].min()
                overlap_mask = train_df["targetEndTimestamp"] >= validation_start_time
                purged_rows = int(overlap_mask.sum())
                train_df = train_df.loc[~overlap_mask].copy().reset_index(drop=True)
                if train_df.empty:
                    raise ValueError(f"Fold {fold_number} train data becomes empty after purge")
            else:
                purged_rows = 0

            fold_checks = self._validate_fold(
                train_df=train_df,
                validation_df=validation_df,
                feature_columns=feature_columns,
                validation_window_sessions=config.validationWindowSessions,
                config=config,
            )

            fold_summaries.append(
                {
                    "foldNumber": fold_number,
                    "train": {
                        "startTime": train_df["featureTimestamp"].min().isoformat() if not train_df.empty else None,
                        "endTime": train_df["featureTimestamp"].max().isoformat() if not train_df.empty else None,
                        "sessionCount": int(train_df["trading_date"].nunique()),
                        "rows": int(len(train_df)),
                        "startSessionIndex": 0,
                        "endSessionIndex": int(train_df["trading_date"].nunique()) - 1,
                    },
                    "validation": {
                        "startTime": validation_df["featureTimestamp"].min().isoformat() if not validation_df.empty else None,
                        "endTime": validation_df["featureTimestamp"].max().isoformat() if not validation_df.empty else None,
                        "sessionCount": int(validation_df["trading_date"].nunique()),
                        "rows": int(len(validation_df)),
                        "startSessionIndex": validation_start_session,
                        "endSessionIndex": validation_end_session - 1,
                    },
                    "purge": {"purgedRows": purged_rows},
                    "embargo": {"embargoedRows": 0},
                    "checks": fold_checks,
                    "trainDataFrame": train_df,
                    "validationDataFrame": validation_df,
                }
            )
            current_train_end = validation_end_session

        if not fold_summaries:
            raise ValueError("INSUFFICIENT_HISTORY_FOR_WALK_FORWARD")
        if fold_summaries[-1]["validation"]["endSessionIndex"] != total_sessions - 1:
            raise ValueError("WALK_FORWARD_TEST_OVERLAP: final validation does not end at development boundary")
        return fold_summaries

    @staticmethod
    def _fold_fingerprint(
        *,
        dataset_fingerprint: str,
        locked_test: dict[str, Any],
        development_data: dict[str, Any],
        fold_count: int,
        validation_window_sessions: int,
        expanding_window: bool,
        session_aligned: bool,
        purge_enabled: bool,
        embargo_bars: int,
        fold_boundaries: list[dict[str, Any]],
        candidate_params: dict[str, Any],
        model_identity: dict[str, Any],
    ) -> str:
        payload = {
            "datasetFingerprint": dataset_fingerprint,
            "validationStrategy": "WALK_FORWARD",
            "lockedTestBoundary": {
                "startTime": locked_test["startTime"],
                "endTime": locked_test["endTime"],
                "firstCandleId": locked_test["firstCandleId"],
                "lastCandleId": locked_test["lastCandleId"],
            },
            "developmentBoundary": {
                "startTime": development_data["startTime"],
                "endTime": development_data["endTime"],
            },
            "foldCount": fold_count,
            "validationWindowSessions": validation_window_sessions,
            "expandingWindow": expanding_window,
            "sessionAligned": session_aligned,
            "purgeEnabled": purge_enabled,
            "embargoBars": embargo_bars,
            "foldBoundaries": fold_boundaries,
            "candidateParams": candidate_params,
            "modelIdentity": model_identity,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()


class WalkForwardEvaluator:
    """Aggregate validation metrics across folds and compare candidates."""

    @staticmethod
    def _metric_series(values: list[float]) -> dict[str, float]:
        values_array = np.asarray(values, dtype=float)
        if len(values_array) == 0:
            return {"mean": 0.0, "median": 0.0, "stdDev": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "stdDev": float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
        }

    @staticmethod
    def aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
        macro_f1_values = [float(fold["metrics"]["macroF1"]) for fold in folds if "metrics" in fold]
        balanced_accuracy_values = [float(fold["metrics"]["balancedAccuracy"]) for fold in folds if "metrics" in fold]
        log_loss_values = [float(fold["metrics"]["logLoss"]) for fold in folds if "metrics" in fold]
        return {
            "macroF1": WalkForwardEvaluator._metric_series(macro_f1_values),
            "balancedAccuracy": WalkForwardEvaluator._metric_series(balanced_accuracy_values),
            "logLoss": WalkForwardEvaluator._metric_series(log_loss_values),
            "BUY": {
                "meanPrecision": float(np.mean([float(fold["classMetrics"]["BUY"]["precision"]) for fold in folds])),
                "meanRecall": float(np.mean([float(fold["classMetrics"]["BUY"]["recall"]) for fold in folds])),
                "meanF1": float(np.mean([float(fold["classMetrics"]["BUY"]["f1"]) for fold in folds])),
            },
            "SELL": {
                "meanPrecision": float(np.mean([float(fold["classMetrics"]["SELL"]["precision"]) for fold in folds])),
                "meanRecall": float(np.mean([float(fold["classMetrics"]["SELL"]["recall"]) for fold in folds])),
                "meanF1": float(np.mean([float(fold["classMetrics"]["SELL"]["f1"]) for fold in folds])),
            },
            "HOLD": {
                "meanPrecision": float(np.mean([float(fold["classMetrics"]["HOLD"]["precision"]) for fold in folds])),
                "meanRecall": float(np.mean([float(fold["classMetrics"]["HOLD"]["recall"]) for fold in folds])),
                "meanF1": float(np.mean([float(fold["classMetrics"]["HOLD"]["f1"]) for fold in folds])),
            },
            "worstFold": 0,
        }


class WalkForwardTrainer:
    """Evaluate a candidate configuration across walk-forward folds and select the most stable model."""

    def evaluate_candidate(
        self,
        *,
        rows: pd.DataFrame,
        config: WalkForwardConfig,
        split_config: Any,
        dataset_fingerprint: str,
        prediction_horizon_bars: int,
        trading_style: TradingStyle | str,
        feature_columns: list[str],
        trainer_factory: Any,
        candidate_params: dict[str, Any],
    ) -> dict[str, Any]:
        splitter = WalkForwardSplitter()
        development_rows, test_rows, reservation = splitter.reserve_test_partition(
            rows,
            train_ratio=split_config.trainRatio,
            validation_ratio=split_config.validationRatio,
            test_ratio=split_config.testRatio,
            purge_enabled=split_config.purgeEnabled,
            embargo_bars=split_config.embargoBars,
            trading_style=trading_style,
        )
        fold_specs = splitter.create_folds(development_rows, config)
        fold_results: list[dict[str, Any]] = []
        test_ids = reservation["testRowIds"]
        test_sessions = reservation["testSessions"]
        test_start = reservation["testStartTimestamp"]
        fold_validation_ids: set[str] = set()
        validation_sessions_seen: set[Any] = set()
        initial_training_sessions = (
            development_rows["trading_date"].nunique()
            - config.foldCount * config.validationWindowSessions
        )
        if initial_training_sessions < (config.minimumTrainingSessions or 1):
            raise ValueError("INSUFFICIENT_HISTORY_FOR_WALK_FORWARD")

        development_session_order = sorted(set(development_rows["trading_date"]))
        prior_train_sessions: set[Any] = set()
        test_row_id_intersection_count = 0
        test_session_intersection_count = 0
        maximum_fold_target_end: pd.Timestamp | None = None
        for fold_index, fold in enumerate(fold_specs):
            train_frame = fold["trainDataFrame"]
            validation_frame = fold["validationDataFrame"]
            train_ids = set(train_frame["candle_id"].astype(str))
            validation_ids = set(validation_frame["candle_id"].astype(str))
            train_sessions = set(train_frame["trading_date"])
            validation_sessions = set(validation_frame["trading_date"])
            expected_validation_sessions = set(
                development_session_order[
                    initial_training_sessions + fold_index * config.validationWindowSessions:
                    initial_training_sessions + (fold_index + 1) * config.validationWindowSessions
                ]
            )
            row_intersections = (train_ids & test_ids) | (validation_ids & test_ids)
            session_intersections = (train_sessions & test_sessions) | (validation_sessions & test_sessions)
            test_row_id_intersection_count += len(row_intersections)
            test_session_intersection_count += len(session_intersections)
            if row_intersections:
                raise ValueError("WALK_FORWARD_TEST_OVERLAP: TEST row ID intersects a fold")
            if session_intersections:
                raise ValueError("WALK_FORWARD_TEST_OVERLAP: TEST session intersects a fold")
            if validation_sessions != expected_validation_sessions:
                raise ValueError("WALK_FORWARD_FOLD_INVALID: validation sessions do not match the anchored window")
            if validation_ids & fold_validation_ids:
                raise ValueError("WALK_FORWARD_FOLD_OVERLAP: validation row reused across folds")
            if not prior_train_sessions.issubset(train_sessions):
                raise ValueError("WALK_FORWARD_FOLD_INVALID: expanding training sessions regressed")
            if validation_frame["featureTimestamp"].max() >= test_start:
                raise ValueError("WALK_FORWARD_TEST_OVERLAP: validation timestamp reaches TEST")
            fold_target_end = pd.concat(
                [train_frame["targetEndTimestamp"], validation_frame["targetEndTimestamp"]],
                ignore_index=True,
            ).max()
            if maximum_fold_target_end is None or fold_target_end > maximum_fold_target_end:
                maximum_fold_target_end = pd.Timestamp(fold_target_end)
            if fold_target_end >= test_start:
                raise ValueError("WALK_FORWARD_TEST_OVERLAP: fold target interval reaches TEST")
            fold_validation_ids.update(validation_ids)
            validation_sessions_seen.update(validation_sessions)
            prior_train_sessions = train_sessions

        for fold in fold_specs:
            train_df = fold["trainDataFrame"].copy().reset_index(drop=True)
            validation_df = fold["validationDataFrame"].copy().reset_index(drop=True)
            X_train = XGBoostTrainer.validate_feature_matrix(train_df, feature_columns)
            X_validation = XGBoostTrainer.validate_feature_matrix(validation_df, feature_columns)
            y_train = train_df["label"].astype(str)
            y_validation = validation_df["label"].astype(str)
            trial_params = dict(candidate_params)
            trainer = trainer_factory(model_path=f"/tmp/walkforward_fold_{fold['foldNumber']}.joblib")
            fit_result = trainer._fit_candidate(
                X_train=X_train,
                X_validation=X_validation,
                y_train=y_train,
                y_validation=y_validation,
                hyperparameters=trial_params,
                label_order=["SELL", "HOLD", "BUY"],
            )
            probabilities = np.asarray(fit_result["probabilities"], dtype=float)
            validation_predictions = np.asarray(fit_result["trainedModel"].predict(X_validation), dtype=int)
            validation_predictions_labels = trainer._decode_canonical_labels(validation_predictions)
            if len(y_validation) != len(validation_predictions_labels):
                raise ValueError("Validation prediction count does not match validation rows")
            if probabilities.shape[1] != 3:
                raise ValueError("Probability rows do not contain exactly three class outputs")
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("Probability rows do not sum to approximately 1")
            labels = ["SELL", "HOLD", "BUY"]
            metrics = ModelEvaluator.evaluate_predictions(y_validation, validation_predictions_labels, labels=labels)
            metrics["logLoss"] = float(ModelEvaluator.log_loss_metrics(y_validation, probabilities, labels=labels))
            cm = np.asarray(metrics["confusionMatrix"]["matrix"], dtype=int)
            if cm.size == 0:
                raise ValueError("Confusion matrix is empty")
            if cm.sum() != len(validation_df):
                raise ValueError(f"Confusion matrix reconciliation failed: {cm.sum()} != {len(validation_df)}")
            if cm.sum() == 0 and len(validation_df) > 0:
                raise ValueError("All-zero confusion matrix while validation rows are present")
            if not np.allclose(cm.sum(axis=1), pd.Series(y_validation).value_counts().reindex(labels, fill_value=0).to_numpy(dtype=int)):
                raise ValueError("Confusion matrix row sums do not match true class support")
            if np.isnan(metrics["macroF1"]) or np.isinf(metrics["macroF1"]):
                raise ValueError("macroF1 is invalid")
            class_metrics = metrics["classMetrics"]
            fold_result = {
                "foldNumber": fold["foldNumber"],
                "train": {"startTime": train_df["featureTimestamp"].min().isoformat(), "endTime": train_df["featureTimestamp"].max().isoformat(), "sessions": int(train_df["trading_date"].nunique()), "rows": int(len(train_df))},
                "validation": {"startTime": validation_df["featureTimestamp"].min().isoformat(), "endTime": validation_df["featureTimestamp"].max().isoformat(), "sessions": int(validation_df["trading_date"].nunique()), "rows": int(len(validation_df))},
                "purgedRows": fold["purge"]["purgedRows"],
                "embargoedRows": 0,
                "labelDistribution": {"BUY": int((y_validation == "BUY").sum()), "HOLD": int((y_validation == "HOLD").sum()), "SELL": int((y_validation == "SELL").sum())},
                "metrics": {
                    "accuracy": float(metrics["accuracy"]),
                    "balancedAccuracy": float(metrics["balancedAccuracy"]),
                    "macroPrecision": float(metrics["macroPrecision"]),
                    "macroRecall": float(metrics["macroRecall"]),
                    "macroF1": float(metrics["macroF1"]),
                    "weightedF1": float(metrics["weightedF1"]),
                    "logLoss": float(metrics["logLoss"]),
                },
                "classMetrics": {
                    "SELL": {
                        "precision": float(class_metrics.get("SELL", {}).get("precision", 0.0)),
                        "recall": float(class_metrics.get("SELL", {}).get("recall", 0.0)),
                        "f1": float(class_metrics.get("SELL", {}).get("f1", 0.0)),
                        "support": int(class_metrics.get("SELL", {}).get("support", 0)),
                    },
                    "HOLD": {
                        "precision": float(class_metrics.get("HOLD", {}).get("precision", 0.0)),
                        "recall": float(class_metrics.get("HOLD", {}).get("recall", 0.0)),
                        "f1": float(class_metrics.get("HOLD", {}).get("f1", 0.0)),
                        "support": int(class_metrics.get("HOLD", {}).get("support", 0)),
                    },
                    "BUY": {
                        "precision": float(class_metrics.get("BUY", {}).get("precision", 0.0)),
                        "recall": float(class_metrics.get("BUY", {}).get("recall", 0.0)),
                        "f1": float(class_metrics.get("BUY", {}).get("f1", 0.0)),
                        "support": int(class_metrics.get("BUY", {}).get("support", 0)),
                    },
                },
                "confusionMatrix": {
                    "classOrder": CANONICAL_CLASS_ORDER,
                    "matrix": metrics["confusionMatrix"]["matrix"],
                },
                "testLocked": True,
                "testEvaluated": False,
                "modelPath": str(trainer.model_path),
            }
            fold_results.append(fold_result)

        aggregate = WalkForwardEvaluator.aggregate(fold_results)
        aggregate["worstFold"] = min(
            range(len(fold_results)),
            key=lambda index: float(fold_results[index]["metrics"]["macroF1"]),
            default=0,
        ) + 1
        if len(fold_results) != config.foldCount:
            raise ValueError("WALK_FORWARD_FOLD_INVALID: generated fold count differs from configuration")
        last_validation_time = max(
            pd.Timestamp(fold["validation"]["endTime"]) for fold in fold_results
        )
        if maximum_fold_target_end is None:
            raise ValueError("WALK_FORWARD_FOLD_INVALID: no fold target-end timestamp")
        if last_validation_time >= test_start or maximum_fold_target_end >= test_start:
            raise ValueError("WALK_FORWARD_TEST_OVERLAP: final fold reaches TEST")
        development_sessions = set(development_rows["trading_date"])
        unused_sessions = sorted(development_sessions - validation_sessions_seen - set(
            sorted(development_sessions)[:initial_training_sessions]
        ))
        if unused_sessions:
            raise ValueError("WALK_FORWARD_FOLD_INVALID: unused development sessions remain")
        fold_boundaries = [
            {
                "foldNumber": fold["foldNumber"],
                "trainStart": fold["train"]["startTime"],
                "trainEnd": fold["train"]["endTime"],
                "validationStart": fold["validation"]["startTime"],
                "validationEnd": fold["validation"]["endTime"],
            }
            for fold in fold_results
        ]
        model_identity = {
            "symbolToken": str(rows["symbol_token"].iloc[0]) if "symbol_token" in rows.columns else "",
            "tradingStyle": TradingStyle.normalize(trading_style).value,
            "timeframe": str(rows["timeframe"].iloc[0]) if "timeframe" in rows.columns else "",
            "predictionHorizonBars": int(prediction_horizon_bars),
            "featureVersion": FEATURE_SCHEMA.version,
            "featureCount": len(feature_columns),
        }
        fingerprint = splitter._fold_fingerprint(
            dataset_fingerprint=dataset_fingerprint,
            locked_test=reservation["lockedTest"],
            development_data=reservation["developmentData"],
            fold_count=config.foldCount,
            validation_window_sessions=config.validationWindowSessions,
            expanding_window=config.expandingWindow,
            session_aligned=config.sessionAligned,
            purge_enabled=config.purgeEnabled,
            embargo_bars=config.embargoBars,
            fold_boundaries=fold_boundaries,
            candidate_params=candidate_params,
            model_identity=model_identity,
        )
        coverage = {
            "initialTrainingSessions": int(initial_training_sessions),
            "validationSessionsPerFold": int(config.validationWindowSessions),
            "totalValidationSessions": int(config.foldCount * config.validationWindowSessions),
            "unusedDevelopmentSessions": int(len(unused_sessions)),
            "firstUnusedSession": str(unused_sessions[0]) if unused_sessions else None,
            "lastUnusedSession": str(unused_sessions[-1]) if unused_sessions else None,
            "lastValidationTime": last_validation_time.isoformat(),
            "maxWalkForwardValidationTimestamp": last_validation_time.isoformat(),
            "maxWalkForwardTargetEndTimestamp": maximum_fold_target_end.isoformat(),
            "testStartTime": reservation["lockedTest"]["startTime"],
            "testRowIdIntersectionCount": int(test_row_id_intersection_count),
            "testSessionIntersectionCount": int(test_session_intersection_count),
            "noWalkForwardTestOverlap": True,
        }
        return {
            "developmentRows": int(len(development_rows)),
            "testRows": int(len(test_rows)),
            "lockedTest": reservation["lockedTest"],
            "developmentData": reservation["developmentData"],
            "excludedBoundaryRows": reservation["excludedRows"],
            "rowReconciliationValid": reservation["rowReconciliationValid"],
            "walkForwardCoverage": coverage,
            "walkForwardFingerprint": fingerprint,
            "foldCount": len(fold_results),
            "folds": fold_results,
            "aggregate": aggregate,
            "testLocked": True,
            "testEvaluated": False,
        }
