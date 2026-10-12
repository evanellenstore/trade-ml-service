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
from app.training.oof_analysis import OOFAnalyzer
from app.training.probability_calibration import ProbabilityCalibrationEvaluator
from app.training.xgboost_trainer import XGBoostTrainer
from app.training.walk_forward import WalkForwardConfig, WalkForwardTrainer


DEFAULT_EXPERIMENTS = (
    {
        "experimentId": "A",
        "predictionHorizonBars": 20,
        "buyThresholdPct": 0.10,
        "sellThresholdPct": -0.10,
    },
    {
        "experimentId": "B",
        "predictionHorizonBars": 30,
        "buyThresholdPct": 0.20,
        "sellThresholdPct": -0.20,
    },
    {
        "experimentId": "C",
        "predictionHorizonBars": 45,
        "buyThresholdPct": 0.25,
        "sellThresholdPct": -0.25,
    },
    {
        "experimentId": "D",
        "predictionHorizonBars": 60,
        "buyThresholdPct": 0.30,
        "sellThresholdPct": -0.30,
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
            thresholds=[0.40, 0.45, 0.50, 0.55, 0.60],
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

        final_fit = trainer._fit_candidate(
            X_train=trainer.validate_feature_matrix(train_df, feature_columns),
            X_validation=trainer.validate_feature_matrix(validation_df, feature_columns),
            y_train=train_df["label"].astype(str),
            y_validation=validation_df["label"].astype(str),
            hyperparameters={**best_hyperparameters, "n_estimators": 200, "early_stopping_rounds": 20},
            label_order=self._canonical_label_order(),
        )
        probabilities = final_fit["probabilities"]

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

        feature_importance = ModelEvaluator.feature_importance_report(
            final_fit["trainedModel"],
            feature_columns,
            trainer.validate_feature_matrix(validation_df, feature_columns),
            validation_labels,
        )
        predicted_labels = [self._canonical_label_order()[int(np.argmax(row))] for row in probabilities]
        oof_records = OOFAnalyzer.create_records(
            validation_rows=validation_df,
            actual_labels=validation_labels,
            predicted_labels=np.asarray(predicted_labels, dtype=object),
            probabilities=probabilities,
            fold_number=1,
        )
        oof_analysis = OOFAnalyzer.analyze(
            records=oof_records,
            development_rows=rows,
            trading_style=config.trading_style,
            locked_test_start=test_df["featureTimestamp"].min(),
            confidence_thresholds=[0.40, 0.45, 0.50, 0.55, 0.60],
            costs={
                "brokerage": 20.0,
                "transactionCostPct": 0.01,
                "slippagePct": 0.02,
                "sttPct": 0.025,
                "exchangeFeePct": 0.003,
            },
            minimum_trade_count=100,
            maximum_drawdown_pct=10.0,
            minimum_profit_factor=1.5,
            minimum_balanced_accuracy=0.45,
            minimum_macro_f1=0.45,
        )

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
            "signalQuality": oof_analysis["confidenceAnalysis"],
            "costAwareBacktests": oof_analysis["backtests"],
            "costAwareRecommendation": oof_analysis["finalCandidateSelection"],
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
        start_time: datetime | None = None,
        end_time: datetime | None = None,
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
        results = [
            self._run_walk_forward_experiment(
                symbol_token=symbol_token,
                config=config,
                start_time=start_time,
                end_time=end_time,
            )
            for config in experiments
        ]
        leaderboard = self._finalize_leaderboard(results)
        selection = self.select_best_experiment(results)
        selected = next(item for item in results if item["experimentId"] == selection["selectedExperimentId"])
        selected_config = next(item for item in experiments if item.experiment_id == selected["experimentId"])
        validation_candidate = bool(selected["costAwareRecommendation"].get("eligible"))
        if validation_candidate:
            locked_test = self._evaluate_locked_test(
                symbol_token=symbol_token,
                config=selected_config,
                hyperparameters=selected["bestHyperparameters"],
                validation_policy=selected["costAwareRecommendation"].get("selectedPolicy"),
                start_time=start_time,
                end_time=end_time,
            )
            selection["lockedTestEvaluation"] = locked_test
            test_candidate = bool(locked_test["productionGatesPassed"])
        else:
            locked_test = {
                "evaluated": False,
                "evaluationCount": 0,
                "reason": "NO_VALIDATION_CANDIDATE_PASSED_PRODUCTION_GATES",
            }
            selection["lockedTestEvaluation"] = locked_test
            test_candidate = False
        selection["productionEligible"] = validation_candidate and test_candidate
        selection["finalRecommendation"] = (
            "PRODUCTION_CANDIDATE" if selection["productionEligible"] else "REJECT: validation or locked-test gates failed"
        )
        return {
            "results": results,
            "leaderboard": leaderboard,
            "selected": selection,
        }

    def _evaluate_locked_test(
        self,
        *,
        symbol_token: str,
        config: ExperimentConfig,
        hyperparameters: dict[str, Any],
        validation_policy: dict[str, Any] | None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        dataset_summary = self.generator.generate_dataset(
            symbol_token=symbol_token,
            timeframe=config.timeframe,
            prediction_horizon_bars=config.prediction_horizon_bars,
            buy_threshold_pct=config.buy_threshold_pct,
            sell_threshold_pct=config.sell_threshold_pct,
            trading_style=config.trading_style,
            start_time=start_time,
            end_time=end_time,
        )
        rows = self.generator.supervised_rows
        if rows is None or rows.empty:
            raise ValueError("LOCKED_TEST_EMPTY: selected target generated no supervised rows")
        split = self.splitter.split_frames(
            rows,
            DatasetSplitConfig(
                trainRatio=0.70,
                validationRatio=0.15,
                testRatio=0.15,
                purgeEnabled=True,
                embargoBars=0,
            ),
            dataset_fingerprint=dataset_summary.datasetFingerprint or "",
            prediction_horizon_bars=config.prediction_horizon_bars,
            trading_style=config.trading_style,
        )
        train_rows, validation_rows, test_rows = split
        if train_rows.empty or validation_rows.empty or test_rows.empty:
            raise ValueError("LOCKED_TEST_EMPTY: chronological partitions must all contain rows")

        feature_columns = list(FEATURE_SCHEMA.columns)
        trainer = self.trainer_factory(model_path=f"/tmp/{config.experiment_id}_locked_test.joblib")
        fit = trainer._fit_candidate(
            X_train=trainer.validate_feature_matrix(train_rows, feature_columns),
            X_validation=trainer.validate_feature_matrix(validation_rows, feature_columns),
            y_train=train_rows["label"].astype(str),
            y_validation=validation_rows["label"].astype(str),
            hyperparameters={**hyperparameters, "n_estimators": 200, "early_stopping_rounds": 20},
            label_order=self._canonical_label_order(),
        )
        y_test = test_rows["label"].astype(str)
        calibrated_model, test_probabilities, calibration = ProbabilityCalibrationEvaluator().fit_select(
            fit["trainedModel"],
            trainer.validate_feature_matrix(train_rows, feature_columns),
            train_rows["label"].astype(str),
            trainer.validate_feature_matrix(test_rows, feature_columns),
            y_test,
            gap_bars=config.prediction_horizon_bars,
        )
        if calibrated_model is None or test_probabilities is None:
            test_probabilities = np.asarray(fit["trainedModel"].predict_proba(
                trainer.validate_feature_matrix(test_rows, feature_columns)
            ), dtype=float)
            calibration["lockedTestFallback"] = "UNCALIBRATED_BASE_MODEL"
        predicted = np.asarray(self._canonical_label_order(), dtype=object)[test_probabilities.argmax(axis=1)]
        metrics = ModelEvaluator.evaluate_predictions(y_test, predicted, labels=self._canonical_label_order())
        metrics["logLoss"] = ModelEvaluator.log_loss_metrics(
            y_test,
            test_probabilities,
            labels=self._canonical_label_order(),
        )
        records = OOFAnalyzer.create_records(
            validation_rows=test_rows.reset_index(drop=True),
            actual_labels=y_test.reset_index(drop=True),
            predicted_labels=predicted,
            probabilities=test_probabilities,
            fold_number=1,
        )
        if validation_policy is None:
            validation_policy = {"policyType": "CONFIDENCE", "confidenceThreshold": 0.50}
        threshold = float(validation_policy.get("confidenceThreshold", 0.50))
        margin_threshold = validation_policy.get("marginThreshold")
        oof_rows = pd.DataFrame.from_records(records)
        selected_mask = oof_rows["confidence"] >= threshold
        if margin_threshold is not None:
            selected_mask &= oof_rows["margin"] >= float(margin_threshold)
        selected_records = oof_rows.loc[selected_mask].copy()
        execution_rows = OOFAnalyzer._attach_execution_prices(oof_rows, rows, config.trading_style)
        execution_selected = execution_rows.loc[execution_rows["candleId"].isin(selected_records["candleId"])]
        test_end = pd.to_datetime(test_rows["targetEndTimestamp"]).max() + pd.Timedelta(nanoseconds=1)
        costs = {
            "brokerage": 20.0,
            "transactionCostPct": 0.01,
            "slippagePct": 0.02,
            "sttPct": 0.025,
            "exchangeFeePct": 0.003,
        }
        backtest = OOFAnalyzer._backtest(
            execution_selected,
            policy={"policyType": validation_policy.get("policyType", "CONFIDENCE"), "confidenceThreshold": threshold, "marginThreshold": margin_threshold},
            costs=costs,
            capital_per_trade=100000.0,
            locked_test_start=pd.Timestamp(test_end),
        )
        production_gates = {
            "profitFactor": backtest["profitFactor"] is not None and backtest["profitFactor"] > 1.5,
            "tradeCount": backtest["tradeCount"] >= 100,
            "maxDrawdown": backtest["maxDrawdownPct"] < 10.0,
            "balancedAccuracy": metrics["balancedAccuracy"] > 0.45,
            "macroF1": metrics["macroF1"] > 0.45,
            "positiveExpectancyAfterCosts": backtest["expectancyPct"] is not None and backtest["expectancyPct"] > 0.0,
        }
        return {
            "evaluated": True,
            "evaluationCount": 1,
            "testRows": int(len(test_rows)),
            "startTime": pd.Timestamp(test_rows["featureTimestamp"].min()).isoformat(),
            "endTime": pd.Timestamp(test_rows["featureTimestamp"].max()).isoformat(),
            "metrics": metrics,
            "backtest": backtest,
            "calibration": calibration,
            "productionGates": production_gates,
            "productionGatesPassed": all(production_gates.values()),
            "locked": True,
            "tuningAfterEvaluation": False,
        }

    def _run_walk_forward_experiment(
        self,
        *,
        symbol_token: str,
        config: ExperimentConfig,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        dataset_summary = self.generator.generate_dataset(
            symbol_token=symbol_token,
            timeframe=config.timeframe,
            prediction_horizon_bars=config.prediction_horizon_bars,
            buy_threshold_pct=config.buy_threshold_pct,
            sell_threshold_pct=config.sell_threshold_pct,
            trading_style=config.trading_style,
            start_time=start_time,
            end_time=end_time,
        )
        rows = self.generator.supervised_rows
        if rows is None or rows.empty:
            raise ValueError(f"No rows generated for experiment {config.experiment_id}")

        feature_columns = list(FEATURE_SCHEMA.columns)
        validation_config = WalkForwardConfig(
            foldCount=4,
            validationWindowSessions=60,
            expandingWindow=True,
            sessionAligned=True,
            purgeEnabled=True,
            embargoBars=0,
        )
        split_config = DatasetSplitConfig(
            trainRatio=0.70,
            validationRatio=0.15,
            testRatio=0.15,
            purgeEnabled=True,
            embargoBars=0,
        )
        evaluation = WalkForwardTrainer().evaluate_candidate(
            rows=rows,
            config=validation_config,
            split_config=split_config,
            dataset_fingerprint=dataset_summary.datasetFingerprint or "",
            prediction_horizon_bars=config.prediction_horizon_bars,
            trading_style=config.trading_style,
            feature_columns=feature_columns,
            trainer_factory=self.trainer_factory,
            candidate_params={
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
            },
            oof_artifact_directory="/tmp/trade-ml-target-sweep-oof",
        )
        oof = evaluation["outOfFold"]
        aggregate = evaluation["aggregate"]
        return {
            "experimentId": config.experiment_id,
            "predictionHorizonBars": config.prediction_horizon_bars,
            "buyThresholdPct": config.buy_threshold_pct,
            "sellThresholdPct": config.sell_threshold_pct,
            "datasetFingerprint": dataset_summary.datasetFingerprint,
            "datasetRowCount": int(dataset_summary.datasetRowCount),
            "labelDistribution": self._safe_label_distribution(rows),
            "validationStrategy": "WALK_FORWARD",
            "folds": evaluation["folds"],
            "aggregate": aggregate,
            "validationMetrics": {
                "macroF1": aggregate["macroF1"]["mean"],
                "balancedAccuracy": aggregate["balancedAccuracy"]["mean"],
            },
            "bestHyperparameters": {
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
            },
            "featureImportance": aggregate.get("featureImportance"),
            "signalQuality": oof["confidenceAnalysis"],
            "costAwareBacktests": oof["backtests"],
            "costAwareRecommendation": oof["finalCandidateSelection"],
            "testEvaluated": False,
            "testLocked": True,
            "lockedTest": evaluation["lockedTest"],
            "walkForwardFingerprint": evaluation["walkForwardFingerprint"],
            "featureCount": len(feature_columns),
            "featureSchema": feature_columns,
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
            key=lambda item: (
                bool(item.get("costAwareRecommendation", {}).get("eligible", False)),
                float(item["validationMetrics"].get("macroF1", 0.0)),
                float(item["validationMetrics"].get("balancedAccuracy", 0.0)),
                float(item.get("costAwareRecommendation", {}).get("selectedPolicy", {}).get("netReturnPct") or 0.0),
            ),
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
            "productionEligible": bool(best.get("costAwareRecommendation", {}).get("eligible", False)),
            "selectedHorizon": best["predictionHorizonBars"],
            "selectedThreshold": {
                "buyThresholdPct": best["buyThresholdPct"],
                "sellThresholdPct": best["sellThresholdPct"],
            },
            "selectedHyperparameters": best["bestHyperparameters"],
            "selectionMetric": "macroF1",
            "selectionMetricValue": float(best["validationMetrics"].get("macroF1", 0.0)),
            "qualityStatus": "PRODUCTION_CANDIDATE" if best.get("costAwareRecommendation", {}).get("eligible") else quality_status,
            "experiment": best,
        }
