from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from app.features.feature_schema import FEATURE_SCHEMA

CANONICAL_CLASS_ORDER = ["SELL", "HOLD", "BUY"]


class ModelEvaluator:
    """Evaluation utilities for validation-only model comparison and baseline reporting."""

    @staticmethod
    def canonical_label_order(labels: pd.Series | list[str] | np.ndarray | None = None) -> list[str]:
        if labels is None:
            return list(CANONICAL_CLASS_ORDER)
        ordered = []
        for label in CANONICAL_CLASS_ORDER:
            if label in pd.Series(labels).astype(str).unique():
                ordered.append(label)
        extras = [
            str(label)
            for label in pd.Series(labels).astype(str).unique()
            if str(label) not in CANONICAL_CLASS_ORDER
        ]
        return ordered + extras

    @staticmethod
    def majority_class_baseline(labels: pd.Series | list[str] | np.ndarray) -> dict[str, Any]:
        values = pd.Series(labels).astype(str)
        counts = values.value_counts(dropna=False)
        winner = str(counts.index[0])
        preds = pd.Series([winner] * len(values), index=values.index)
        label_order = ModelEvaluator.canonical_label_order(values)
        metrics = ModelEvaluator.evaluate_predictions(values, preds, labels=label_order)
        return {
            "baseline_type": "majority",
            "prediction": winner,
            "accuracy": float(metrics["accuracy"]),
            "balancedAccuracy": float(metrics["balancedAccuracy"]),
            "macroPrecision": float(metrics["macroPrecision"]),
            "macroRecall": float(metrics["macroRecall"]),
            "macroF1": float(metrics["macroF1"]),
            "weightedF1": float(metrics["weightedF1"]),
            "confusionMatrix": {
                "classOrder": list(label_order),
                "matrix": metrics["confusionMatrix"]["matrix"],
            },
            "classCounts": {str(key): int(value) for key, value in counts.items()},
            "n_samples": int(len(values)),
        }

    @staticmethod
    def stratified_baseline(
        labels: pd.Series | list[str] | np.ndarray,
        random_state: int = 42,
    ) -> dict[str, Any]:
        values = pd.Series(labels).astype(str)
        X = np.arange(len(values), dtype=float).reshape(-1, 1)
        clf = DummyClassifier(strategy="stratified", random_state=random_state)
        clf.fit(X, values)
        predictions = pd.Series(clf.predict(X)).astype(str)
        label_order = ModelEvaluator.canonical_label_order(values)
        metrics = ModelEvaluator.evaluate_predictions(values, predictions, labels=label_order)
        counts = values.value_counts(dropna=False)
        return {
            "baseline_type": "stratified",
            "accuracy": float(metrics["accuracy"]),
            "balancedAccuracy": float(metrics["balancedAccuracy"]),
            "macroPrecision": float(metrics["macroPrecision"]),
            "macroRecall": float(metrics["macroRecall"]),
            "macroF1": float(metrics["macroF1"]),
            "weightedF1": float(metrics["weightedF1"]),
            "confusionMatrix": {
                "classOrder": list(label_order),
                "matrix": metrics["confusionMatrix"]["matrix"],
            },
            "classCounts": {str(key): int(value) for key, value in counts.items()},
            "n_samples": int(len(values)),
        }

    @staticmethod
    def evaluate_predictions(
        y_true: pd.Series | list[str] | np.ndarray,
        y_pred: pd.Series | list[str] | np.ndarray,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        truth = pd.Series(y_true).astype(str)
        preds = pd.Series(y_pred).astype(str)
        label_order = list(labels) if labels is not None else ModelEvaluator.canonical_label_order(truth)
        if not label_order:
            label_order = list(CANONICAL_CLASS_ORDER)
        cm = confusion_matrix(truth, preds, labels=label_order)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth,
            preds,
            labels=label_order,
            average=None,
            zero_division=0,
        )
        present_class_mask = support > 0
        balanced_accuracy = float(np.mean(recall[present_class_mask])) if present_class_mask.any() else 0.0
        metrics = {
            "accuracy": float(accuracy_score(truth, preds)),
            "balancedAccuracy": balanced_accuracy,
            "macroPrecision": float(np.mean(precision)),
            "macroRecall": float(np.mean(recall)),
            "macroF1": float(f1_score(truth, preds, labels=label_order, average="macro", zero_division=0)),
            "weightedF1": float(f1_score(truth, preds, labels=label_order, average="weighted", zero_division=0)),
            "confusionMatrix": {"classOrder": list(label_order), "matrix": cm.tolist()},
            "classMetrics": {},
        }
        for idx, label in enumerate(label_order):
            metrics["classMetrics"][label] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        return metrics

    @staticmethod
    def log_loss_metrics(
        y_true: pd.Series | list[str] | np.ndarray,
        probabilities: np.ndarray,
        labels: list[str] | None = None,
    ) -> float:
        truth = pd.Series(y_true).astype(str)
        label_order = list(labels or CANONICAL_CLASS_ORDER)
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape != (len(truth), len(label_order)):
            raise ValueError("Predict-proba probabilities must align with rows and the supplied label order")
        if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
            raise ValueError("Predict-proba probabilities contain invalid values")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Predict-proba probability rows must sum to one")
        label_indices = {label: index for index, label in enumerate(label_order)}
        unknown = sorted(set(truth.unique()) - set(label_indices))
        if unknown:
            raise ValueError(f"Log-loss labels are not represented in the probability axis: {unknown}")
        encoded_truth = truth.map(label_indices).to_numpy(dtype=int)
        return float(log_loss(encoded_truth, probabilities, labels=list(range(len(label_order)))))

    @staticmethod
    def confidence_analysis(
        y_true: pd.Series | list[str] | np.ndarray,
        probabilities: np.ndarray,
        labels: list[str] | None = None,
        thresholds: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        truth = pd.Series(y_true).astype(str)
        label_order = list(labels or CANONICAL_CLASS_ORDER)
        thresholds = thresholds or [0.4, 0.5, 0.6, 0.7, 0.8]
        if probabilities.ndim == 1:
            raise ValueError("Predict-proba probabilities must be 2D")
        confidence = np.max(probabilities, axis=1)
        predictions = np.asarray([label_order[np.argmax(probs)] for probs in probabilities], dtype=object)
        analyses: list[dict[str, Any]] = []
        for threshold in thresholds:
            mask = confidence >= threshold
            sample_count = int(mask.sum())
            if sample_count == 0:
                analyses.append({
                    "threshold": threshold,
                    "sampleCount": 0,
                    "coveragePct": 0.0,
                    "accuracy": 0.0,
                    "buyPrecision": 0.0,
                    "buyRecall": 0.0,
                    "sellPrecision": 0.0,
                    "sellRecall": 0.0,
                })
                continue
            selected_truth = truth.iloc[mask]
            selected_pred = pd.Series(predictions[mask]).astype(str)
            metrics = ModelEvaluator.evaluate_predictions(selected_truth, selected_pred, labels=label_order)
            analyses.append({
                "threshold": threshold,
                "sampleCount": sample_count,
                "coveragePct": float((sample_count / len(truth)) * 100.0),
                "accuracy": float(metrics["accuracy"]),
                "buyPrecision": float(metrics["classMetrics"].get("BUY", {}).get("precision", 0.0)),
                "buyRecall": float(metrics["classMetrics"].get("BUY", {}).get("recall", 0.0)),
                "sellPrecision": float(metrics["classMetrics"].get("SELL", {}).get("precision", 0.0)),
                "sellRecall": float(metrics["classMetrics"].get("SELL", {}).get("recall", 0.0)),
            })
        return analyses

    @staticmethod
    def promote_candidate_if_better(
        incumbent: dict[str, Any] | None,
        challenger: dict[str, Any] | None,
        *,
        minimum_promotion_delta: float = 0.0,
    ) -> dict[str, Any]:
        if challenger is None:
            return {
                "incumbent": incumbent,
                "challenger": None,
                "promotionDecision": "INCUMBENT_RETAINED",
                "promotionReason": "NO_CHALLENGER",
                "selectedModel": incumbent,
            }
        if incumbent is None:
            return {
                "incumbent": None,
                "challenger": challenger,
                "promotionDecision": "CHALLENGER_SELECTED",
                "promotionReason": "NO_INCUMBENT",
                "selectedModel": challenger,
            }

        incumbent_score = float(incumbent.get("validationMetrics", {}).get("macroF1", 0.0))
        challenger_score = float(challenger.get("validationMetrics", {}).get("macroF1", 0.0))
        threshold = incumbent_score + float(minimum_promotion_delta)
        if challenger_score >= threshold:
            return {
                "incumbent": incumbent,
                "challenger": challenger,
                "promotionDecision": "CHALLENGER_SELECTED",
                "promotionReason": (
                    f"challenger_macroF1={challenger_score:.12f} >= incumbent_macroF1={incumbent_score:.12f} + "
                    f"minimumPromotionDelta={minimum_promotion_delta:.12f}"
                ),
                "selectedModel": challenger,
            }
        return {
            "incumbent": incumbent,
            "challenger": challenger,
            "promotionDecision": "INCUMBENT_RETAINED",
            "promotionReason": (
                f"challenger_macroF1={challenger_score:.12f} < incumbent_macroF1={incumbent_score:.12f} + "
                f"minimumPromotionDelta={minimum_promotion_delta:.12f}"
            ),
            "selectedModel": incumbent,
        }

    @staticmethod
    def feature_importance_summary(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
        if not hasattr(model, "feature_importances_"):
            return []
        importances = np.asarray(model.feature_importances_, dtype=float)
        ranked = sorted(
            zip(feature_names, importances),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        return [
            {"feature": feature, "importance": float(score)}
            for feature, score in ranked
        ]

    @staticmethod
    def feature_importance_report(
        model: Any,
        feature_names: list[str],
        X_validation: pd.DataFrame,
        y_validation: pd.Series | list[str] | np.ndarray,
        *,
        permutation_repeats: int = 5,
        random_state: int = 42,
        minimum_importance_pct: float = 0.5,
    ) -> dict[str, Any]:
        """Rank features with booster gain/weight and validation permutation importance."""
        if len(feature_names) != len(FEATURE_SCHEMA.columns) or list(feature_names) != FEATURE_SCHEMA.columns:
            raise ValueError("Feature importance requires the canonical feature schema")
        if X_validation.empty or len(X_validation) != len(y_validation):
            raise ValueError("Feature importance requires aligned validation rows and labels")

        estimators = [getattr(model, "estimator", model)]
        if hasattr(model, "calibrated_classifiers_"):
            estimators = [
                getattr(item, "estimator", getattr(item, "base_estimator", None))
                for item in model.calibrated_classifiers_
            ]
            estimators = [getattr(item, "estimator", item) for item in estimators if item is not None]

        def expand(values: dict[str, float]) -> np.ndarray:
            scores = np.zeros(len(feature_names), dtype=float)
            for index, name in enumerate(feature_names):
                scores[index] = float(values.get(name, values.get(f"f{index}", 0.0)))
            return scores

        gain_arrays = []
        weight_arrays = []
        for estimator in estimators:
            booster = estimator.get_booster() if hasattr(estimator, "get_booster") else None
            if booster is not None:
                gain_arrays.append(expand(booster.get_score(importance_type="gain")))
                weight_arrays.append(expand(booster.get_score(importance_type="weight")))
            elif hasattr(estimator, "feature_importances_"):
                gain_arrays.append(np.asarray(estimator.feature_importances_, dtype=float))
                weight_arrays.append(np.zeros(len(feature_names), dtype=float))
        gain = np.mean(gain_arrays, axis=0) if gain_arrays else np.zeros(len(feature_names), dtype=float)
        weight = np.mean(weight_arrays, axis=0) if weight_arrays else np.zeros(len(feature_names), dtype=float)
        truth = pd.Series(y_validation).astype(str).reset_index(drop=True)
        validation = X_validation.loc[:, feature_names].reset_index(drop=True).copy()
        if len(validation) > 2000:
            sample = np.random.default_rng(random_state).choice(len(validation), size=2000, replace=False)
            validation = validation.iloc[sample].reset_index(drop=True)
            truth = truth.iloc[sample].reset_index(drop=True)
        random = np.random.default_rng(random_state)

        def decode_predictions(values: np.ndarray) -> list[str]:
            predictions = np.asarray(values)
            if np.issubdtype(predictions.dtype, np.number):
                if not np.equal(predictions, np.floor(predictions)).all():
                    raise ValueError("Classifier predictions are not discrete class identifiers")
                return [CANONICAL_CLASS_ORDER[int(value)] for value in predictions]
            return predictions.astype(str).tolist()

        baseline = ModelEvaluator.evaluate_predictions(
            truth,
            decode_predictions(model.predict(validation)),
            labels=CANONICAL_CLASS_ORDER,
        )["macroF1"]
        permutation = np.zeros(len(feature_names), dtype=float)
        for index, feature in enumerate(feature_names):
            score_drops = []
            for _ in range(max(1, permutation_repeats)):
                permuted = validation.copy()
                permuted[feature] = random.permutation(permuted[feature].to_numpy())
                predictions = decode_predictions(model.predict(permuted))
                permuted_score = ModelEvaluator.evaluate_predictions(
                    truth,
                    predictions,
                    labels=CANONICAL_CLASS_ORDER,
                )["macroF1"]
                score_drops.append(max(0.0, float(baseline) - float(permuted_score)))
            permutation[index] = float(np.mean(score_drops))

        def percentages(scores: np.ndarray) -> np.ndarray:
            total = float(scores.sum())
            return scores / total * 100.0 if total > 0.0 else np.zeros_like(scores)

        gain_pct = percentages(np.maximum(gain, 0.0))
        weight_pct = percentages(np.maximum(weight, 0.0))
        permutation_pct = percentages(np.maximum(permutation, 0.0))
        combined_pct = (gain_pct + weight_pct + permutation_pct) / 3.0
        ranking = [
            {
                "feature": feature,
                "gain": float(gain[index]),
                "gainImportancePct": float(gain_pct[index]),
                "weight": float(weight[index]),
                "weightImportancePct": float(weight_pct[index]),
                "permutationImportance": float(permutation[index]),
                "permutationImportancePct": float(permutation_pct[index]),
                "combinedImportancePct": float(combined_pct[index]),
            }
            for index, feature in enumerate(feature_names)
        ]
        ranking.sort(key=lambda item: item["combinedImportancePct"], reverse=True)
        return {
            "ranking": ranking,
            "top10": ranking[:10],
            "bottom10": ranking[-10:],
            "belowMinimumImportance": [
                item["feature"]
                for item in ranking
                if item["combinedImportancePct"] < minimum_importance_pct
            ],
            "minimumImportancePct": float(minimum_importance_pct),
            "permutationRepeats": max(1, permutation_repeats),
            "permutationSampleCount": int(len(validation)),
            "scoringMetric": "macroF1",
            "validationOnly": True,
        }

    @staticmethod
    def has_tree_importance(model: Any) -> bool:
        def supports_importance(estimator: Any) -> bool:
            estimator = getattr(estimator, "estimator", estimator)
            return hasattr(estimator, "get_booster") or hasattr(estimator, "feature_importances_")

        if hasattr(model, "calibrated_classifiers_"):
            return any(
                supports_importance(getattr(item, "estimator", getattr(item, "base_estimator", item)))
                for item in model.calibrated_classifiers_
            )
        return supports_importance(model)

    @staticmethod
    def compare_with_baselines(
        y_true: pd.Series | list[str] | np.ndarray,
        y_pred: pd.Series | list[str] | np.ndarray,
    ) -> dict[str, Any]:
        truth = pd.Series(y_true).astype(str)
        predictions = pd.Series(y_pred).astype(str)
        metrics = ModelEvaluator.evaluate_predictions(truth, predictions)
        majority = ModelEvaluator.majority_class_baseline(truth)
        stratified = ModelEvaluator.stratified_baseline(truth)
        return {
            "accuracy": metrics["accuracy"],
            "macroF1": metrics["macroF1"],
            "majorityBaselineAccuracy": majority["accuracy"],
            "stratifiedBaselineAccuracy": stratified["accuracy"],
            "accuracyLiftVsMajority": metrics["accuracy"] - majority["accuracy"],
            "macroF1LiftVsMajority": metrics["macroF1"] - majority["macroF1"],
        }
