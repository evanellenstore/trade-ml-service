from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

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
        defined_order = [label for label in label_order if label in set(truth.unique()).union(set(preds.unique()))]
        if not defined_order:
            defined_order = list(CANONICAL_CLASS_ORDER)
        if len(defined_order) != len(label_order):
            label_order = defined_order
        cm = confusion_matrix(truth, preds, labels=label_order)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth,
            preds,
            labels=label_order,
            average=None,
            zero_division=0,
        )
        metrics = {
            "accuracy": float(accuracy_score(truth, preds)),
            "balancedAccuracy": float(balanced_accuracy_score(truth, preds)),
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
        label_order = labels or sorted(truth.unique())
        if probabilities.ndim == 1:
            raise ValueError("Predict-proba probabilities must be 2D")
        return float(log_loss(truth, probabilities, labels=label_order))

    @staticmethod
    def confidence_analysis(
        y_true: pd.Series | list[str] | np.ndarray,
        probabilities: np.ndarray,
        labels: list[str] | None = None,
        thresholds: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        truth = pd.Series(y_true).astype(str)
        label_order = labels or sorted(truth.unique())
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
