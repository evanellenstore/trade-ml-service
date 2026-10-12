from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from app.training.model_evaluator import CANONICAL_CLASS_ORDER, ModelEvaluator


class ProbabilityCalibrationEvaluator:
    """Choose probability calibration using a chronological inner holdout only."""

    def __init__(self, *, n_splits: int = 3, calibration_fraction: float = 0.20) -> None:
        self.n_splits = n_splits
        self.calibration_fraction = calibration_fraction

    @staticmethod
    def _align_probabilities(estimator: Any, probabilities: np.ndarray) -> np.ndarray:
        aligned = np.zeros((len(probabilities), len(CANONICAL_CLASS_ORDER)), dtype=float)
        for column, class_value in enumerate(estimator.classes_):
            label = CANONICAL_CLASS_ORDER[int(class_value)] if isinstance(class_value, (int, np.integer)) else str(class_value)
            if label in CANONICAL_CLASS_ORDER:
                aligned[:, CANONICAL_CLASS_ORDER.index(label)] = probabilities[:, column]
        row_sums = aligned.sum(axis=1, keepdims=True)
        return np.divide(aligned, row_sums, out=np.zeros_like(aligned), where=row_sums > 0)

    @staticmethod
    def _metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
        truth = pd.Series(y_true).astype(str).reset_index(drop=True)
        encoded = truth.map({label: index for index, label in enumerate(CANONICAL_CLASS_ORDER)}).to_numpy(dtype=int)
        clipped = np.clip(probabilities, 1e-15, 1.0)
        one_hot = np.eye(len(CANONICAL_CLASS_ORDER))[encoded]
        confidence = clipped.max(axis=1)
        predicted = clipped.argmax(axis=1)
        correct = predicted == encoded
        bins = np.linspace(0.0, 1.0, 11)
        calibration_curve = []
        expected_calibration_error = 0.0
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (confidence >= lower) & ((confidence < upper) if upper < 1.0 else (confidence <= upper))
            count = int(mask.sum())
            if count == 0:
                continue
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            expected_calibration_error += (count / len(confidence)) * abs(mean_confidence - accuracy)
            calibration_curve.append(
                {"lower": float(lower), "upper": float(upper), "count": count, "meanConfidence": mean_confidence, "accuracy": accuracy}
            )
        return {
            "logLoss": float(ModelEvaluator.log_loss_metrics(truth, clipped, labels=CANONICAL_CLASS_ORDER)),
            "brierScore": float(np.mean(np.sum((clipped - one_hot) ** 2, axis=1))),
            "expectedCalibrationError": float(expected_calibration_error),
            "calibrationCurve": calibration_curve,
        }

    @staticmethod
    def _new_calibrator(estimator: Any, method: str, n_splits: int, gap: int) -> CalibratedClassifierCV:
        base = clone(estimator)
        if "early_stopping_rounds" in base.get_params(deep=False):
            base.set_params(early_stopping_rounds=None)
        return CalibratedClassifierCV(
            estimator=base,
            method=method,
            cv=TimeSeriesSplit(n_splits=n_splits, gap=max(0, gap)),
            ensemble=True,
        )

    def fit_select(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        *,
        gap_bars: int = 0,
    ) -> tuple[Any | None, np.ndarray | None, dict[str, Any]]:
        estimator = getattr(estimator, "estimator", estimator)
        y = pd.Series(y_train).astype(str).reset_index(drop=True)
        minimum_size = max(60, self.n_splits * 10)
        if len(X_train) < minimum_size or set(y.unique()) != set(CANONICAL_CLASS_ORDER):
            return None, None, {
                "status": "SKIPPED_INSUFFICIENT_CHRONOLOGICAL_TRAINING_DATA",
                "selectedMethod": None,
                "comparison": {},
            }

        holdout_size = max(1, int(np.ceil(len(X_train) * self.calibration_fraction)))
        fit_end = len(X_train) - holdout_size
        inner_X = X_train.iloc[:fit_end].reset_index(drop=True)
        inner_y = y.iloc[:fit_end].reset_index(drop=True)
        holdout_X = X_train.iloc[fit_end:].reset_index(drop=True)
        holdout_y = y.iloc[fit_end:].reset_index(drop=True)
        comparison: dict[str, Any] = {}
        fitted_calibrators: dict[str, Any] = {}
        for method in ("sigmoid", "isotonic"):
            try:
                inner_folds = min(self.n_splits, max(2, len(inner_X) // 20))
                calibrator = self._new_calibrator(estimator, method, inner_folds, gap_bars)
                calibrator.fit(inner_X, inner_y.map({label: index for index, label in enumerate(CANONICAL_CLASS_ORDER)}))
                aligned = self._align_probabilities(calibrator, calibrator.predict_proba(holdout_X))
                comparison[method] = {"status": "OK", **self._metrics(holdout_y, aligned)}
                fitted_calibrators[method] = calibrator
            except (ValueError, TypeError, RuntimeError) as exc:
                comparison[method] = {"status": "UNAVAILABLE", "reason": str(exc)}

        available = [method for method in ("sigmoid", "isotonic") if method in fitted_calibrators]
        if not available:
            return None, None, {
                "status": "SKIPPED_CALIBRATORS_UNAVAILABLE",
                "selectedMethod": None,
                "comparison": comparison,
            }
        selected_method = min(available, key=lambda method: comparison[method]["logLoss"])
        try:
            final_splits = min(self.n_splits, max(2, len(X_train) // 20))
            final_calibrator = self._new_calibrator(estimator, selected_method, final_splits, gap_bars)
            encoded_y = y.map({label: index for index, label in enumerate(CANONICAL_CLASS_ORDER)})
            final_calibrator.fit(X_train, encoded_y)
            validation_probabilities = self._align_probabilities(
                final_calibrator,
                final_calibrator.predict_proba(X_validation),
            )
            return final_calibrator, validation_probabilities, {
                "status": "CALIBRATED",
                "selectedMethod": selected_method,
                "selectionData": "chronological training-fold holdout",
                "comparison": comparison,
                "validationMetrics": self._metrics(y_validation, validation_probabilities),
            }
        except (ValueError, TypeError, RuntimeError) as exc:
            return None, None, {
                "status": "SKIPPED_FINAL_CALIBRATION_FAILED",
                "selectedMethod": selected_method,
                "comparison": comparison,
                "reason": str(exc),
            }
