from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss

from app.features.feature_schema import FEATURE_COUNT, FEATURE_SCHEMA
from app.training.model_evaluator import ModelEvaluator

CANONICAL_CLASS_ORDER = ["SELL", "HOLD", "BUY"]
CANONICAL_CLASS_ENCODING = {label: index for index, label in enumerate(CANONICAL_CLASS_ORDER)}
CANONICAL_CLASS_DECODING = {index: label for label, index in CANONICAL_CLASS_ENCODING.items()}

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-missing environments
    XGBClassifier = None  # type: ignore[assignment]


class XGBoostTrainer:
    """Train and persist an XGBoost classifier for the canonical feature schema."""

    available = XGBClassifier is not None

    def __init__(self, model_path: str | Path, *, random_state: int = 42) -> None:
        self.model_path = Path(model_path)
        self.random_state = random_state


    @staticmethod
    def validate_feature_matrix(
        df: pd.DataFrame,
        feature_columns: list[str],
        *,
        forbidden_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        if not feature_columns:
            raise ValueError("At least one feature column is required")
        if len(feature_columns) != FEATURE_COUNT:
            raise ValueError(f"feature count must be {FEATURE_COUNT}; received {len(feature_columns)}")
        if list(feature_columns) != list(FEATURE_SCHEMA.columns):
            raise ValueError("feature names or order do not match the canonical schema")

        missing = [column for column in feature_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in input data: {missing}")

        forbidden = forbidden_columns or {
            "future_close",
            "future_return",
            "future_return_pct",
            "future_high",
            "future_low",
            "future_volume",
            "targetEndTimestamp",
            "label",
            "candle_time",
            "candle_id",
        }
        if set(feature_columns) & forbidden:
            raise ValueError(
                f"Forbidden columns detected in model input: {sorted(set(feature_columns) & forbidden)}"
            )

        X = df.loc[:, feature_columns].copy()
        X = X.replace([np.inf, -np.inf], 0.0)
        X = X.fillna(0.0)
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Model input contains non-finite values")
        return X.astype(float)

    @staticmethod
    def _candidate_grid() -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = [
            {"max_depth": 2, "learning_rate": 0.005, "min_child_weight": 1, "subsample": 0.60, "colsample_bytree": 0.60, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 0.5},
            {"max_depth": 2, "learning_rate": 0.01, "min_child_weight": 3, "subsample": 0.75, "colsample_bytree": 0.75, "gamma": 0.05, "reg_alpha": 0.01, "reg_lambda": 1.0},
            {"max_depth": 2, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 2.0},
            {"max_depth": 3, "learning_rate": 0.005, "min_child_weight": 1, "subsample": 0.70, "colsample_bytree": 0.70, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
            {"max_depth": 3, "learning_rate": 0.01, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.01, "reg_lambda": 2.0},
            {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 0.90, "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 5.0},
            {"max_depth": 4, "learning_rate": 0.01, "min_child_weight": 1, "subsample": 0.75, "colsample_bytree": 0.75, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
            {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.05, "reg_alpha": 0.01, "reg_lambda": 2.0},
            {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 1.0, "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 5.0},
            {"max_depth": 5, "learning_rate": 0.01, "min_child_weight": 1, "subsample": 0.80, "colsample_bytree": 0.80, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0},
            {"max_depth": 5, "learning_rate": 0.03, "min_child_weight": 3, "subsample": 0.90, "colsample_bytree": 0.90, "gamma": 0.05, "reg_alpha": 0.01, "reg_lambda": 5.0},
            {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.95, "colsample_bytree": 1.0, "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 10.0},
            {"max_depth": 6, "learning_rate": 0.01, "min_child_weight": 1, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 5.0},
            {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 3, "subsample": 0.90, "colsample_bytree": 0.90, "gamma": 0.05, "reg_alpha": 0.1, "reg_lambda": 10.0},
            {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 1.0, "gamma": 0.2, "reg_alpha": 0.1, "reg_lambda": 10.0},
            {"max_depth": 8, "learning_rate": 0.005, "min_child_weight": 3, "subsample": 0.70, "colsample_bytree": 0.70, "gamma": 0.1, "reg_alpha": 0.01, "reg_lambda": 2.0},
            {"max_depth": 8, "learning_rate": 0.01, "min_child_weight": 5, "subsample": 0.80, "colsample_bytree": 0.80, "gamma": 0.2, "reg_alpha": 0.1, "reg_lambda": 5.0},
            {"max_depth": 8, "learning_rate": 0.03, "min_child_weight": 8, "subsample": 0.90, "colsample_bytree": 0.90, "gamma": 0.25, "reg_alpha": 0.1, "reg_lambda": 10.0},
        ]
        return candidates

    @staticmethod
    def _encode_canonical_labels(values: pd.Series | list[str] | np.ndarray) -> np.ndarray:
        normalized = pd.Series(values).astype(str)
        encoded = normalized.map(CANONICAL_CLASS_ENCODING).to_numpy(dtype=int)
        if np.isnan(encoded).any():
            missing = sorted(set(normalized.unique()) - set(CANONICAL_CLASS_ENCODING))
            raise ValueError(f"Unsupported canonical labels detected: {missing}")
        return encoded

    @staticmethod
    def _decode_canonical_labels(values: pd.Series | list[int] | np.ndarray) -> np.ndarray:
        as_int = pd.Series(values).astype(int)
        decoded = as_int.map(CANONICAL_CLASS_DECODING).astype(str).to_numpy(dtype=object)
        if (as_int < 0).any() or (as_int > 2).any():
            raise ValueError("Decoded prediction values fall outside canonical encoding")
        return decoded

    def _fit_candidate(
        self,
        *,
        X_train: pd.DataFrame,
        X_validation: pd.DataFrame,
        y_train: pd.Series,
        y_validation: pd.Series,
        hyperparameters: dict[str, Any],
        label_order: list[str],
    ) -> dict[str, Any]:
        normalized_train = pd.Series(y_train).astype(str).reset_index(drop=True)
        local_classes = [label for label in CANONICAL_CLASS_ORDER if label in set(normalized_train)]
        if not local_classes:
            raise ValueError("Training fold contains no canonical target classes")
        local_encoding = {label: index for index, label in enumerate(local_classes)}
        y_train_encoded = normalized_train.map(local_encoding).to_numpy(dtype=int)
        normalized_validation = pd.Series(y_validation).astype(str).reset_index(drop=True)
        validation_mask = normalized_validation.isin(local_classes).to_numpy()
        if len(local_classes) == 1:
            model = DummyClassifier(strategy="constant", constant=0)
            model.fit(X_train, y_train_encoded)
        else:
            model_kwargs = {
                "objective": "binary:logistic" if len(local_classes) == 2 else "multi:softprob",
                "eval_metric": "logloss" if len(local_classes) == 2 else "mlogloss",
                "n_estimators": hyperparameters.get("n_estimators", 200),
                "max_depth": int(hyperparameters.get("max_depth", 5)),
                "learning_rate": float(hyperparameters.get("learning_rate", 0.05)),
                "subsample": float(hyperparameters.get("subsample", 0.85)),
                "colsample_bytree": float(hyperparameters.get("colsample_bytree", 0.85)),
                "min_child_weight": int(hyperparameters.get("min_child_weight", 1)),
                "gamma": float(hyperparameters.get("gamma", 0.0)),
                "reg_alpha": float(hyperparameters.get("reg_alpha", 0.0)),
                "reg_lambda": float(hyperparameters.get("reg_lambda", 1.0)),
                "random_state": self.random_state,
                "n_jobs": 1,
            }
            if len(local_classes) > 2:
                model_kwargs["num_class"] = len(local_classes)
            eval_set = []
            if validation_mask.any():
                model_kwargs["early_stopping_rounds"] = int(hyperparameters.get("early_stopping_rounds", 20))
                eval_set = [
                    (
                        X_validation.loc[validation_mask],
                        normalized_validation.loc[validation_mask].map(local_encoding).to_numpy(dtype=int),
                    )
                ]
            model = XGBClassifier(**model_kwargs)
            model.fit(X_train, y_train_encoded, eval_set=eval_set or None, verbose=False)
        adapted_model = CanonicalClassAdapter(model, local_classes)
        validation_probabilities = adapted_model.predict_proba(X_validation)
        validation_predictions_labels = np.asarray(
            [CANONICAL_CLASS_ORDER[index] for index in adapted_model.predict(X_validation)],
            dtype=object,
        )
        metrics = ModelEvaluator.evaluate_predictions(y_validation, validation_predictions_labels, labels=label_order)
        metrics["logLoss"] = float(
            ModelEvaluator.log_loss_metrics(
                y_validation,
                validation_probabilities,
                labels=CANONICAL_CLASS_ORDER,
            )
        )
        metrics["bestIteration"] = getattr(model, "best_iteration", None)
        metrics["bestScore"] = getattr(model, "best_score", None)
        return {
            "hyperparameters": hyperparameters,
            "validationMetrics": metrics,
            "validation_accuracy": float(metrics["accuracy"]),
            "probabilities": validation_probabilities,
            "trainedModel": adapted_model,
            "labelEncoder": {"mapping": local_encoding, "reverseMapping": {value: key for key, value in local_encoding.items()}},
        }

    def train(
        self,
        *,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "label",
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if XGBClassifier is None:
            raise RuntimeError(
                "XGBoost is not installed in the active Python environment. "
                "Install the project requirements with `pip install -r requirements.txt`."
            )
        X_train = self.validate_feature_matrix(train_df, feature_columns)
        X_validation = self.validate_feature_matrix(validation_df, feature_columns)
        if X_train.shape != (len(train_df), FEATURE_COUNT):
            raise ValueError(f"X_train shape mismatch: expected {(len(train_df), FEATURE_COUNT)}, received {X_train.shape}")
        if X_validation.shape != (len(validation_df), FEATURE_COUNT):
            raise ValueError(f"X_validation shape mismatch: expected {(len(validation_df), FEATURE_COUNT)}, received {X_validation.shape}")
        y_train = train_df[label_column].astype(str)
        y_validation = validation_df[label_column].astype(str)
        label_order = list(CANONICAL_CLASS_ORDER)
        if len(label_order) < 2:
            raise ValueError("Training data must contain at least two target classes")
        params = hyperparameters or {
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 20,
        }
        result = self._fit_candidate(
            X_train=X_train,
            X_validation=X_validation,
            y_train=y_train,
            y_validation=y_validation,
            hyperparameters=params,
            label_order=label_order,
        )
        model = result["trainedModel"]
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        metadata_path = self.model_path.with_suffix(self.model_path.suffix + ".metadata.json")
        metadata = {
            "model_path": str(self.model_path),
            "feature_columns": feature_columns,
            "label_column": label_column,
            "model_type": "XGBClassifier",
            "validation_rows": int(len(validation_df)),
            "validation_accuracy": float(result["validation_accuracy"]),
            "validation_metrics": result["validationMetrics"],
            "hyperparameters": params,
            "trained_at": pd.Timestamp.now().isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {
            "model_path": str(self.model_path),
            "model_metadata_path": str(metadata_path),
            "feature_columns": feature_columns,
            "validation_rows": int(len(validation_df)),
            "validation_accuracy": float(result["validation_accuracy"]),
            "validation_metrics": result["validationMetrics"],
            "hyperparameters": params,
            "probabilities_shape": list(result["probabilities"].shape),
            "class_order": label_order,
        }

    def search(
        self,
        *,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "label",
        candidate_count: int = 12,
    ) -> dict[str, Any]:
        X_train = self.validate_feature_matrix(train_df, feature_columns)
        X_validation = self.validate_feature_matrix(validation_df, feature_columns)
        y_train = train_df[label_column].astype(str)
        y_validation = validation_df[label_column].astype(str)
        label_order = list(CANONICAL_CLASS_ORDER)
        candidates = self._candidate_grid()[:candidate_count]
        leaderboard: list[dict[str, Any]] = []
        for params in candidates:
            result = self._fit_candidate(
                X_train=X_train,
                X_validation=X_validation,
                y_train=y_train,
                y_validation=y_validation,
                hyperparameters={**params, "n_estimators": 200, "early_stopping_rounds": 20},
                label_order=label_order,
            )
            score = float(result["validationMetrics"]["macroF1"])
            leaderboard.append({
                "hyperparameters": params,
                "validationMetrics": result["validationMetrics"],
                "score": score,
            })
        leaderboard.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(leaderboard, start=1):
            item["rank"] = rank
        best = leaderboard[0]
        return {
            "candidatesEvaluated": len(leaderboard),
            "leaderboard": leaderboard,
            "bestHyperparameters": best["hyperparameters"],
            "bestValidationMetrics": best["validationMetrics"],
            "bestModel": best,
        }


class CanonicalClassAdapter:
    """Expose a fold-specific classifier on the stable SELL/HOLD/BUY axis."""

    def __init__(self, estimator: Any, local_classes: list[str]) -> None:
        self.estimator = estimator
        self.local_classes = list(local_classes)
        self.classes_ = np.arange(len(CANONICAL_CLASS_ORDER))

    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        local_predictions = np.asarray(self.estimator.predict(matrix), dtype=int)
        return np.asarray(
            [CANONICAL_CLASS_ORDER.index(self.local_classes[int(value)]) for value in local_predictions],
            dtype=int,
        )

    def predict_proba(self, matrix: pd.DataFrame) -> np.ndarray:
        local_probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=float)
        aligned = np.zeros((len(local_probabilities), len(CANONICAL_CLASS_ORDER)), dtype=float)
        estimator_classes = np.asarray(
            getattr(self.estimator, "classes_", np.arange(len(self.local_classes))),
            dtype=int,
        )
        for local_column, estimator_class in enumerate(estimator_classes):
            label = self.local_classes[int(estimator_class)]
            aligned[:, CANONICAL_CLASS_ORDER.index(label)] = local_probabilities[:, local_column]
        row_sums = aligned.sum(axis=1, keepdims=True)
        return np.divide(aligned, row_sums, out=np.zeros_like(aligned), where=row_sums > 0.0)

    def get_booster(self):
        return self.estimator.get_booster() if hasattr(self.estimator, "get_booster") else None

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(getattr(self.estimator, "feature_importances_", []), dtype=float)

    @property
    def n_features_in_(self) -> int:
        if hasattr(self.estimator, "n_features_in_"):
            return int(self.estimator.n_features_in_)
        if hasattr(self.estimator, "get_booster"):
            return int(self.estimator.get_booster().num_features())
        raise AttributeError("Wrapped estimator does not expose its input feature count")
