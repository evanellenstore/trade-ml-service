from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from app.features.feature_schema import FEATURE_SCHEMA
from app.training.model_evaluator import ModelEvaluator

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-missing environments
    LGBMClassifier = None  # type: ignore[assignment]


class LightGBMTrainer:
    """Train and persist a LightGBM classifier when the dependency is available."""

    available = LGBMClassifier is not None

    def __init__(self, model_path: str | Path, *, random_state: int = 42) -> None:
        self.model_path = Path(model_path)
        self.random_state = random_state

    @staticmethod
    def validate_feature_matrix(
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        if len(feature_columns) != 26:
            raise ValueError(f"feature count must be 26; received {len(feature_columns)}")
        if list(feature_columns) != list(FEATURE_SCHEMA.columns):
            raise ValueError("feature names or order do not match the canonical schema")
        X = df.loc[:, feature_columns].copy()
        X = X.replace([np.inf, -np.inf], 0.0)
        X = X.fillna(0.0)
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Model input contains non-finite values")
        return X.astype(float)

    def train(
        self,
        *,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "label",
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if LGBMClassifier is None:
            raise RuntimeError(
                "LIGHTGBM_UNAVAILABLE: lightgbm is not installed in the active Python environment. "
                "Install the optional dependency before attempting LightGBM training."
            )

        X_train = self.validate_feature_matrix(train_df, feature_columns)
        X_validation = self.validate_feature_matrix(validation_df, feature_columns)
        y_train = train_df[label_column].astype(str)
        y_validation = validation_df[label_column].astype(str)
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_validation_encoded = label_encoder.transform(y_validation)
        params = hyperparameters or {
            "objective": "multiclass",
            "num_class": len(label_encoder.classes_),
            "n_estimators": 200,
            "learning_rate": 0.08,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": self.random_state,
            "n_jobs": 1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train_encoded, eval_set=[(X_validation, y_validation_encoded)], verbose=False)
        validation_predictions = model.predict(X_validation)
        validation_predictions_labels = label_encoder.inverse_transform(validation_predictions)
        probabilities = model.predict_proba(X_validation)
        metrics = ModelEvaluator.evaluate_predictions(y_validation, validation_predictions_labels, labels=list(label_encoder.classes_))
        metrics["logLoss"] = float(log_loss(y_validation, probabilities, labels=list(label_encoder.classes_)))
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        metadata_path = self.model_path.with_suffix(self.model_path.suffix + ".metadata.json")
        metadata_path.write_text(json.dumps({
            "model_path": str(self.model_path),
            "feature_columns": feature_columns,
            "label_column": label_column,
            "model_type": "LGBMClassifier",
            "validation_rows": int(len(validation_df)),
            "validation_metrics": metrics,
            "hyperparameters": params,
            "trained_at": pd.Timestamp.now().isoformat(),
        }, indent=2), encoding="utf-8")
        return {
            "model_path": str(self.model_path),
            "model_metadata_path": str(metadata_path),
            "feature_columns": feature_columns,
            "validation_rows": int(len(validation_df)),
            "validation_metrics": metrics,
            "hyperparameters": params,
            "class_order": list(label_encoder.classes_),
            "probabilities_shape": list(probabilities.shape),
        }
