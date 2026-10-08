from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class ScikitLearnTrainer:
    """Train and persist a deterministic classifier for the canonical feature schema."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def train(
        self,
        *,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "label",
    ) -> dict[str, Any]:
        if not feature_columns:
            raise ValueError("At least one feature column is required")
        if label_column not in train_df.columns or label_column not in validation_df.columns:
            raise ValueError(f"Missing label column: {label_column}")

        missing_train = [column for column in feature_columns if column not in train_df.columns]
        missing_validation = [column for column in feature_columns if column not in validation_df.columns]
        if missing_train or missing_validation:
            raise ValueError(
                f"Missing feature columns: train={missing_train}, validation={missing_validation}"
            )

        X_train = train_df.loc[:, feature_columns].astype(float)
        y_train = train_df[label_column].astype(str)
        X_validation = validation_df.loc[:, feature_columns].astype(float)
        y_validation = validation_df[label_column].astype(str)

        if X_train.empty or X_validation.empty:
            raise ValueError("Training and validation datasets must not be empty")
        if y_train.nunique() < 2:
            raise ValueError("Training data must contain at least two target classes")

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train, y_train)

        validation_predictions = model.predict(X_validation)
        validation_accuracy = float(accuracy_score(y_validation, validation_predictions))
        model_report = classification_report(
            y_validation,
            validation_predictions,
            labels=sorted(y_train.unique()),
            output_dict=True,
            zero_division=0,
        )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)

        metadata = {
            "model_path": str(self.model_path),
            "feature_columns": feature_columns,
            "label_column": label_column,
            "model_type": "LogisticRegression",
            "validation_rows": int(len(validation_df)),
            "validation_accuracy": validation_accuracy,
            "validation_report": model_report,
            "trained_at": pd.Timestamp.now().isoformat(),
        }
        metadata_path = self.model_path.with_suffix(self.model_path.suffix + ".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "model_path": str(self.model_path),
            "model_metadata_path": str(metadata_path),
            "feature_columns": feature_columns,
            "validation_rows": int(len(validation_df)),
            "validation_accuracy": validation_accuracy,
            "validation_report": model_report,
        }
