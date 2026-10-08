from __future__ import annotations

import joblib

import pandas as pd

from app.features.feature_schema import FEATURE_SCHEMA
from app.prediction.prediction_service import PredictionService
from app.training.xgboost_trainer import XGBoostTrainer


def _training_rows() -> pd.DataFrame:
    labels = ["BUY", "HOLD", "SELL"] * 15
    rows = pd.DataFrame(
        {
            "candle_id": range(45),
            "label": labels,
        }
    )
    for column in FEATURE_SCHEMA.columns:
        rows[column] = range(len(rows), 0, -1)
    return rows


def test_trainer_fits_saves_and_returns_validation_metrics(tmp_path) -> None:
    rows = _training_rows()
    model_path = tmp_path / "model.joblib"
    trainer = XGBoostTrainer(model_path=model_path)

    result = trainer.train(
        train_df=rows.iloc[:30],
        validation_df=rows.iloc[30:],
        feature_columns=list(FEATURE_SCHEMA.columns),
        label_column="label",
    )

    assert model_path.exists()
    assert result["model_path"] == str(model_path)
    assert result["validation_rows"] == 15
    assert 0.0 <= result["validation_accuracy"] <= 1.0
    assert joblib.load(model_path).n_features_in_ == len(FEATURE_SCHEMA.columns)


def test_prediction_service_uses_saved_model_and_canonical_features(tmp_path) -> None:
    rows = _training_rows()
    model_path = tmp_path / "model.joblib"
    XGBoostTrainer(model_path=model_path).train(
        train_df=rows.iloc[:30],
        validation_df=rows.iloc[30:],
        feature_columns=list(FEATURE_SCHEMA.columns),
        label_column="label",
    )

    service = PredictionService(model_path=str(model_path))
    prediction = service.predict(rows.iloc[[0]])

    assert prediction["prediction"] in {"BUY", "HOLD", "SELL"}
    assert 0.0 <= prediction["probability"] <= 1.0
    assert set(prediction["probabilities"]) == {"BUY", "HOLD", "SELL"}
