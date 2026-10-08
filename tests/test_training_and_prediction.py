from __future__ import annotations

import joblib

import numpy as np
import pandas as pd

from app.features.feature_schema import FEATURE_SCHEMA
from app.prediction.prediction_service import PredictionService
from app.training.lightgbm_trainer import LightGBMTrainer
from app.training.model_evaluator import ModelEvaluator
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


def test_model_evaluator_returns_majority_and_stratified_baseline_metrics() -> None:
    labels = pd.Series(["BUY", "BUY", "SELL", "SELL", "HOLD", "BUY"])

    majority = ModelEvaluator.majority_class_baseline(labels)
    stratified = ModelEvaluator.stratified_baseline(labels, random_state=42)

    assert majority["baseline_type"] == "majority"
    assert stratified["baseline_type"] == "stratified"
    assert 0.0 <= majority["accuracy"] <= 1.0
    assert 0.0 <= stratified["accuracy"] <= 1.0


def test_lightgbm_trainer_reports_explicit_missing_dependency() -> None:
    trainer = LightGBMTrainer(model_path="/tmp/unused.joblib")

    assert trainer.available is False
    try:
        trainer.train(
            train_df=_training_rows().iloc[:30],
            validation_df=_training_rows().iloc[30:],
            feature_columns=list(FEATURE_SCHEMA.columns),
            label_column="label",
        )
        raise AssertionError("Expected a RuntimeError when LightGBM is unavailable")
    except RuntimeError as exc:
        assert "lightgbm" in str(exc).lower()


def test_model_evaluator_support_uses_true_class_counts() -> None:
    y_true = (
        ["SELL"] * 8224
        + ["HOLD"] * 6930
        + ["BUY"] * 8642
    )
    y_pred = (
        ["SELL"] * 3099 + ["HOLD"] * 2958 + ["BUY"] * 2167
        + ["SELL"] * 2301 + ["HOLD"] * 3228 + ["BUY"] * 1401
        + ["SELL"] * 3405 + ["HOLD"] * 2894 + ["BUY"] * 2343
    )
    metrics = ModelEvaluator.evaluate_predictions(y_true, y_pred, labels=["SELL", "HOLD", "BUY"])

    assert metrics["classMetrics"]["SELL"]["support"] == 8224
    assert metrics["classMetrics"]["HOLD"]["support"] == 6930
    assert metrics["classMetrics"]["BUY"]["support"] == 8642


def test_xgboost_search_ranks_by_macro_f1_desc(monkeypatch) -> None:
    trainer = XGBoostTrainer(model_path="/tmp/rank-test.joblib")
    rows = _training_rows()
    df = rows.iloc[:60].copy()
    scores = {3: 0.41, 4: 0.46, 5: 0.43}

    def fake_fit_candidate(**kwargs):
        hyper = kwargs["hyperparameters"]
        macro_f1 = scores[int(hyper["max_depth"])]
        metrics = {
            "accuracy": 0.5,
            "balancedAccuracy": 0.5,
            "macroPrecision": 0.5,
            "macroRecall": 0.5,
            "macroF1": macro_f1,
            "weightedF1": 0.5,
            "confusionMatrix": {"classOrder": ["SELL", "HOLD", "BUY"], "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "classMetrics": {},
        }
        return {"validationMetrics": metrics, "validation_accuracy": 0.5, "probabilities": np.zeros((len(df), 3)), "trainedModel": object(), "labelEncoder": object()}

    monkeypatch.setattr(trainer, "_fit_candidate", fake_fit_candidate)
    monkeypatch.setattr(trainer, "_candidate_grid", lambda: [{"max_depth": 3}, {"max_depth": 4}, {"max_depth": 5}])
    monkeypatch.setattr(trainer, "validate_feature_matrix", lambda df, feature_columns: df.loc[:, feature_columns].astype(float))

    result = trainer.search(
        train_df=df.iloc[:40],
        validation_df=df.iloc[40:],
        feature_columns=list(FEATURE_SCHEMA.columns),
        label_column="label",
        candidate_count=3,
    )

    ranks = [item["rank"] for item in result["leaderboard"]]
    scores_list = [item["score"] for item in result["leaderboard"]]
    assert ranks == [1, 2, 3]
    assert scores_list == sorted(scores_list, reverse=True)


def test_model_training_service_retains_incumbent_when_challenger_is_weaker() -> None:
    incumbent = {"validationMetrics": {"macroF1": 0.40, "accuracy": 0.38}, "model_path": "/tmp/incumbent.joblib"}
    challenger = {"validationMetrics": {"macroF1": 0.35, "accuracy": 0.36}, "model_path": "/tmp/challenger.joblib"}

    decision = ModelEvaluator.promote_candidate_if_better(incumbent, challenger, minimum_promotion_delta=0.0)

    assert decision["promotionDecision"] == "INCUMBENT_RETAINED"
    assert decision["selectedModel"] == incumbent
