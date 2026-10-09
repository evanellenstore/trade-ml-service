from __future__ import annotations

import pandas as pd
import numpy as np

from app.dataset.split.dataset_split_config import DatasetSplitConfig
from app.dataset.split.purged_chronological_dataset_splitter import PurgedChronologicalDatasetSplitter
from app.features.feature_schema import FEATURE_SCHEMA
from app.schemas.model_schema import ModelTrainRequest, ValidationStrategy, WalkForwardConfig
from app.training.walk_forward import WalkForwardConfig as WalkForwardSpec
from app.training.walk_forward import WalkForwardSplitter, WalkForwardTrainer
from app.training.oof_analysis import OOFAnalyzer


def _synthetic_rows(*, session_count: int = 360, rows_per_session: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2025-01-06")
    for session_index in range(session_count):
        session_start = base + pd.Timedelta(days=session_index)
        for minute_offset in range(rows_per_session):
            candle_time = session_start + pd.Timedelta(minutes=minute_offset)
            feature_time = candle_time
            target_end = candle_time + pd.Timedelta(minutes=60)
            rows.append(
                {
                    "candle_id": f"{session_index}-{minute_offset}",
                    "symbol_token": "14154",
                    "timeframe": "ONE_MINUTE",
                    "candle_time": candle_time,
                    "trading_date": session_start.date(),
                    "featureTimestamp": feature_time,
                    "targetEndTimestamp": target_end,
                    "open": float(100 + session_index + minute_offset),
                    "future_close": float(101 + session_index + minute_offset),
                    "label": "HOLD",
                    **{column: float(minute_offset + session_index + idx) for idx, column in enumerate(FEATURE_SCHEMA.columns)},
                }
            )
    frame = pd.DataFrame(rows)
    frame["label"] = "HOLD"
    frame.loc[frame["featureTimestamp"].dt.minute % 7 == 0, "label"] = "BUY"
    frame.loc[frame["featureTimestamp"].dt.minute % 11 == 0, "label"] = "SELL"
    return frame


def test_model_train_request_accepts_walk_forward_strategy() -> None:
    payload = ModelTrainRequest(
        symbolToken="14154",
        timeframe="ONE_MINUTE",
        predictionHorizonBars=60,
        buyThresholdPct=0.20,
        sellThresholdPct=-0.20,
        validationStrategy=ValidationStrategy.WALK_FORWARD,
        walkForward=WalkForwardConfig(
            foldCount=4,
            validationWindowSessions=60,
            expandingWindow=True,
            sessionAligned=True,
            purgeEnabled=True,
            embargoBars=0,
        ),
    )

    assert payload.validationStrategy == ValidationStrategy.WALK_FORWARD
    assert payload.walkForward is not None
    assert payload.walkForward.foldCount == 4
    assert payload.oofAnalysis.confidenceThresholds == [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    assert payload.oofAnalysis.marginThresholds == [0.05, 0.10, 0.15, 0.20, 0.25]


def test_walk_forward_splitter_generates_non_overlapping_validation_windows() -> None:
    rows = _synthetic_rows(session_count=360, rows_per_session=5)
    splitter = WalkForwardSplitter()

    dev_rows, test_rows = splitter.reserve_test_rows(rows, validation_ratio=0.15, test_ratio=0.15, train_ratio=0.70)
    assert len(test_rows) > 0
    assert dev_rows["trading_date"].nunique() + test_rows["trading_date"].nunique() == rows["trading_date"].nunique()

    config = WalkForwardConfig(foldCount=4, validationWindowSessions=60, expandingWindow=True, sessionAligned=True)
    folds = splitter.create_folds(dev_rows, config)

    assert len(folds) == 4
    validation_starts = [fold["validation"]["startSessionIndex"] for fold in folds]
    validation_ends = [fold["validation"]["endSessionIndex"] for fold in folds]
    for idx in range(1, len(folds)):
        assert validation_starts[idx] > validation_ends[idx - 1]
    assert all(fold["validation"]["sessionCount"] == 60 for fold in folds)
    assert all(fold["train"]["sessionCount"] > 0 for fold in folds)
    assert all(fold["train"]["rows"] > 0 for fold in folds)
    assert all(fold["validation"]["rows"] > 0 for fold in folds)


def test_walk_forward_rejects_insufficient_history() -> None:
    rows = _synthetic_rows(session_count=180, rows_per_session=5)
    splitter = WalkForwardSplitter()
    _, test_rows = splitter.reserve_test_rows(rows, validation_ratio=0.15, test_ratio=0.15, train_ratio=0.70)
    dev_rows = rows[~rows["candle_id"].isin(test_rows["candle_id"])].copy()

    config = WalkForwardConfig(foldCount=4, validationWindowSessions=60, expandingWindow=True, sessionAligned=True)
    try:
        splitter.create_folds(dev_rows, config)
        raise AssertionError("Expected insufficient history error")
    except ValueError as exc:
        assert "INSUFFICIENT_HISTORY_FOR_WALK_FORWARD" in str(exc)


def test_walk_forward_test_lock_is_preserved() -> None:
    rows = _synthetic_rows(session_count=300, rows_per_session=5)
    splitter = WalkForwardSplitter()
    dev_rows, test_rows = splitter.reserve_test_rows(rows, validation_ratio=0.15, test_ratio=0.15, train_ratio=0.70)
    assert not test_rows.empty
    assert all(test_row not in dev_rows["candle_id"].tolist() for test_row in test_rows["candle_id"].tolist())


def test_walk_forward_reserves_the_unchanged_holdout_test_sessions() -> None:
    rows = _synthetic_rows(session_count=360, rows_per_session=5)
    rows = rows.loc[
        ~((rows["trading_date"].astype(str) < "2025-02-15") & rows["candle_id"].str.endswith("-4"))
    ].reset_index(drop=True)
    config = DatasetSplitConfig(trainRatio=0.70, validationRatio=0.15, testRatio=0.15)
    _, _, holdout_test = PurgedChronologicalDatasetSplitter().split_frames(
        rows,
        config,
        dataset_fingerprint="reservation-regression",
    )

    development_rows, locked_test_rows, metadata = WalkForwardSplitter.reserve_test_partition(
        rows,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
    )

    assert set(locked_test_rows["candle_id"].astype(str)) == set(holdout_test["candle_id"].astype(str))
    test_sessions = set(locked_test_rows["trading_date"])
    assert test_sessions.isdisjoint(set(development_rows["trading_date"]))
    assert metadata["rowReconciliationValid"] is True
    assert metadata["lockedTest"]["locked"] is True
    assert metadata["lockedTest"]["evaluated"] is False
    assert development_rows["targetEndTimestamp"].max() < locked_test_rows["featureTimestamp"].min()


def test_fold_four_uses_latest_development_sessions_without_test_overlap() -> None:
    rows = _synthetic_rows(session_count=360, rows_per_session=5)
    development_rows, test_rows = WalkForwardSplitter.reserve_test_rows(
        rows,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
    )
    folds = WalkForwardSplitter().create_folds(
        development_rows,
        WalkForwardConfig(foldCount=4, validationWindowSessions=60, expandingWindow=True),
    )
    test_ids = set(test_rows["candle_id"].astype(str))
    test_sessions = set(test_rows["trading_date"])
    fold_four_validation = folds[-1]["validationDataFrame"]

    for fold in folds:
        train = fold["trainDataFrame"]
        validation = fold["validationDataFrame"]
        assert set(train["candle_id"].astype(str)).isdisjoint(test_ids)
        assert set(validation["candle_id"].astype(str)).isdisjoint(test_ids)
        assert set(train["trading_date"]).isdisjoint(test_sessions)
        assert set(validation["trading_date"]).isdisjoint(test_sessions)
        assert validation["trading_date"].nunique() == 60

    assert fold_four_validation["trading_date"].max() == development_rows["trading_date"].max()
    assert fold_four_validation["featureTimestamp"].max() < test_rows["featureTimestamp"].min()
    assert folds[0]["train"]["sessionCount"] < folds[-1]["train"]["sessionCount"]


def test_walk_forward_evaluation_never_predicts_on_locked_test_rows(monkeypatch, tmp_path) -> None:
    rows = _synthetic_rows(session_count=10, rows_per_session=5)
    rows.loc[rows["candle_id"].str.endswith("-1"), "label"] = "BUY"
    config = DatasetSplitConfig(trainRatio=0.70, validationRatio=0.15, testRatio=0.15)
    _, expected_test, _ = WalkForwardSplitter.reserve_test_partition(
        rows,
        train_ratio=config.trainRatio,
        validation_ratio=config.validationRatio,
        test_ratio=config.testRatio,
    )
    test_ids = set(expected_test["candle_id"].astype(str))
    predicted_ids: list[str] = []
    predict_proba_calls: list[int] = []

    class FakeModel:
        def predict(self, matrix: pd.DataFrame) -> np.ndarray:
            predicted_ids.extend(matrix.index.astype(str))
            offset_predictions = {"0": 0, "1": 2, "2": 1, "3": 2, "4": 0}
            return np.asarray(
                [offset_predictions[candle_id.rsplit("-", 1)[-1]] for candle_id in matrix.index.astype(str)],
                dtype=int,
            )

        def predict_proba(self, matrix: pd.DataFrame) -> np.ndarray:
            predict_proba_calls.append(len(matrix))
            raise AssertionError("predict_proba must not be called for TEST")

    class FakeTrainer:
        model_path = "/tmp/walk-forward-test-spy.joblib"

        @staticmethod
        def _fit_candidate(**kwargs):
            validation = kwargs["X_validation"]
            class_ids = [
                {"0": 0, "1": 2, "2": 1, "3": 2, "4": 0}[candle_id.rsplit("-", 1)[-1]]
                for candle_id in validation.index.astype(str)
            ]
            probabilities = np.asarray(
                [
                    [[0.75, 0.15, 0.10], [0.10, 0.15, 0.75], [0.10, 0.80, 0.10]][class_id]
                    for class_id in class_ids
                ]
            )
            return {
                "probabilities": probabilities,
                "trainedModel": FakeModel(),
            }

        @staticmethod
        def _decode_canonical_labels(values: np.ndarray) -> list[str]:
            return [ ["SELL", "HOLD", "BUY"][int(value)] for value in values ]

    def indexed_feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        matrix = frame.loc[:, feature_columns].copy()
        matrix.index = frame["candle_id"].astype(str)
        return matrix.astype(float)

    from app.training.xgboost_trainer import XGBoostTrainer

    monkeypatch.setattr(
        XGBoostTrainer,
        "validate_feature_matrix",
        staticmethod(indexed_feature_matrix),
    )
    result = WalkForwardTrainer().evaluate_candidate(
        rows=rows,
        config=WalkForwardSpec(
            foldCount=2,
            validationWindowSessions=2,
            expandingWindow=True,
            minimumTrainingSessions=2,
        ),
        split_config=config,
        dataset_fingerprint="prediction-spy",
        prediction_horizon_bars=60,
        trading_style="INTRADAY",
        feature_columns=list(FEATURE_SCHEMA.columns),
        trainer_factory=lambda **kwargs: FakeTrainer(),
        candidate_params={"max_depth": 2},
        oof_artifact_directory=str(tmp_path),
    )

    assert set(predicted_ids).isdisjoint(test_ids)
    assert predict_proba_calls == []
    assert result["testEvaluated"] is False
    assert result["testLocked"] is True
    assert result["walkForwardCoverage"]["noWalkForwardTestOverlap"] is True
    assert result["rowReconciliationValid"] is True
    out_of_fold = result["outOfFold"]
    assert out_of_fold["predictionCount"] == sum(fold["validation"]["rows"] for fold in result["folds"])
    stored_oof = pd.read_csv(out_of_fold["predictionArtifact"])
    expected_validation_ids = {
        str(candle_id)
        for fold in WalkForwardSplitter().create_folds(
            WalkForwardSplitter.reserve_test_rows(
                rows,
                train_ratio=config.trainRatio,
                validation_ratio=config.validationRatio,
                test_ratio=config.testRatio,
            )[0],
            WalkForwardConfig(foldCount=2, validationWindowSessions=2, expandingWindow=True),
        )
        for candle_id in fold["validationDataFrame"]["candle_id"]
    }
    assert set(stored_oof["candleId"].astype(str)) == expected_validation_ids
    assert set(stored_oof["candleId"].astype(str)).isdisjoint(test_ids)
    assert out_of_fold["confidenceAnalysis"]
    assert out_of_fold["marginAnalysis"]
    assert out_of_fold["combinedAnalysis"]
    assert out_of_fold["backtests"]
    assert all(policy["holdRule"] == "NO_TRADE" for policy in out_of_fold["signalPolicies"])
    assert all(backtest["buyTradeCount"] + backtest["sellTradeCount"] == backtest["tradeCount"] for backtest in out_of_fold["backtests"])
    if out_of_fold["bestConfidencePolicy"] is not None:
        assert out_of_fold["bestConfidencePolicy"]["tradeCount"] > 0
    if out_of_fold["bestCombinedPolicy"] is not None:
        assert out_of_fold["bestCombinedPolicy"]["tradeCount"] > 0
    assert all(fold["confusionMatrix"]["classOrder"] == ["SELL", "HOLD", "BUY"] for fold in result["folds"])
    assert all(
        sum(map(sum, fold["confusionMatrix"]["matrix"])) == fold["validation"]["rows"]
        for fold in result["folds"]
    )


def test_walk_forward_fingerprint_changes_with_test_boundary() -> None:
    fields = {
        "dataset_fingerprint": "dataset",
        "development_data": {"startTime": "2025-01-01", "endTime": "2025-08-01"},
        "fold_count": 4,
        "validation_window_sessions": 60,
        "expanding_window": True,
        "session_aligned": True,
        "purge_enabled": True,
        "embargo_bars": 0,
        "fold_boundaries": [{"validationEnd": "2025-08-01"}],
        "candidate_params": {"max_depth": 4},
        "model_identity": {"symbolToken": "14154"},
    }
    first = WalkForwardSplitter._fold_fingerprint(
        locked_test={"startTime": "2025-08-02", "endTime": "2025-08-31", "firstCandleId": "a", "lastCandleId": "z"},
        **fields,
    )
    repeated = WalkForwardSplitter._fold_fingerprint(
        locked_test={"startTime": "2025-08-02", "endTime": "2025-08-31", "firstCandleId": "a", "lastCandleId": "z"},
        **fields,
    )
    changed = WalkForwardSplitter._fold_fingerprint(
        locked_test={"startTime": "2025-08-03", "endTime": "2025-08-31", "firstCandleId": "b", "lastCandleId": "z"},
        **fields,
    )

    assert first == repeated
    assert first != changed


def test_oof_records_and_threshold_metrics_use_only_validation_predictions() -> None:
    rows = _synthetic_rows(session_count=8, rows_per_session=3)
    validation_rows = rows.iloc[:6].copy()
    actual = pd.Series(["BUY", "HOLD", "HOLD", "SELL", "SELL", "HOLD"])
    predictions = np.asarray(["BUY", "SELL", "HOLD", "BUY", "SELL", "HOLD"])
    probabilities = np.asarray(
        [
            [0.10, 0.15, 0.75],
            [0.75, 0.15, 0.10],
            [0.10, 0.80, 0.10],
            [0.08, 0.12, 0.80],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
        ]
    )
    records = OOFAnalyzer.create_records(
        validation_rows=validation_rows,
        actual_labels=actual,
        predicted_labels=predictions,
        probabilities=probabilities,
        fold_number=1,
    )
    result = OOFAnalyzer.analyze(
        records=records,
        development_rows=rows,
        trading_style="INTRADAY",
        locked_test_start=(rows["featureTimestamp"].max() + pd.Timedelta(days=1)),
        confidence_thresholds=[0.5, 0.75],
        margin_thresholds=[0.1, 0.5],
        combined_thresholds=[{"confidenceThreshold": 0.75, "marginThreshold": 0.5}],
        costs={
            "brokerage": 10.0,
            "transactionCostPct": 0.01,
            "slippagePct": 0.02,
            "sttPct": 0.01,
            "exchangeFeePct": 0.01,
        },
        capital_per_trade=100000,
        minimum_trade_count=1,
    )

    assert len(records) == result["predictionCount"] == len(validation_rows)
    assert records[0]["confidence"] == 0.75
    assert records[0]["margin"] == 0.60
    threshold = result["confidenceAnalysis"][1]
    assert threshold["predictionCount"] == len(validation_rows)
    assert threshold["BUY"]["predictedCount"] == 2
    assert threshold["SELL"]["predictedCount"] == 2
    assert threshold["HOLD"]["predictedCount"] == 2
    assert result["combinedAnalysis"][0]["predictionCount"] == 6
    assert result["backtestAssumptions"]["costs"]["brokerage"] == 10.0
    assert all(backtest["tradeCount"] >= 0 for backtest in result["backtests"])
    assert all(backtest["grossReturnPct"] >= backtest["netReturnPct"] for backtest in result["backtests"])
    assert all(backtest["buySignalCount"] + backtest["sellSignalCount"] == backtest["signalCount"] for backtest in result["backtests"])
    assert all(abs(backtest["roundTripCostPct"] - 0.11) < 1e-9 for backtest in result["backtests"])
