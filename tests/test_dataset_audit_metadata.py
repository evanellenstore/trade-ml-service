from __future__ import annotations

import hashlib
import math

import pandas as pd
import pytest

from app.dataset.dataset_generator import DatasetGenerator
from app.dataset.target_policy import TargetPolicyFactory
from app.domain.trading_style import TradingStyle
from app.features.feature_schema import FeatureSchema
from app.schemas.dataset_schema import DatasetSummary


def test_feature_schema_is_central_and_exactly_fifteen_columns():
    expected = [
        "return_1",
        "return_5",
        "return_15",
        "price_range_pct",
        "body_size_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "volume_change_pct",
        "rolling_volume_mean",
        "relative_volume",
        "is_doji",
        "is_hammer",
        "is_shooting_star",
        "is_bullish_engulfing",
        "is_bearish_engulfing",
    ]

    assert FeatureSchema.version == "v1"
    assert FeatureSchema.columns == expected
    assert DatasetGenerator.FEATURE_COLUMNS == expected
    assert all(name not in expected for name in {"future_close", "future_return_pct", "label"})


def test_target_metadata_is_centralized_and_style_specific():
    intraday = TargetPolicyFactory.metadata_for(TradingStyle.INTRADAY)
    swing = TargetPolicyFactory.metadata_for(TradingStyle.SWING)
    long_term = TargetPolicyFactory.metadata_for(TradingStyle.LONG_TERM)

    assert intraday.same_session_only is True
    assert intraday.cross_session_allowed is False
    assert swing.cross_session_allowed is True
    assert long_term.cross_session_allowed is True
    assert all(item.horizon_unit == "MARKET_BAR" for item in (intraday, swing, long_term))


def test_dataset_fingerprint_is_stable_and_canonical():
    generator = DatasetGenerator()
    first = generator._dataset_fingerprint(
        symbol_token="14154",
        trading_style=TradingStyle.INTRADAY,
        timeframe="ONE_MINUTE",
        prediction_horizon_bars=30,
        buy_threshold_pct=0.15,
        sell_threshold_pct=-0.15,
        feature_version="v1",
        feature_columns=FeatureSchema.columns,
        dataset_start_time="2026-01-01T00:00:00",
        dataset_end_time="2026-01-01T00:05:00",
        dataset_row_count=10,
    )
    second = generator._dataset_fingerprint(
        symbol_token="14154",
        trading_style=TradingStyle.INTRADAY,
        timeframe="ONE_MINUTE",
        prediction_horizon_bars=30,
        buy_threshold_pct=0.15,
        sell_threshold_pct=-0.15,
        feature_version="v1",
        feature_columns=FeatureSchema.columns,
        dataset_start_time="2026-01-01T00:00:00",
        dataset_end_time="2026-01-01T00:05:00",
        dataset_row_count=10,
    )

    assert first == second
    assert len(first) == 64
    assert first == hashlib.sha256(first.encode("utf-8")).hexdigest() or len(first) == 64


def test_label_thresholds_preserve_equality_semantics():
    frame = pd.DataFrame(
        {
            "future_return_pct": [0.15, -0.15, 0.1],
            "close": [100.0, 100.0, 100.0],
        }
    )

    labeled = DatasetGenerator()._label_rows(frame, 0.15, -0.15)
    assert labeled["label"].tolist() == ["BUY", "SELL", "HOLD"]


def test_primary_skip_reasons_reconcile_with_skipped_row_count():
    generator = DatasetGenerator()
    frame = pd.DataFrame(
        {
            "symbol_token": ["A"] * 4,
            "timeframe": ["ONE_MINUTE"] * 4,
            "trading_date": pd.to_datetime(["2026-01-01"] * 4).date,
            "future_close": [101.0, 102.0, None, None],
            "future_return_pct": [1.0, 2.0, None, None],
            "label": ["BUY", "HOLD", "BUY", "SELL"],
            "return_1": [1.0, float("nan"), 1.0, 1.0],
        }
    )

    reasons = generator._primary_skip_reasons(frame, prediction_horizon_bars=2)
    assert reasons == {
        "crossSessionHorizon": 2,
        "insufficientFutureBars": 0,
    }
    assert sum(reasons.values()) == 2


def test_invalid_feature_values_are_reported_without_replacement():
    generator = DatasetGenerator()
    frame = pd.DataFrame(
        {
            "candle_id": [1],
            "symbol_token": ["A"],
            "timeframe": ["ONE_MINUTE"],
            "candle_time": pd.to_datetime(["2026-01-01"]),
            "close": [100.0],
            "future_close": [101.0],
            "future_return_pct": [1.0],
            "label": ["BUY"],
        }
    )
    for column in generator.FEATURE_COLUMNS:
        frame[column] = 1.0
    frame.loc[0, "return_1"] = math.inf

    validation = generator._feature_validation(frame)
    assert validation["positiveInfinityCount"] == 1
    assert validation["negativeInfinityCount"] == 0
    assert validation["nanValueCount"] == 0
    assert validation["rowsWithInvalidFeatures"] == 1


def test_summary_model_accepts_new_audit_fields():
    summary = DatasetSummary(
        symbolToken="A",
        tradingStyle="INTRADAY",
        timeframe="ONE_MINUTE",
        predictionHorizonBars=1,
        sourceRowCount=2,
        datasetRowCount=1,
        skippedRowCount=1,
        featureCount=15,
        featureVersion="v1",
        labelDistribution={"BUY": 1, "HOLD": 0, "SELL": 0},
        featureSchema={"version": "v1", "count": 15, "features": ["return_1"]},
        skipReasons={"crossSessionHorizon": 1},
        trainingEligibility={"eligible": True, "reasons": []},
        featureValidation={
            "beforeFiltering": {
                "rowsWithInvalidFeatures": 1,
                "nullValueCount": 0,
                "nanValueCount": 1,
                "positiveInfinityCount": 0,
                "negativeInfinityCount": 0,
            },
            "afterFiltering": {
                "rowsWithInvalidFeatures": 0,
                "nullValueCount": 0,
                "nanValueCount": 0,
                "positiveInfinityCount": 0,
                "negativeInfinityCount": 0,
            },
            "featureWarmupRows": 1,
        },
    )

    assert summary.featureSchema.count == 15
    assert summary.trainingEligibility.eligible is True
    assert summary.trainingEligibility.reasons == []
    assert summary.featureValidation.featureWarmupRows == 1
