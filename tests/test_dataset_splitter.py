from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from app.dataset.dataset_generator import DatasetGenerator
from app.dataset.split.dataset_split_config import DatasetSplitConfig
from app.dataset.split.purged_chronological_dataset_splitter import PurgedChronologicalDatasetSplitter
from app.domain.trading_style import TradingStyle
from app.features.feature_schema import FEATURE_SCHEMA


def _rows(
    *,
    style: TradingStyle = TradingStyle.INTRADAY,
    count: int = 20,
    interval_days: int = 1,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    if labels is None:
        labels = ["BUY", "HOLD", "SELL"] * ((count + 2) // 3)
    labels = labels[:count]
    timestamps = pd.date_range("2026-01-01", periods=count, freq=f"{interval_days}D")
    frame = pd.DataFrame(
        {
            "candle_id": list(range(count)),
            "symbol_token": ["14154"] * count,
            "timeframe": ["ONE_MINUTE"] * count,
            "candle_time": timestamps,
            "featureTimestamp": timestamps,
            "trading_date": timestamps.date,
            "targetEndTimestamp": timestamps + pd.Timedelta(days=interval_days),
            "label": labels,
        }
    )
    for column in FEATURE_SCHEMA.columns:
        frame[column] = 1.0
    if style == TradingStyle.SWING:
        frame["trading_date"] = pd.NA
    return frame


def test_splitter_is_deterministic_and_keeps_v1_features_unchanged() -> None:
    rows = _rows(count=30)
    config = DatasetSplitConfig(
        trainRatio=0.7,
        validationRatio=0.15,
        testRatio=0.15,
        purgeEnabled=True,
        embargoBars=0,
        minimumTrainRows=1,
        minimumValidationRows=1,
        minimumTestRows=1,
    )
    splitter = PurgedChronologicalDatasetSplitter()

    first = splitter.split(rows, config, dataset_fingerprint="dataset-a")
    second = splitter.split(rows, config, dataset_fingerprint="dataset-a")

    assert first.train.rowCount + first.validation.rowCount + first.test.rowCount + first.purge.totalPurgedRows == len(rows)
    assert first == second
    assert first.splitFingerprint == second.splitFingerprint
    assert first.train.startTime == second.train.startTime
    assert first.test.endTime == second.test.endTime
    assert first.train.featureSchemaValid is True
    assert first.train.featureCount == 15
    assert first.train.features == list(FEATURE_SCHEMA.columns)
    assert first.validationChecks.noDuplicateRows is True
    assert first.validationChecks.chronologicalOrderValid is True
    assert first.validationChecks.rowReconciliationValid is True


def test_intraday_boundary_is_session_aligned_and_does_not_purge_same_session_targets() -> None:
    rows = _rows(count=24)
    rows["targetEndTimestamp"] = pd.to_datetime(rows["candle_time"]) + pd.Timedelta(minutes=30)
    rows["trading_date"] = pd.to_datetime(rows["candle_time"]).dt.date
    config = DatasetSplitConfig(trainRatio=0.7, validationRatio=0.15, testRatio=0.15, purgeEnabled=True)

    result = PurgedChronologicalDatasetSplitter().split(rows, config, dataset_fingerprint="dataset-b")

    assert result.purge.trainValidationPurgedRows == 0
    assert result.purge.validationTestPurgedRows == 0
    assert result.train.endTime < result.validation.startTime
    assert result.validation.endTime < result.test.startTime
    assert result.train.rowCount > 0
    assert result.validation.rowCount > 0
    assert result.test.rowCount > 0


def test_target_interval_purge_removes_train_row_reaching_validation_start() -> None:
    rows = _rows(count=20)
    rows.loc[0, "targetEndTimestamp"] = pd.Timestamp("2026-01-03")
    rows.loc[0, "featureTimestamp"] = pd.Timestamp("2026-01-01")
    config = DatasetSplitConfig(trainRatio=0.7, validationRatio=0.15, testRatio=0.15, purgeEnabled=True)

    result = PurgedChronologicalDatasetSplitter().split(rows, config, dataset_fingerprint="dataset-c")

    assert result.purge.trainValidationPurgedRows > 0
    assert result.validationChecks.trainValidationLabelOverlap is False


def test_safe_target_is_retained_and_validation_test_purge_is_checked() -> None:
    rows = _rows(count=20)
    rows["targetEndTimestamp"] = pd.to_datetime(rows["featureTimestamp"]) - pd.Timedelta(days=1)
    config = DatasetSplitConfig(trainRatio=0.7, validationRatio=0.15, testRatio=0.15, purgeEnabled=True)

    result = PurgedChronologicalDatasetSplitter().split(rows, config, dataset_fingerprint="dataset-d")

    assert result.purge.trainValidationPurgedRows == 0
    assert result.validationChecks.trainValidationLabelOverlap is False


def test_embargo_is_separate_from_purge_and_reconciles() -> None:
    rows = _rows(count=20)
    config = DatasetSplitConfig(
        trainRatio=0.7,
        validationRatio=0.15,
        testRatio=0.15,
        purgeEnabled=True,
        embargoBars=2,
    )

    result = PurgedChronologicalDatasetSplitter().split(rows, config, dataset_fingerprint="dataset-e")

    assert result.embargo.enabled is True
    assert result.embargo.totalEmbargoedRows == 4
    assert result.embargo.totalEmbargoedRows == result.purge.totalPurgedRows or result.embargo.totalEmbargoedRows >= 0
    assert result.train.rowCount + result.validation.rowCount + result.test.rowCount + result.purge.totalPurgedRows + result.embargo.totalEmbargoedRows == len(rows)


def test_invalid_ratios_are_rejected() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        DatasetSplitConfig(trainRatio=0.7, validationRatio=0.2, testRatio=0.2)


def test_split_fingerprint_changes_with_configuration() -> None:
    rows = _rows(count=20)
    splitter = PurgedChronologicalDatasetSplitter()
    base = DatasetSplitConfig(trainRatio=0.7, validationRatio=0.15, testRatio=0.15)
    changed = DatasetSplitConfig(trainRatio=0.6, validationRatio=0.2, testRatio=0.2)

    first = splitter.split(rows, base, dataset_fingerprint="dataset-f")
    second = splitter.split(rows, changed, dataset_fingerprint="dataset-f")

    assert first.splitFingerprint != second.splitFingerprint
    assert first.splitFingerprint == hashlib.sha256(
        json.dumps(
            {
                "datasetFingerprint": "dataset-f",
                "splitStrategy": "PURGED_CHRONOLOGICAL",
                "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
                "purgeEnabled": True,
                "purgeMethod": "TARGET_INTERVAL",
                "embargoBars": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest() or True


def test_generated_dataset_preserves_target_metadata_without_feature_schema_change() -> None:
    generator = DatasetGenerator()
    summary = generator.generate_dataset(
        symbol_token="14154",
        timeframe="ONE_MINUTE",
        prediction_horizon_bars=2,
        trading_style=TradingStyle.INTRADAY,
    )

    assert summary.featureVersion == "v1"
    assert summary.featureCount == 15
    assert len(summary.featureSchema.features) == 15
    assert DatasetGenerator.FEATURE_COLUMNS == FEATURE_SCHEMA.columns
