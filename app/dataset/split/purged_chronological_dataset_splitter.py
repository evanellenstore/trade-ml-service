from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pandas as pd

from app.dataset.split.dataset_split_config import DatasetSplitConfig
from app.dataset.split.dataset_split_result import (
    DatasetSplitChecks,
    DatasetSplitEmbargo,
    DatasetSplitPurge,
    DatasetSplitResult,
    DatasetSplitSummary,
)
from app.domain.trading_style import TradingStyle
from app.features.feature_schema import FEATURE_SCHEMA


class PurgedChronologicalDatasetSplitter:
    """Deterministically partition validated rows without randomization."""

    def split(
        self,
        rows: pd.DataFrame,
        config: DatasetSplitConfig,
        *,
        dataset_fingerprint: str,
        prediction_horizon_bars: int | None = None,
        trading_style: TradingStyle | str = TradingStyle.INTRADAY,
    ) -> DatasetSplitResult:
        prepared = self._prepare_rows(rows, trading_style)
        if prepared.empty:
            raise ValueError("Dataset is empty for splitting")

        boundaries = self._determine_boundaries(prepared, config, trading_style)
        candidate_train = prepared.iloc[: boundaries["train_end"] + 1].copy()
        candidate_validation = prepared.iloc[
            boundaries["train_end"] + 1 : boundaries["validation_end"] + 1
        ].copy()
        candidate_test = prepared.iloc[boundaries["validation_end"] + 1 :].copy()

        purge = self._purge_target_overlap(
            candidate_train,
            candidate_validation,
            candidate_test,
            boundaries,
            config.purgeEnabled,
        )
        train = purge["train"]
        validation = purge["validation"]
        test = purge["test"]
        embargo = self._apply_embargo(
            train,
            validation,
            test,
            boundaries,
            config.embargoBars,
        )
        train = embargo["train"]
        validation = embargo["validation"]
        test = embargo["test"]

        validation_checks = self._validate(
            train,
            validation,
            test,
            purge,
            embargo,
            prepared,
            config,
        )
        fingerprints = self._fingerprint(
            dataset_fingerprint,
            config,
            boundaries,
            prediction_horizon_bars,
            trading_style,
        )
        summary_train = self._summary(train, "train", config)
        summary_validation = self._summary(validation, "validation", config)
        summary_test = self._summary(test, "test", config)

        return DatasetSplitResult(
            datasetFingerprint=dataset_fingerprint,
            splitFingerprint=fingerprints,
            inputRowCount=int(len(prepared)),
            strategy="PURGED_CHRONOLOGICAL",
            requestedRatios={
                "train": config.trainRatio,
                "validation": config.validationRatio,
                "test": config.testRatio,
            },
            predictionHorizonBars=prediction_horizon_bars,
            purge=DatasetSplitPurge(
                enabled=config.purgeEnabled,
                method="TARGET_INTERVAL",
                trainValidationPurgedRows=int(purge["trainValidationPurgedRows"]),
                validationTestPurgedRows=int(purge["validationTestPurgedRows"]),
                totalPurgedRows=int(purge["totalPurgedRows"]),
            ),
            embargo=DatasetSplitEmbargo(
                enabled=config.embargoBars > 0,
                bars=config.embargoBars,
                totalEmbargoedRows=int(embargo["totalEmbargoedRows"]),
            ),
            train=summary_train,
            validation=summary_validation,
            test=summary_test,
            validationChecks=validation_checks,
        )

    def split_frames(
        self,
        rows: pd.DataFrame,
        config: DatasetSplitConfig,
        *,
        dataset_fingerprint: str,
        prediction_horizon_bars: int | None = None,
        trading_style: TradingStyle | str = TradingStyle.INTRADAY,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return the actual chronological splits used by the training service."""
        prepared = self._prepare_rows(rows, trading_style)
        if prepared.empty:
            raise ValueError("Dataset is empty for splitting")

        boundaries = self._determine_boundaries(prepared, config, trading_style)
        candidate_train = prepared.iloc[: boundaries["train_end"] + 1].copy()
        candidate_validation = prepared.iloc[
            boundaries["train_end"] + 1 : boundaries["validation_end"] + 1
        ].copy()
        candidate_test = prepared.iloc[boundaries["validation_end"] + 1 :].copy()

        purge = self._purge_target_overlap(
            candidate_train,
            candidate_validation,
            candidate_test,
            boundaries,
            config.purgeEnabled,
        )
        embargo = self._apply_embargo(
            purge["train"],
            purge["validation"],
            purge["test"],
            boundaries,
            config.embargoBars,
        )
        return (
            embargo["train"].reset_index(drop=True),
            embargo["validation"].reset_index(drop=True),
            embargo["test"].reset_index(drop=True),
        )

    @staticmethod
    def _prepare_rows(rows: pd.DataFrame, trading_style: TradingStyle | str) -> pd.DataFrame:
        required = [
            "candle_id",
            "featureTimestamp",
            "targetEndTimestamp",
            "label",
            *FEATURE_SCHEMA.columns,
        ]
        missing = [column for column in required if column not in rows.columns]
        if missing:
            raise ValueError(f"Dataset is missing required split columns: {missing}")
        prepared = rows.copy()
        prepared["featureTimestamp"] = pd.to_datetime(prepared["featureTimestamp"], errors="raise")
        prepared["targetEndTimestamp"] = pd.to_datetime(
            prepared["targetEndTimestamp"], errors="raise"
        )
        prepared["candle_id"] = prepared["candle_id"].astype(str)
        prepared["label"] = prepared["label"].astype(str)
        prepared["trading_date"] = pd.to_datetime(prepared.get("trading_date", pd.Series(index=prepared.index, dtype="object"))).dt.date if "trading_date" in prepared.columns else pd.NA
        prepared = prepared.sort_values(
            ["featureTimestamp", "candle_id"], kind="mergesort"
        ).reset_index(drop=True)
        if prepared["featureTimestamp"].isna().any() or prepared["targetEndTimestamp"].isna().any():
            raise ValueError("Dataset contains missing target or feature timestamps")
        if not prepared["featureTimestamp"].is_monotonic_increasing:
            raise ValueError("Dataset must be chronologically ordered")
        if prepared["candle_id"].duplicated().any():
            raise ValueError("Dataset contains duplicate candle_id values")
        if prepared["label"].isin(["BUY", "HOLD", "SELL"]).all() is False:
            raise ValueError("Dataset contains a label outside BUY, HOLD, and SELL")
        return prepared

    @staticmethod
    def _determine_boundaries(
        rows: pd.DataFrame,
        config: DatasetSplitConfig,
        trading_style: TradingStyle | str,
    ) -> dict[str, int]:
        row_count = len(rows)
        train_cut = min(row_count - 1, max(0, math.floor(row_count * config.trainRatio)))
        validation_cut = min(
            row_count - 1,
            max(train_cut, math.floor(row_count * (config.trainRatio + config.validationRatio))),
        )
        if TradingStyle.normalize(trading_style) == TradingStyle.INTRADAY and "trading_date" in rows.columns:
            session_dates = rows["trading_date"].to_numpy()
            session_indices = pd.Series(session_dates).map(
                {date: position for position, date in enumerate(pd.unique(session_dates))}
            )
            session_indices = pd.Series(session_dates).map(
                {date: position for position, date in enumerate(pd.unique(session_dates))}
            )
            train_index = int(
                rows.index[session_indices.to_numpy() == session_indices.iloc[train_cut]].max()
            )
            validation_index = int(
                rows.index[session_indices.to_numpy() == session_indices.iloc[validation_cut]].max()
            )
            if validation_index <= train_index:
                validation_index = min(row_count - 1, train_index + 1)
        else:
            train_index = train_cut
            validation_index = validation_cut
        return {"train_end": train_index, "validation_end": validation_index}

    def _purge_target_overlap(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        boundaries: dict[str, int],
        purge_enabled: bool,
    ) -> dict[str, Any]:
        if not purge_enabled:
            return {
                "train": train,
                "validation": validation,
                "test": test,
                "trainValidationPurgedRows": 0,
                "validationTestPurgedRows": 0,
                "totalPurgedRows": 0,
            }
        validation_start = validation["featureTimestamp"].min() if not validation.empty else pd.NaT
        test_start = test["featureTimestamp"].min() if not test.empty else pd.NaT
        train_overlap = train["targetEndTimestamp"] >= validation_start
        validation_overlap = validation["targetEndTimestamp"] >= test_start
        train_purged = train.loc[train_overlap].copy()
        validation_purged = validation.loc[validation_overlap].copy()
        train_remaining = train.loc[~train_overlap].copy()
        validation_remaining = validation.loc[~validation_overlap].copy()
        return {
            "train": train_remaining,
            "validation": validation_remaining,
            "test": test,
            "trainValidationPurgedRows": int(len(train_purged)),
            "validationTestPurgedRows": int(len(validation_purged)),
            "totalPurgedRows": int(len(train_purged) + len(validation_purged)),
        }

    @staticmethod
    def _apply_embargo(
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        boundaries: dict[str, int],
        embargo_bars: int,
    ) -> dict[str, Any]:
        if embargo_bars <= 0:
            return {
                "train": train,
                "validation": validation,
                "test": test,
                "totalEmbargoedRows": 0,
            }
        validation_embargoed = validation.head(embargo_bars).copy()
        test_embargoed = test.head(embargo_bars).copy()
        validation_remaining = validation.iloc[embargo_bars:].copy()
        test_remaining = test.iloc[embargo_bars:].copy()
        return {
            "train": train,
            "validation": validation_remaining,
            "test": test_remaining,
            "totalEmbargoedRows": int(len(validation_embargoed) + len(test_embargoed)),
        }

    def _validate(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        purge: dict[str, Any],
        embargo: dict[str, Any],
        prepared: pd.DataFrame,
        config: DatasetSplitConfig,
    ) -> DatasetSplitChecks:
        all_splits = [train, validation, test]
        no_duplicates = all(
            not any(set(split["candle_id"]).intersection(other["candle_id"]) for other in all_splits if other is not split)
            for split in all_splits
        )
        train_overlap = bool(
            train["targetEndTimestamp"].ge(validation["featureTimestamp"].min()).any()
            if not validation.empty
            else False
        )
        validation_overlap = bool(
            validation["targetEndTimestamp"].ge(test["featureTimestamp"].min()).any()
            if not test.empty
            else False
        )
        feature_schema_valid = (
            list(FEATURE_SCHEMA.columns) == list(FEATURE_SCHEMA.columns)
            and all(column in prepared.columns for column in FEATURE_SCHEMA.columns)
            and all(
                prepared[column].notna().all()
                and prepared[column].map(math.isfinite).all()
                for column in FEATURE_SCHEMA.columns
            )
        )
        retained_count = len(train) + len(validation) + len(test)
        reconciliation = retained_count + purge["totalPurgedRows"] + embargo["totalEmbargoedRows"] == len(prepared)
        return DatasetSplitChecks(
            chronologicalOrderValid=all(split["featureTimestamp"].is_monotonic_increasing for split in all_splits),
            noDuplicateRows=bool(no_duplicates),
            trainValidationLabelOverlap=bool(train_overlap),
            validationTestLabelOverlap=bool(validation_overlap),
            featureSchemaValid=feature_schema_valid,
            rowReconciliationValid=bool(reconciliation),
        )

    @staticmethod
    def _summary(rows: pd.DataFrame, split_name: str, config: DatasetSplitConfig) -> DatasetSplitSummary:
        labels = ["BUY", "HOLD", "SELL"]
        distribution: dict[str, dict[str, int | float]] = {}
        row_count = len(rows)
        for label in labels:
            count = int((rows["label"] == label).sum())
            distribution[label] = {"count": count, "percentage": float(count / row_count * 100.0) if row_count else 0.0}
        reasons: list[str] = []
        minimum = {
            "train": config.minimumTrainRows,
            "validation": config.minimumValidationRows,
            "test": config.minimumTestRows,
        }[split_name]
        if row_count == 0:
            reasons.append("EMPTY_SPLIT")
        elif row_count < minimum:
            reasons.append("INSUFFICIENT_ROWS")
        missing_classes = [label for label in labels if label not in rows["label"].unique()]
        if missing_classes:
            reasons.append("MISSING_TARGET_CLASS")
        if rows[FEATURE_SCHEMA.columns].isna().any().any():
            reasons.append("INVALID_FEATURES")
        return DatasetSplitSummary(
            rowCount=row_count,
            startTime=rows["featureTimestamp"].min().isoformat() if not rows.empty else None,
            endTime=rows["featureTimestamp"].max().isoformat() if not rows.empty else None,
            featureCount=len(FEATURE_SCHEMA.columns),
            featureSchemaValid=True,
            features=list(FEATURE_SCHEMA.columns),
            labelDistribution=distribution,
            eligible=not reasons,
            reasons=reasons,
        )

    @staticmethod
    def _fingerprint(
        dataset_fingerprint: str,
        config: DatasetSplitConfig,
        boundaries: dict[str, int],
        prediction_horizon_bars: int | None,
        trading_style: TradingStyle | str,
    ) -> str:
        canonical = {
            "datasetFingerprint": dataset_fingerprint,
            "splitStrategy": "PURGED_CHRONOLOGICAL",
            "ratios": {
                "train": config.trainRatio,
                "validation": config.validationRatio,
                "test": config.testRatio,
            },
            "boundaries": boundaries,
            "tradingStyle": TradingStyle.normalize(trading_style).value,
            "predictionHorizonBars": prediction_horizon_bars,
            "purgeEnabled": config.purgeEnabled,
            "purgeMethod": "TARGET_INTERVAL",
            "embargoBars": config.embargoBars,
        }
        serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
