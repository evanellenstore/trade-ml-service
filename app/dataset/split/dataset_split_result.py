from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class DatasetSplitSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rowCount: int
    startTime: Optional[str]
    endTime: Optional[str]
    featureCount: int
    featureSchemaValid: bool
    features: List[str]
    labelDistribution: Dict[str, Dict[str, object]]
    eligible: bool
    reasons: List[str]


class DatasetSplitChecks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chronologicalOrderValid: bool
    noDuplicateRows: bool
    trainValidationLabelOverlap: bool
    validationTestLabelOverlap: bool
    featureSchemaValid: bool
    rowReconciliationValid: bool


class DatasetSplitPurge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    method: str
    trainValidationPurgedRows: int
    validationTestPurgedRows: int
    totalPurgedRows: int


class DatasetSplitEmbargo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    bars: int
    totalEmbargoedRows: int


class DatasetSplitResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasetFingerprint: str
    splitFingerprint: str
    inputRowCount: int
    strategy: str
    requestedRatios: dict[str, float]
    predictionHorizonBars: Optional[int] = None
    purge: DatasetSplitPurge
    embargo: DatasetSplitEmbargo
    train: DatasetSplitSummary
    validation: DatasetSplitSummary
    test: DatasetSplitSummary
    validationChecks: DatasetSplitChecks
