from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.dataset.dataset_generator import DatasetGenerator
from app.dataset.split.dataset_split_config import DatasetSplitConfig
from app.dataset.split.purged_chronological_dataset_splitter import PurgedChronologicalDatasetSplitter
from app.schemas.dataset_schema import (
    DatasetGenerationResponse,
    DatasetRequest,
    DatasetSplitRequest,
    DatasetSplitResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


@router.post("/dataset/generate", response_model=DatasetGenerationResponse)
def generate_dataset(payload: DatasetRequest) -> DatasetGenerationResponse:
    try:
        generator = DatasetGenerator()
        
        # Preserve the requested timeframe and bar horizon for provider routing and target generation.
        dataset_summary = generator.generate_dataset(symbol_token=payload.symbolToken,timeframe=payload.timeframe,prediction_horizon_bars=payload.predictionHorizonBars,
            buy_threshold_pct=payload.buyThresholdPct,sell_threshold_pct=payload.sellThresholdPct,trading_style=payload.tradingStyle,
            start_time=payload.startTime,end_time=payload.endTime,)
        
        return DatasetGenerationResponse(success=True,message="Training dataset generated",data=dataset_summary,)
    except ValueError as exc:
        logger.warning("Dataset generation validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive layer
        logger.exception("Unexpected dataset generation error")
        raise HTTPException(status_code=500, detail="Dataset generation failed") from exc


@router.post("/dataset/split", response_model=DatasetSplitResponse)
def split_dataset(payload: DatasetSplitRequest) -> DatasetSplitResponse:
    try:
        generator = DatasetGenerator()
        generation_summary = generator.generate_dataset(
            symbol_token=payload.symbolToken,
            timeframe=payload.timeframe,
            prediction_horizon_bars=payload.predictionHorizonBars,
            buy_threshold_pct=payload.buyThresholdPct,
            sell_threshold_pct=payload.sellThresholdPct,
            trading_style=payload.tradingStyle,
            start_time=payload.startTime,
            end_time=payload.endTime,
        )
        if generator.supervised_rows is None:
            raise ValueError("Dataset generator did not produce supervised rows")
        split_config = DatasetSplitConfig(
            trainRatio=payload.trainRatio,
            validationRatio=payload.validationRatio,
            testRatio=payload.testRatio,
            purgeEnabled=payload.purgeEnabled,
            embargoBars=payload.embargoBars,
        )
        splitter = PurgedChronologicalDatasetSplitter()
        split_result = splitter.split(
            generator.supervised_rows,
            split_config,
            dataset_fingerprint=generation_summary.datasetFingerprint or "",
            prediction_horizon_bars=payload.predictionHorizonBars,
            trading_style=payload.tradingStyle,
        )
        return DatasetSplitResponse(
            success=True,
            message="Dataset split completed",
            data=split_result.model_dump(mode="json"),
        )
    except ValueError as exc:
        logger.warning("Dataset split validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive layer
        logger.exception("Unexpected dataset split error")
        raise HTTPException(status_code=500, detail="Dataset split failed") from exc
