from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.dataset.dataset_generator import DatasetGenerator
from app.schemas.dataset_schema import DatasetGenerationResponse, DatasetRequest

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
