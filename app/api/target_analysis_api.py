from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.dataset.target_analysis_service import TargetAnalysisService
from app.schemas.target_analysis_schema import TargetAnalysisRequest, TargetAnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


@router.post("/dataset/analyze-target", response_model=TargetAnalysisResponse)
def analyze_target(payload: TargetAnalysisRequest) -> TargetAnalysisResponse:
    try:
        data = TargetAnalysisService().analyze(
            symbol_token=payload.symbolToken,
            timeframe=payload.timeframe,
            prediction_horizons=payload.predictionHorizons,
            thresholds_pct=payload.thresholdsPct,
            start_time=payload.startTime,
            end_time=payload.endTime,
        )
        return TargetAnalysisResponse(
            success=True,
            message="Horizon-threshold target analysis completed",
            data=data,
        )
    except ValueError as exc:
        logger.warning("Target analysis validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive layer
        logger.exception("Unexpected target analysis error")
        raise HTTPException(status_code=500, detail="Target analysis failed") from exc