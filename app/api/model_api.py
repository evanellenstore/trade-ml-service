from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.prediction.prediction_service import PredictionService
from app.schemas.model_schema import ModelTrainRequest, ModelTrainingResponse, PredictionRequest
from app.training.model_training_service import ModelTrainingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


@router.post("/model/train", response_model=ModelTrainingResponse)
def train_model(payload: ModelTrainRequest) -> ModelTrainingResponse:
    try:
        result = ModelTrainingService().train(
            symbol_token=payload.symbolToken,
            timeframe=payload.timeframe,
            prediction_horizon_bars=payload.predictionHorizonBars,
            buy_threshold_pct=payload.buyThresholdPct,
            sell_threshold_pct=payload.sellThresholdPct,
            trading_style=payload.tradingStyle,
            start_time=payload.startTime,
            end_time=payload.endTime,
            train_ratio=payload.trainRatio,
            validation_ratio=payload.validationRatio,
            test_ratio=payload.testRatio,
            purge_enabled=payload.purgeEnabled,
            embargo_bars=payload.embargoBars,
            validation_strategy=payload.validationStrategy,
            walk_forward=payload.walkForward,
            oof_analysis=payload.oofAnalysis,
        )
        
        print(f"---------- Model training result: {result}")  # Debugging line to print the result
        return ModelTrainingResponse(
            success=True,
            message="Model trained and persisted",
            data=result,
        )
    except ValueError as exc:
        logger.warning("Model training validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Model training failed")
        raise HTTPException(status_code=500, detail="Model training failed") from exc


@router.post("/prediction")
def predict(payload: PredictionRequest) -> dict:
    try:
        service = PredictionService(trading_style=payload.tradingStyle)
        return service.predict_market_data(
            symbol_token=payload.symbolToken,
            timeframe=payload.timeframe,
            prediction_horizon_bars=payload.predictionHorizonBars,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
