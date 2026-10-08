"""Live prediction feature preparation and model inference."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config.settings import settings
from app.domain.trading_style import TradingStyle
from app.features.feature_engineering import FeatureEngineering
from app.features.feature_schema import FEATURE_SCHEMA
from app.market_data.market_data_provider import MarketDataProvider
from app.model.model_registry import model_path_for


class PredictionService:
    """Build live prediction vectors through the same feature pipeline as training."""

    def __init__(
        self,
        model_path: str | None = None,
        provider: MarketDataProvider | None = None,
        trading_style: TradingStyle | str = TradingStyle.INTRADAY,
    ) -> None:
        if model_path is None:
            model_path = model_path_for(
                trading_style,
                Path(settings.model_directory),
            )
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path) if self.model_path.exists() else None
        self.feature_engineering = FeatureEngineering()
        self.provider = provider or MarketDataProvider()

    def feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.feature_engineering.feature_matrix(df)

    def _decode_prediction(self, prediction: int | str) -> str:
        if hasattr(self.model, "label_encoder"):
            return str(self.model.label_encoder.inverse_transform([prediction])[0])
        if hasattr(self.model, "class_labels_"):
            labels = getattr(self.model, "class_labels_")
            if isinstance(labels, (list, tuple, pd.Index)):
                return str(labels[int(prediction)])
            if hasattr(labels, "__len__") and len(labels) > 0:
                return str(labels[int(prediction)])
        if hasattr(self.model, "classes_"):
            classes = getattr(self.model, "classes_")
            if hasattr(classes, "__len__") and len(classes) > 0:
                return str(classes[int(prediction)])
        return str(prediction)

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        if self.model is None:
            raise FileNotFoundError(f"Trained model not found: {self.model_path}")
        matrix = self.feature_matrix(df)
        probabilities = self.model.predict_proba(matrix)[0]
        prediction = self._decode_prediction(self.model.predict(matrix)[0])
        return {
            "prediction": prediction,
            "probability": float(max(probabilities)),
            "probabilities": {
                self._decode_prediction(int(label)): float(value)
                for label, value in zip(self.model.classes_, probabilities)
            },
            "feature_columns": list(FEATURE_SCHEMA.columns),
        }

    def predict_market_data(
        self,
        *,
        symbol_token: str,
        timeframe: str,
        prediction_horizon_bars: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise FileNotFoundError(f"Trained model not found: {self.model_path}")
        market_data = self.provider.get_market_data(
            symbol_token,
            timeframe,
            start_time,
            end_time,
        )
        if market_data.data.empty:
            raise ValueError("No market data found for live prediction")

        latest = market_data.data.sort_values("candle_time").iloc[[-1]].copy()
        feature_matrix = self.feature_matrix(latest)
        probabilities = self.model.predict_proba(feature_matrix)[0]
        prediction = self._decode_prediction(self.model.predict(feature_matrix)[0])
        return {
            "prediction": prediction,
            "probability": float(max(probabilities)),
            "probabilities": {
                self._decode_prediction(int(label)): float(value)
                for label, value in zip(self.model.classes_, probabilities)
            },
            "feature_columns": list(FEATURE_SCHEMA.columns),
            "candle_time": latest["candle_time"].iloc[0].isoformat(),
            "prediction_horizon_bars": prediction_horizon_bars,
        }
