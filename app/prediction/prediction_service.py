"""Live prediction feature preparation."""

from __future__ import annotations

import pandas as pd

from app.features.feature_engineering import FeatureEngineering


class PredictionService:
    """Build live prediction vectors through the same feature pipeline as training."""

    def __init__(self) -> None:
        self.feature_engineering = FeatureEngineering()

    def feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.feature_engineering.feature_matrix(df)

    def predict(self, df: pd.DataFrame):
        return self.feature_matrix(df)
