from __future__ import annotations

import pandas as pd

from app.features.candle_patterns import add_candle_pattern_features
from app.features.price_features import add_price_features
from app.features.technical_features import add_technical_features
from app.features.volume_features import add_volume_features


class FeatureEngineering:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the single reusable feature pipeline for both historical and live use."""
        result = df.copy()
        result = add_price_features(result)
        result = add_volume_features(result)
        result = add_technical_features(result)
        result = add_candle_pattern_features(result)
        return result
