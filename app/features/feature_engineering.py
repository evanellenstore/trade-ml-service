from __future__ import annotations

import pandas as pd

from app.features.candle_patterns import add_candle_pattern_features
from app.features.feature_leakage import FeatureLeakageValidator
from app.features.feature_schema import FEATURE_SCHEMA
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

    def feature_matrix(
        self,
        df: pd.DataFrame,
        require_finite: bool = True,
    ) -> pd.DataFrame:
        """Return the exact canonical feature matrix used by training and prediction."""
        if all(column in df.columns for column in FEATURE_SCHEMA.columns):
            engineered = df.copy()
        else:
            engineered = self.transform(df)
        missing = [column for column in FEATURE_SCHEMA.columns if column not in engineered.columns]
        if missing:
            raise ValueError(f"Feature matrix missing columns: {missing}")
        matrix = engineered.loc[:, FEATURE_SCHEMA.columns].copy()
        if require_finite:
            invalid = matrix.isna().any(axis=1) | (
                matrix.eq(float("inf")) | matrix.eq(float("-inf"))
            ).any(axis=1)
            if invalid.any():
                raise ValueError(
                    f"Feature matrix contains invalid values in {int(invalid.sum())} rows"
                )
        validator = FeatureLeakageValidator().validate(matrix)
        if not validator["passed"]:
            raise ValueError(f"Feature matrix failed leakage validation: {validator}")
        return matrix
