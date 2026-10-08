from __future__ import annotations

from typing import Any

import pandas as pd

from app.features.feature_schema import FEATURE_SCHEMA


class FeatureLeakageValidator:
    """Validate that a prediction matrix contains only canonical historical features."""

    TARGET_COLUMNS = {
        "future_close",
        "future_return_pct",
        "label",
        "target",
        "featureTimestamp",
        "targetEndTimestamp",
        "candle_id",
        "symbol_token",
        "timeframe",
        "candle_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    FUTURE_PREFIXES = ("future_", "next_", "predicted_", "forecast_")

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        feature_columns = list(FEATURE_SCHEMA.columns)
        available_columns = list(df.columns)
        target_columns_in_features = sorted(
            column for column in available_columns if column in self.TARGET_COLUMNS
        )
        future_feature_columns = sorted(
            column
            for column in available_columns
            if column.startswith(self.FUTURE_PREFIXES)
        )
        schema_order_valid = all(
            column in available_columns for column in feature_columns
        ) and [column for column in available_columns if column in feature_columns] == feature_columns
        feature_count_valid = len(available_columns) == len(feature_columns)
        passed = (
            not target_columns_in_features
            and not future_feature_columns
            and schema_order_valid
            and feature_count_valid
        )
        return {
            "passed": passed,
            "futureFeatureColumnsDetected": future_feature_columns,
            "targetColumnsInFeatures": target_columns_in_features,
            "schemaOrderValid": schema_order_valid,
            "featureCountValid": feature_count_valid,
        }
