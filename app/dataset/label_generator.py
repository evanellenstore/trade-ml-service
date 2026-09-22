from __future__ import annotations

import pandas as pd


class LabelGenerator:
    """Generate future-return labels from the next N candles."""

    def calculate_future_returns(self, df: pd.DataFrame, prediction_horizon: int) -> pd.DataFrame:
        if prediction_horizon <= 0:
            raise ValueError("predictionHorizon must be greater than zero")

        result = df.copy()
        required_columns = {"symbol_token", "timeframe", "close"}
        if not required_columns.issubset(result.columns):
            raise ValueError("symbol_token, timeframe, and close columns are required")
        if "candle_time" in result.columns:
            result = result.sort_values(
                ["symbol_token", "timeframe", "candle_time"]
            ).reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)

        group_columns = ["symbol_token", "timeframe"]
        result["future_close"] = result.groupby(group_columns, dropna=False)["close"].shift(-prediction_horizon)
        result["future_return_pct"] = (
            (result["future_close"] - result["close"])
            / result["close"].replace(0, float("nan"))
        ) * 100.0
        return result

    def generate_labels(
        self,
        df: pd.DataFrame,
        prediction_horizon: int,
        buy_threshold_pct: float,
        sell_threshold_pct: float,
    ) -> pd.DataFrame:
        if prediction_horizon <= 0:
            raise ValueError("predictionHorizon must be greater than zero")
        if buy_threshold_pct <= 0:
            raise ValueError("buyThresholdPct must be positive")
        if sell_threshold_pct >= 0:
            raise ValueError("sellThresholdPct must be negative")

        result = self.calculate_future_returns(df, prediction_horizon)

        result["label"] = "HOLD"
        result.loc[result["future_return_pct"] >= buy_threshold_pct, "label"] = "BUY"
        result.loc[result["future_return_pct"] <= sell_threshold_pct, "label"] = "SELL"
        return result
