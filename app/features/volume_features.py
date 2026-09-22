from __future__ import annotations

import numpy as np
import pandas as pd


def add_volume_features(df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    """Adds volume-derived features using only current and historical data."""
    result = df.copy()
    if result.empty:
        result["volume_change_pct"] = pd.Series(dtype=float)
        result["rolling_volume_mean"] = pd.Series(dtype=float)
        result["relative_volume"] = pd.Series(dtype=float)
        return result

    volume = result["volume"].replace(0, np.nan)
    result["volume_change_pct"] = result.groupby(["symbol_token", "timeframe"], dropna=False)["volume"].pct_change(1).fillna(0.0)
    result["rolling_volume_mean"] = (
        result.groupby(["symbol_token", "timeframe"], dropna=False)["volume"]
        .transform(lambda s: s.rolling(window=rolling_window, min_periods=1).mean())
    )
    result["relative_volume"] = volume / result["rolling_volume_mean"].replace(0, np.nan)
    result["volume_change_pct"] = result["volume_change_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["relative_volume"] = result["relative_volume"].replace([np.inf, -np.inf], np.nan)
    return result
