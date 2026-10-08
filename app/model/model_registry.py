from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.domain.trading_style import TradingStyle


def model_path_for(
    trading_style: TradingStyle | str,
    model_directory: str | Path,
) -> Path:
    """Return the style-specific model path."""
    normalized_style = TradingStyle.normalize(trading_style)
    filename = f"trade_ml_model_{normalized_style.value}.joblib"
    return Path(model_directory) / filename


def model_id_for(
    *,
    symbol_token: str,
    trading_style: TradingStyle | str,
    timeframe: str,
    prediction_horizon_bars: int,
    algorithm: str,
    feature_version: str,
    dataset_fingerprint: str,
    feature_count: int,
) -> str:
    normalized_style = TradingStyle.normalize(trading_style)
    payload = {
        "symbolToken": symbol_token,
        "tradingStyle": normalized_style.value,
        "timeframe": timeframe.upper(),
        "predictionHorizonBars": int(prediction_horizon_bars),
        "algorithm": algorithm,
        "featureVersion": feature_version,
        "datasetFingerprint": dataset_fingerprint,
        "featureCount": int(feature_count),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return (
        f"{symbol_token}_{normalized_style.value}_{timeframe.upper()}_"
        f"{int(prediction_horizon_bars)}_{algorithm}_{feature_version}_{digest}"
    )


def artifact_dir_for(
    *,
    symbol_token: str,
    trading_style: TradingStyle | str,
    timeframe: str,
    prediction_horizon_bars: int,
    algorithm: str,
    feature_version: str,
    dataset_fingerprint: str,
    feature_count: int,
    model_directory: str | Path,
) -> Path:
    model_id = model_id_for(
        symbol_token=symbol_token,
        trading_style=trading_style,
        timeframe=timeframe,
        prediction_horizon_bars=prediction_horizon_bars,
        algorithm=algorithm,
        feature_version=feature_version,
        dataset_fingerprint=dataset_fingerprint,
        feature_count=feature_count,
    )
    return Path(model_directory) / model_id
